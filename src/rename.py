"""Speaker-renaming subsystem.

Owns: example building, persistence, post-hoc rewrites, cross-platform
non-blocking input, ffplay playback, and the interactive rename UI loop.
The rest of the app calls into here; this module knows nothing about the
transcription pipeline beyond the JSON segment shape.
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_DEFAULT_SPEAKER_RE = re.compile(r"^SPEAKER_\d+$")

PERSIST_VERSION = 1


# ─── Raw keyboard input ───────────────────────────────────────────────────────
# Semantic key tokens returned by decode_key/read_key. Printable characters are
# returned as the character itself (e.g. "a", " "); everything else is one of:
KEY_LEFT = "LEFT"
KEY_RIGHT = "RIGHT"
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_ENTER = "ENTER"
KEY_BACKSPACE = "BACKSPACE"
KEY_ESC = "ESC"
KEY_EOF = "EOF"          # stdin closed — caller should abort
KEY_UNKNOWN = "UNKNOWN"  # an unrecognised escape/function key — caller ignores

_WIN_ARROWS = {"H": KEY_UP, "P": KEY_DOWN, "K": KEY_LEFT, "M": KEY_RIGHT}
_POSIX_ARROWS = {"A": KEY_UP, "B": KEY_DOWN, "C": KEY_RIGHT, "D": KEY_LEFT}


def decode_key(getch: Callable[[], str]) -> str:
    """Decode one logical keypress from a raw character source.

    `getch()` returns the next raw character, or "" when no more input is
    available. Split out from read_key() so the (fiddly, cross-platform)
    decoding logic is unit-testable without a real terminal.
    """
    ch = getch()
    if ch == "":
        return KEY_EOF
    if ch == "\x03":                       # Ctrl-C
        raise KeyboardInterrupt
    if ch in ("\x00", "\xe0"):             # Windows function/arrow prefix
        return _WIN_ARROWS.get(getch(), KEY_UNKNOWN)
    if ch == "\x1b":                       # ESC — bare, or start of a sequence
        nxt = getch()
        if nxt == "[":
            return _POSIX_ARROWS.get(getch(), KEY_UNKNOWN)
        return KEY_ESC
    if ch in ("\r", "\n"):
        return KEY_ENTER
    if ch in ("\x7f", "\x08"):
        return KEY_BACKSPACE
    return ch                              # printable (incl. " ")


def read_key(_getch: Callable[[], str] | None = None) -> str:
    """Block for one logical keypress and return a key token (KEY_* constant) or
    a printable character. Raises KeyboardInterrupt on Ctrl-C; returns KEY_EOF on
    closed stdin. Assumes the terminal is already in cbreak (see raw_mode()).

    `_getch` is a test seam; production callers pass nothing.
    """
    if _getch is not None:
        return decode_key(_getch)

    if sys.platform == "win32":
        import msvcrt

        prev: str | None = None

        def getch() -> str:
            nonlocal prev
            if prev is None:
                prev = msvcrt.getwch()        # blocking — wait for the first key
                return prev
            if prev in ("\x00", "\xe0"):
                # The trail code of a function/arrow key sits in the CRT's
                # internal pushback buffer, which kbhit() can NOT see — it
                # only checks the console event queue (verified on a real
                # console: kbhit() stays False while a blocking getwch()
                # returns the code instantly). Gating this read on kbhit()
                # makes the trail byte leak into the next read and get typed
                # as a letter. The prefix is never delivered without its
                # trail, so a blocking read is safe.
                prev = msvcrt.getwch()
                return prev
            # Follow-up of an ESC-style sequence (VT input mode): genuinely
            # new console events, which kbhit() does see — poll briefly,
            # otherwise it was a lone ESC.
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    prev = msvcrt.getwch()
                    return prev
                time.sleep(0.002)
            return ""                          # genuinely nothing more (lone ESC)

        return decode_key(getch)

    # POSIX: first byte blocking, follow-up bytes (for ESC sequences) opportunistic.
    import select

    first = True

    def getch() -> str:
        nonlocal first
        if first:
            first = False
            return sys.stdin.read(1)
        r, _, _ = select.select([sys.stdin], [], [], 0.05)
        return sys.stdin.read(1) if r else ""

    return decode_key(getch)


@contextlib.contextmanager
def raw_mode():
    """Put the terminal into cbreak so single keys read immediately (POSIX).
    No-op on Windows (msvcrt reads raw) and when stdin isn't a TTY. Restores
    terminal settings on exit."""
    if sys.platform == "win32" or not sys.stdin.isatty():
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@dataclass
class Snippet:
    text: str
    start: float
    end: float


@dataclass
class SpeakerExample:
    label: str
    total_seconds: float
    segment_count: int
    snippets: list[Snippet] = field(default_factory=list)


# ─── Sample selection (speaker-pure, ~3-6s) ──────────────────────────────────
_SAMPLE_MIN_S = 1.2        # drop runs shorter than this (a one-word "Yeah.")
_SAMPLE_MAX_S = 6.0        # trim a clean run to about this many seconds
_SAMPLES_PER_SPEAKER = 6   # samples offered per speaker (flip with ← →)
_RUN_GAP_S = 1.0           # a silence this long inside a run = someone else's turn
_MIN_CHARS_PER_S = 3.0     # below this the timing is an alignment artifact


def _speaker_pure_runs(words: list[dict], label: str) -> list[list[dict]]:
    """Split `words` into maximal contiguous runs spoken by `label`. A word
    tagged as a different speaker, one missing timing, or a time gap of more
    than _RUN_GAP_S since the previous word (untranscribed audio — usually
    another speaker or dead air) ends the current run; untagged words continue
    it (alignment occasionally drops the tag). Returns [] when there's no
    usable word-level timing."""
    runs: list[list[dict]] = []
    cur: list[dict] = []
    for word in words:
        wspk = word.get("speaker")
        if word.get("start") is None or word.get("end") is None or (
            wspk is not None and wspk != label
        ):
            if cur:
                runs.append(cur)
                cur = []
            continue
        if cur and float(word["start"]) - float(cur[-1]["end"]) > _RUN_GAP_S:
            runs.append(cur)
            cur = []
        cur.append(word)
    if cur:
        runs.append(cur)
    return runs


def _snippet_from_run(run: list[dict]) -> "Snippet | None":
    """Build a Snippet from a word run, trimmed to ~_SAMPLE_MAX_S from its start
    so the played window stays short and matches the shown text."""
    t0 = float(run[0]["start"])
    kept: list[dict] = []
    for word in run:
        kept.append(word)
        if float(word["end"]) - t0 >= _SAMPLE_MAX_S:
            break
    text = " ".join((w.get("word") or "").strip() for w in kept).strip()
    if not text:
        return None
    return Snippet(text=text, start=float(kept[0]["start"]), end=float(kept[-1]["end"]))


def _duration_band(dur: float) -> float:
    """0.0 inside the 3-6s sweet spot, growing with the distance outside it."""
    if dur < 3.0:
        return 3.0 - dur
    if dur > _SAMPLE_MAX_S:
        return dur - _SAMPLE_MAX_S
    return 0.0


def _sample_rank(sn: "Snippet") -> tuple[float, int]:
    """Sort key: prefer samples inside the 3-6s sweet spot, then longer text."""
    return (_duration_band(sn.end - sn.start), -len(sn.text))


def _trimmed_segment_snippet(s: dict) -> "Snippet | None":
    """Whole-segment snippet for the no-word-timing fallback. Long segments
    are trimmed to _SAMPLE_MAX_S with the text cut proportionally at a word
    boundary (marked with an ellipsis), so the shown quote roughly matches the
    window that will be played instead of dwarfing it."""
    text = (s.get("text") or "").strip()
    if not text:
        return None
    start = float(s.get("start", 0))
    end = float(s.get("end", 0))
    dur = end - start
    if dur <= _SAMPLE_MAX_S:
        return Snippet(text=text, start=start, end=end)
    keep = max(1, int(len(text) * (_SAMPLE_MAX_S / dur)))
    cut = text.rfind(" ", 0, keep + 1)
    if cut <= 0:
        cut = keep
    return Snippet(text=text[:cut].rstrip() + " …", start=start, end=start + _SAMPLE_MAX_S)


def _build_speaker_snippets(segs: list[dict], label: str) -> list[Snippet]:
    """Preferred: speaker-pure, trimmed samples from word-level timing, ranked
    toward the ~3-6s sweet spot. Falls back to trimmed per-segment samples when
    no word timing exists (e.g. alignment was unavailable), preferring segments
    already near the sweet spot — a short natural segment beats a trimmed 30s
    blob, which is likelier to contain other speakers."""
    pure: list[Snippet] = []
    for s in segs:
        for run in _speaker_pure_runs(s.get("words") or [], label):
            sn = _snippet_from_run(run)
            if not sn:
                continue
            dur = sn.end - sn.start
            if dur < _SAMPLE_MIN_S:
                continue
            if len(sn.text) / dur < _MIN_CHARS_PER_S:
                continue  # e.g. one word "aligned" across 9s of other audio
            pure.append(sn)

    if pure:
        pure.sort(key=_sample_rank)
        return pure[:_SAMPLES_PER_SPEAKER]

    ranked: list[tuple[float, Snippet]] = []
    for s in segs:
        sn = _trimmed_segment_snippet(s)
        if sn is None:
            continue
        # Rank by the segment's ORIGINAL duration: a naturally short segment
        # outranks a long one we had to trim, even though both end up ~6s.
        ranked.append((float(s.get("end", 0)) - float(s.get("start", 0)), sn))
    ranked.sort(key=lambda t: (_duration_band(t[0]), -len(t[1].text)))
    return [sn for _, sn in ranked[:_SAMPLES_PER_SPEAKER]]


def build_speaker_examples(segments: list[dict]) -> list[SpeakerExample]:
    """Group segments by speaker. Return examples sorted by total speaking time
    (descending). Each example's `snippets` are speaker-pure samples trimmed to
    a ~3-6s window (so played audio matches the shown text and contains only the
    target speaker); without word-level timing they fall back to per-segment
    samples trimmed to the same window, preferring segments already near it.
    Segments without a `speaker` field are skipped.
    """
    grouped: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        grouped.setdefault(spk, []).append(seg)

    examples: list[SpeakerExample] = []
    for label, segs in grouped.items():
        total = sum(float(s.get("end", 0)) - float(s.get("start", 0)) for s in segs)
        examples.append(SpeakerExample(
            label=label,
            total_seconds=total,
            segment_count=len(segs),
            snippets=_build_speaker_snippets(segs, label),
        ))

    examples.sort(key=lambda e: e.total_seconds, reverse=True)
    return examples


def apply_speaker_mapping(segments: list[dict], mapping: dict[str, str]) -> None:
    """Rename speakers in-place across segments and any nested word-level
    `words[].speaker` entries (which whisperx populates via
    `assign_word_speakers`).
    """
    if not mapping:
        return
    for seg in segments:
        spk = seg.get("speaker")
        if spk in mapping:
            seg["speaker"] = mapping[spk]
        for w in seg.get("words", []) or []:
            wspk = w.get("speaker")
            if wspk in mapping:
                w["speaker"] = mapping[wspk]


def unmapped_speakers(segments: list[dict]) -> list[str]:
    """Return distinct speaker labels that still match the default
    `SPEAKER_\\d+` pattern. Order is insertion order (deterministic)."""
    seen: dict[str, None] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if isinstance(spk, str) and _DEFAULT_SPEAKER_RE.match(spk):
            seen.setdefault(spk, None)
    return list(seen.keys())


# ─── Unassigned (SPEAKER_??) lines ────────────────────────────────────────────
# whisperx leaves a segment with no `speaker` when diarization attributed it to
# no cluster; write_outputs renders those as SPEAKER_??. These helpers find such
# gaps, group adjacent ones, and let the user fold each run into a known speaker.

@dataclass
class UnassignedRun:
    """A contiguous run of no-speaker segments, grouped for one decision."""
    indices: list[int]
    start: float
    end: float
    text: str
    line_count: int


def unassigned_segments(segments: list[dict]) -> list[int]:
    """Indices of segments with no usable `speaker` (None/missing/empty) — the
    lines rendered as SPEAKER_?? in the outputs."""
    return [i for i, seg in enumerate(segments) if not seg.get("speaker")]


def group_contiguous(indices: list[int]) -> list[list[int]]:
    """Split a sorted index list into runs of consecutive integers."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def build_unassigned_runs(segments: list[dict]) -> list[UnassignedRun]:
    """Group the SPEAKER_?? segments into contiguous runs, each carrying its
    combined text and time span (for display + audio playback)."""
    runs: list[UnassignedRun] = []
    for group in group_contiguous(unassigned_segments(segments)):
        segs = [segments[i] for i in group]
        text = " ".join((s.get("text") or "").strip() for s in segs).strip()
        runs.append(UnassignedRun(
            indices=group,
            start=float(segs[0].get("start", 0.0)),
            end=float(segs[-1].get("end", 0.0)),
            text=text,
            line_count=len(group),
        ))
    return runs


def candidate_speakers(segments: list[dict]) -> list[str]:
    """Distinct current speaker labels (named or SPEAKER_XX) a ?? run could be
    assigned to, ordered by total speaking time descending."""
    totals: dict[str, float] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        totals[spk] = totals.get(spk, 0.0) + (
            float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        )
    return sorted(totals, key=lambda s: totals[s], reverse=True)


def assign_segments_speaker(segments: list[dict], indices: list[int], name: str) -> None:
    """Set `speaker = name` on the given segments and their nested words."""
    for i in indices:
        seg = segments[i]
        seg["speaker"] = name
        for w in seg.get("words", []) or []:
            w["speaker"] = name


def _persist_path(transcript_dir: Path, stem: str) -> Path:
    return transcript_dir / f"{stem}.speakers.json"


def load_persisted_mapping(transcript_dir: Path, stem: str) -> dict[str, str] | None:
    """Read `<stem>.speakers.json` from `transcript_dir`. Returns the mapping,
    or None if the file is missing/unreadable/wrong-version.
    """
    path = _persist_path(transcript_dir, stem)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not read persisted mapping %s: %s", path, e)
        return None
    if data.get("version") != PERSIST_VERSION:
        log.warning("Unknown persisted-mapping version in %s; ignoring.", path)
        return None
    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        return None
    return {str(k): str(v) for k, v in mapping.items()}


def save_persisted_mapping(
    transcript_dir: Path,
    stem: str,
    audio_filename: str,
    mapping: dict[str, str],
) -> None:
    """Write `<stem>.speakers.json` next to the transcript outputs."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = _persist_path(transcript_dir, stem)
    payload = {
        "version": PERSIST_VERSION,
        "audio_filename": audio_filename,
        "mapping": dict(mapping),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RenamePending:
    transcript_json: Path
    audio_path: Path | None
    unmapped: list[str]
    unassigned_count: int = 0


def _find_audio_for_stem(incoming_dir: Path, stem: str) -> Path | None:
    # Local import to avoid pulling whisperx into the import graph for tests
    # that only exercise pure-data helpers.
    from src.transcribe import AUDIO_EXTS

    if not incoming_dir.exists():
        return None
    for candidate in incoming_dir.rglob(f"{stem}.*"):
        if candidate.suffix.lower() in AUDIO_EXTS:
            return candidate
    return None


def rewrite_outputs_with_mapping(
    transcript_json: Path,
    mapping: dict[str, str],
) -> None:
    """Load a transcript JSON, apply the mapping, and re-emit
    `.json`, `.txt`, and `.srt` next to it. Used for post-hoc renames
    (post-batch catch-up and the startup `r` option).
    """
    # Local import to keep rename.py importable without forcing whisperx into
    # the module graph for unrelated tests.
    from src.transcribe import write_outputs

    data = json.loads(transcript_json.read_text(encoding="utf-8"))
    apply_speaker_mapping(data.get("segments") or [], mapping)
    out_base = transcript_json.with_suffix("")
    write_outputs(data, out_base)


def ffplay_available() -> bool:
    """True iff `ffplay` is resolvable on PATH."""
    return shutil.which("ffplay") is not None


def play_snippet(
    audio_path: Path, start: float, duration: float = 10.0
) -> "subprocess.Popen | None":
    """Start playing `[start, start+duration]` of `audio_path` via ffplay.

    NON-BLOCKING: spawns ffplay and returns the process handle immediately so the
    rename loop keeps reading keys and updating the display while audio plays.
    `stdin` is detached (DEVNULL) so ffplay can't swallow the keystrokes we're
    reading. Returns None if ffplay can't be launched (errors are swallowed so
    the UI never crashes mid-rename).
    """
    cmd = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-hide_banner",
        "-loglevel", "quiet",
        "-ss", f"{start}",
        "-t", f"{duration}",
        str(audio_path),
    ]
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError) as e:
        log.debug("ffplay invocation failed: %s", e)
        return None


def wait_for_keypress(timeout: float) -> str | None:
    """Non-blocking single-key read with timeout. Returns the first character
    pressed (or first byte of a multi-byte sequence — sufficient for our
    "any key engages" semantics), or None if the deadline passes.

    Windows: msvcrt.kbhit + getwch.
    POSIX:   select+termios (sets stdin to cbreak, restores on exit).
    """
    if timeout <= 0:
        return None
    deadline = time.monotonic() + timeout

    if sys.platform == "win32":
        import msvcrt
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                # Function/arrow keys produce a two-char sequence; the trail
                # code sits in the CRT pushback buffer, which kbhit() cannot
                # see — consume it with an unconditional blocking read so it
                # can't leak into the next prompt as a typed letter.
                if ch in ("\x00", "\xe0"):
                    msvcrt.getwch()
                return ch
            time.sleep(0.03)
        return None

    # POSIX path
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            r, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.1))
            if r:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    # Drain the escape-sequence tail (arrows = ESC [ A) so the
                    # stray bytes don't leak into the next reader as letters.
                    for _ in range(8):
                        if not select.select([sys.stdin], [], [], 0.01)[0]:
                            break
                        sys.stdin.read(1)
                return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


PLAY = "PLAY"  # effect token returned by RenameNav.step


@dataclass
class RenameNav:
    """Pure state machine for the keyboard-native rename loop.

    Feed it key tokens via step(); it owns the current speaker, the per-speaker
    sample cursor, the live name buffer, and the accumulated mapping. step()
    returns an effect token (currently only PLAY) or None. No terminal I/O —
    that lives in interactive_rename.
    """
    examples: list[SpeakerExample]
    speaker_idx: int = 0
    sample_idxs: list[int] = field(default_factory=list)
    name_buffer: str = ""
    mapping: dict[str, str] = field(default_factory=dict)
    finished: bool = False
    aborted: bool = False

    def __post_init__(self) -> None:
        if not self.sample_idxs:
            self.sample_idxs = [0] * len(self.examples)

    @property
    def current(self) -> "SpeakerExample | None":
        if 0 <= self.speaker_idx < len(self.examples):
            return self.examples[self.speaker_idx]
        return None

    @property
    def sample_idx(self) -> int:
        if not self.examples:
            return 0
        return self.sample_idxs[self.speaker_idx]

    def current_snippet(self) -> "Snippet | None":
        ex = self.current
        if not ex or not ex.snippets:
            return None
        return ex.snippets[self.sample_idx % len(ex.snippets)]

    def _move_sample(self, delta: int) -> None:
        ex = self.current
        if not ex or not ex.snippets:
            return
        self.sample_idxs[self.speaker_idx] = (self.sample_idx + delta) % len(ex.snippets)

    def _commit_and_advance(self) -> None:
        ex = self.current
        name = self.name_buffer.strip()
        if ex and name:
            self.mapping[ex.label] = name
        self.name_buffer = ""
        self.speaker_idx += 1
        if self.speaker_idx >= len(self.examples):
            self.finished = True

    def step(self, key: str) -> str | None:
        """Advance state by one key token. Returns PLAY or None."""
        if self.finished or self.aborted:
            return None

        # Arrows flip samples in BOTH modes (they're never text).
        if key == KEY_LEFT:
            self._move_sample(-1)
            return None
        if key == KEY_RIGHT:
            self._move_sample(+1)
            return None

        if key == KEY_EOF:
            self.aborted = True
            return None

        if not self.name_buffer:  # ── command mode ──
            if key == KEY_UP:
                self.speaker_idx = max(0, self.speaker_idx - 1)
            elif key == KEY_DOWN:
                self.speaker_idx = min(len(self.examples) - 1, self.speaker_idx + 1)
            elif key == " ":
                return PLAY
            elif key == KEY_ENTER:
                self._commit_and_advance()   # empty buffer -> keep label, advance
            elif key == KEY_ESC:
                self.finished = True          # finish & save partial
            elif len(key) == 1 and key.isprintable():
                self.name_buffer = key        # start typing
            return None

        # ── typing mode ──
        if key == KEY_ENTER:
            self._commit_and_advance()
        elif key == KEY_ESC:
            self.name_buffer = ""             # cancel name, stay on speaker
        elif key == KEY_BACKSPACE:
            self.name_buffer = self.name_buffer[:-1]
        elif key in (KEY_UP, KEY_DOWN):
            pass                              # ignored while typing
        elif len(key) == 1 and key.isprintable():
            self.name_buffer += key
        return None


@dataclass
class UnassignedNav:
    """Pure state machine for assigning SPEAKER_?? runs to a known speaker.

    Unlike RenameNav (where you *type* a new name per cluster), here you mostly
    *pick* from an existing list, so command-mode digits map to candidates.
    Modes: 'command' (default), 'typing' (after `n`, type a fresh name),
    'pick_all' (after `a`, the next digit applies to every remaining run).
    step() returns PLAY or None; no terminal I/O.
    """
    runs: list[UnassignedRun]
    candidates: list[str]
    run_idx: int = 0
    decisions: dict[int, str] = field(default_factory=dict)
    name_buffer: str = ""
    mode: str = "command"
    finished: bool = False
    aborted: bool = False

    def __post_init__(self) -> None:
        if not self.runs:
            self.finished = True

    @property
    def current(self) -> "UnassignedRun | None":
        if 0 <= self.run_idx < len(self.runs):
            return self.runs[self.run_idx]
        return None

    def _advance(self) -> None:
        self.run_idx += 1
        if self.run_idx >= len(self.runs):
            self.finished = True

    def _assign_current(self, name: str) -> None:
        self.decisions[self.run_idx] = name
        self._advance()

    def step(self, key: str) -> "str | None":
        if self.finished or self.aborted:
            return None

        if key == KEY_EOF:
            self.aborted = True
            return None

        if self.mode == "typing":
            if key == KEY_ENTER:
                name = self.name_buffer.strip()
                self.name_buffer = ""
                self.mode = "command"
                if name:
                    self._assign_current(name)
            elif key == KEY_ESC:
                self.name_buffer = ""
                self.mode = "command"
            elif key == KEY_BACKSPACE:
                self.name_buffer = self.name_buffer[:-1]
            elif len(key) == 1 and key.isprintable():
                self.name_buffer += key
            return None

        if self.mode == "pick_all":
            if key == KEY_ESC:
                self.mode = "command"
            elif len(key) == 1 and key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(self.candidates):
                    name = self.candidates[idx]
                    for j in range(self.run_idx, len(self.runs)):
                        self.decisions.setdefault(j, name)
                    self.mode = "command"
                    self.run_idx = len(self.runs)
                    self.finished = True
            return None

        # ── command mode ──
        if key == " ":
            return PLAY
        if key == KEY_UP:
            self.run_idx = max(0, self.run_idx - 1)
        elif key == KEY_DOWN:
            self.run_idx = min(len(self.runs) - 1, self.run_idx + 1)
        elif key == KEY_ESC:
            self.finished = True
        elif key == "s":
            self._advance()
        elif key == "n":
            self.mode = "typing"
            self.name_buffer = ""
        elif key == "a":
            self.mode = "pick_all"
        elif len(key) == 1 and key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.candidates):
                self._assign_current(self.candidates[idx])
        return None


def _render_nav_card(nav: "RenameNav", can_play: bool):
    """Build the Rich Panel for the keyboard-native rename loop. Pure render:
    returns a renderable, prints nothing."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    ex = nav.current
    total = len(nav.examples)
    if ex is None:  # finished — advanced past the last speaker
        return Panel(
            Text("✓ all speakers named", style="spring_green2 bold"),
            title=Text(f"[ Speaker {total} / {total} ]", style="magenta1 bold"),
            title_align="left",
            border_style="magenta1",
            padding=(0, 1),
        )
    parts: list[Text] = [
        Text(
            f"{ex.label}     {_fmt_mmss(ex.total_seconds)} speaking · "
            f"{ex.segment_count} segments",
            style="bold magenta1",
        ),
        Text(""),
    ]

    snippet = nav.current_snippet()
    if snippet:
        n = len(ex.snippets)
        dur = snippet.end - snippet.start
        parts.append(Text(f"Sample {nav.sample_idx + 1} / {n}   ({dur:.0f}s)", style="dim"))
        parts.append(Text(f'"{snippet.text}"', style="bright_cyan"))
    else:
        parts.append(Text("(no samples)", style="dim"))

    parts.append(Text(""))
    name_line = Text("Name: ", style="bold")
    name_line.append(nav.name_buffer, style="bright_white")
    name_line.append("▍", style="bright_white")
    parts.append(name_line)
    parts.append(Text(""))

    hints = Text("← → sample", style="dim")
    if can_play:
        hints.append("    ␣ play", style="dim")
    hints.append("    ↑ ↓ speaker", style="dim")
    parts.append(hints)
    parts.append(Text("type to name · ⏎ commit/keep · ⌫ edit · esc done", style="dim"))

    return Panel(
        Group(*parts),
        title=Text(f"[ Speaker {nav.speaker_idx + 1} / {total} ]", style="magenta1 bold"),
        title_align="left",
        border_style="magenta1",
        padding=(0, 1),
    )


def _render_speaker_card(
    console,
    idx: int,
    total: int,
    ex: "SpeakerExample",
    snippet: "Snippet | None",
) -> None:
    """Print the speaker-rename panel for one speaker."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    parts: list[Text] = [Text(
        f"[{idx}/{total}] {ex.label} — speaks {_fmt_mmss(ex.total_seconds)} across {ex.segment_count} segments",
        style="bold magenta1",
    )]
    if snippet:
        snip_dur = snippet.end - snippet.start
        parts.append(Text(f"  Longest sample ({snip_dur:.0f}s):", style="dim"))
        parts.append(Text(f'  "{snippet.text}"', style="bright_cyan"))
    else:
        parts.append(Text("  (no text snippets available)", style="dim"))

    console.print(Panel(Group(*parts), border_style="magenta1", padding=(0, 1)))


def interactive_rename(
    examples: list["SpeakerExample"],
    audio_path: Path | None,
    console,
    *,
    ffplay_available: bool,
) -> dict[str, str] | None:
    """Walk the user through naming each speaker.

    On a TTY, runs the keyboard-native raw loop (arrows flip samples, type to
    name, Space plays). When stdin isn't a TTY (pipe/CI), falls back to the
    line-mode prompt. Returns {SPEAKER_XX: name} for speakers actually renamed,
    {} if none were, or None if the user aborted (Ctrl-C / EOF).
    """
    if not examples:
        return {}

    can_play = (
        ffplay_available and audio_path is not None and audio_path.exists()
    )

    if not sys.stdin.isatty():
        return _interactive_rename_lines(
            examples, audio_path, console, ffplay_available=can_play
        )

    from rich.live import Live

    nav = RenameNav(examples=examples)
    player: "subprocess.Popen | None" = None

    def _stop_player() -> None:
        nonlocal player
        if player is not None and player.poll() is None:
            try:
                player.terminate()
            except OSError:
                pass
        player = None

    try:
        with raw_mode(), Live(
            _render_nav_card(nav, can_play),
            console=console,
            refresh_per_second=12,
            screen=False,
            transient=False,
        ) as live:
            while not nav.finished and not nav.aborted:
                # Render the CURRENT state at the top of the loop, so we never
                # draw the post-finish state (speaker_idx past the last speaker).
                live.update(_render_nav_card(nav, can_play))
                try:
                    key = read_key()
                except KeyboardInterrupt:
                    return None
                effect = nav.step(key)
                if effect == PLAY and can_play:
                    snippet = nav.current_snippet()
                    if snippet:
                        _stop_player()  # replace any clip already playing
                        clip = min(10.0, max(0.5, snippet.end - snippet.start))
                        player = play_snippet(audio_path, snippet.start, duration=clip)
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        _stop_player()

    return None if nav.aborted else nav.mapping


def _render_unassigned_card(nav: "UnassignedNav", can_play: bool):
    """Rich Panel for the SPEAKER_?? reassignment loop. Pure render."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    total = len(nav.runs)
    run = nav.current
    if run is None:  # finished — advanced past the last run
        return Panel(
            Text("✓ unassigned lines done", style="spring_green2 bold"),
            title=Text(f"[ Unassigned ?? — {total} / {total} ]", style="magenta1 bold"),
            title_align="left", border_style="magenta1", padding=(0, 1),
        )

    parts: list[Text] = [
        Text(f"{_fmt_mmss(run.start)}–{_fmt_mmss(run.end)}   {run.line_count} line(s)",
             style="bold magenta1"),
        Text(""),
        Text(f'"{run.text}"' if run.text else "(no text)", style="bright_cyan"),
        Text(""),
    ]

    if nav.candidates:
        choices = Text("Assign to:  ", style="bold")
        for i, name in enumerate(nav.candidates, 1):
            choices.append(f"[{i}] {name}   ", style="bright_white")
        parts.append(choices)
    else:
        parts.append(Text("(no named speakers yet — use [n] to name)", style="dim"))

    if nav.mode == "typing":
        name_line = Text("New name: ", style="bold")
        name_line.append(nav.name_buffer, style="bright_white")
        name_line.append("▍", style="bright_white")
        parts.append(name_line)
    elif nav.mode == "pick_all":
        parts.append(Text("Apply to ALL remaining — press a number", style="bold yellow1"))

    parts.append(Text(""))
    hints = Text("1-9 assign", style="dim")
    if can_play:
        hints.append("   ␣ play", style="dim")
    hints.append("   ↑↓ move   s skip   n new   a all   esc done", style="dim")
    parts.append(hints)

    return Panel(
        Group(*parts),
        title=Text(f"[ Unassigned ?? — {nav.run_idx + 1} / {total} ]", style="magenta1 bold"),
        title_align="left", border_style="magenta1", padding=(0, 1),
    )


def reassign_unassigned(
    runs: list["UnassignedRun"],
    candidates: list[str],
    audio_path: Path | None,
    console,
    *,
    ffplay_available: bool,
) -> "dict[int, str] | None":
    """Interactively fold SPEAKER_?? runs into a chosen speaker.

    Returns {run_index: name} for runs actually assigned, {} if none (or no TTY),
    or None if the user aborted (Ctrl-C / EOF). TTY-only — skipped when stdin is
    not a terminal (the unattended watcher must never block here).
    """
    if not runs or not sys.stdin.isatty():
        return {}

    from rich.live import Live

    can_play = ffplay_available and audio_path is not None and audio_path.exists()
    nav = UnassignedNav(runs=runs, candidates=candidates)
    player: "subprocess.Popen | None" = None

    def _stop_player() -> None:
        nonlocal player
        if player is not None and player.poll() is None:
            try:
                player.terminate()
            except OSError:
                pass
        player = None

    try:
        with raw_mode(), Live(
            _render_unassigned_card(nav, can_play),
            console=console, refresh_per_second=12, screen=False, transient=False,
        ) as live:
            while not nav.finished and not nav.aborted:
                live.update(_render_unassigned_card(nav, can_play))
                try:
                    key = read_key()
                except KeyboardInterrupt:
                    return None
                prev_idx = nav.run_idx
                effect = nav.step(key)
                # Cut audio the instant the displayed run changes (or we exit).
                if nav.run_idx != prev_idx or nav.finished or nav.aborted:
                    _stop_player()
                if effect == PLAY and can_play:
                    run = nav.current
                    if run is not None:
                        _stop_player()
                        clip = min(10.0, max(0.5, run.end - run.start))
                        player = play_snippet(audio_path, run.start, duration=clip)
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        _stop_player()

    return None if nav.aborted else nav.decisions


def _interactive_rename_lines(
    examples: list["SpeakerExample"],
    audio_path: Path | None,
    console,
    *,
    ffplay_available: bool,
) -> dict[str, str] | None:
    """Line-mode rename prompt (type + Enter). Used as the non-TTY fallback.
    Returns a mapping {SPEAKER_XX: name} of speakers that were actually renamed
    (skipped speakers are NOT included). Returns None if the user aborted with
    `q`. Returns an empty dict if every speaker was skipped (different from
    None — "ran to completion but no names entered").
    """
    if not examples:
        return {}

    mapping: dict[str, str] = {}
    can_play = ffplay_available and audio_path is not None and audio_path.exists()

    for idx, ex in enumerate(examples, start=1):
        snippet_cursor = 0
        while True:
            snippet = ex.snippets[snippet_cursor] if ex.snippets else None
            _render_speaker_card(console, idx, len(examples), ex, snippet)

            help_keys = ["Enter = keep"]
            if can_play and snippet:
                help_keys.append("p = play")
            if len(ex.snippets) > 1:
                help_keys.append("s = next snippet")
            help_keys.append("q = abort rename")
            prompt = f"  Name ({' · '.join(help_keys)}): "
            try:
                raw = console.input(prompt)
            except (EOFError, KeyboardInterrupt):
                console.print("[yellow]Aborted.[/yellow]")
                return None
            raw = raw.strip()

            if raw == "":
                break  # keep SPEAKER_XX
            if raw.lower() == "q":
                return None
            if raw.lower() == "p" and can_play and snippet:
                clip_dur = min(10.0, max(0.5, snippet.end - snippet.start))
                console.print(f"[dim]  ♪ playing {clip_dur:.0f}s clip...[/dim]")
                play_snippet(audio_path, snippet.start, duration=clip_dur)
                continue
            if raw.lower() == "s" and len(ex.snippets) > 1:
                snippet_cursor = (snippet_cursor + 1) % len(ex.snippets)
                continue

            mapping[ex.label] = raw
            console.print(f"  [green]✓[/green] {ex.label} → [bold]{raw}[/bold]")
            break

    return mapping


def find_rename_pending(
    transcripts_dir: Path,
    incoming_dir: Path,
) -> list[RenamePending]:
    """Walk `transcripts_dir` for JSON transcripts that still contain default
    `SPEAKER_\\d+` labels. For each, also look up the original audio file in
    `incoming_dir` (used for snippet playback)."""
    if not transcripts_dir.exists():
        return []
    pending: list[RenamePending] = []
    for json_path in sorted(transcripts_dir.rglob("*.json")):
        if json_path.name.endswith(".speakers.json"):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segments = data.get("segments") or []
        labels = unmapped_speakers(segments)
        unassigned = unassigned_segments(segments)
        if not labels and not unassigned:
            continue
        pending.append(RenamePending(
            transcript_json=json_path,
            audio_path=_find_audio_for_stem(incoming_dir, json_path.stem),
            unmapped=labels,
            unassigned_count=len(unassigned),
        ))
    return pending
