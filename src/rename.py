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

        first = True

        def getch() -> str:
            nonlocal first
            if first:
                first = False
                return msvcrt.getwch()        # blocking — wait for the first key
            # Follow-up byte of a multi-key sequence (e.g. the code after an
            # \xe0/\x00 arrow prefix). It arrives almost instantly, but kbhit()
            # can briefly lag — so poll for a short window instead of giving up
            # immediately and letting the stray byte get typed as a letter.
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    return msvcrt.getwch()
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


def build_speaker_examples(segments: list[dict]) -> list[SpeakerExample]:
    """Group segments by speaker. Return examples sorted by total speaking time
    (descending). Each example's `snippets` are sorted by text length descending.
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
        snippets = [
            Snippet(
                text=(s.get("text") or "").strip(),
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
            )
            for s in segs
            if (s.get("text") or "").strip()
        ]
        snippets.sort(key=lambda sn: len(sn.text), reverse=True)
        examples.append(SpeakerExample(
            label=label,
            total_seconds=total,
            segment_count=len(segs),
            snippets=snippets,
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
                # Function/arrow keys produce a two-char sequence; consume the
                # second byte so the next prompt isn't polluted.
                if ch in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    return ch
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
                return sys.stdin.read(1)
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
        if not labels:
            continue
        pending.append(RenamePending(
            transcript_json=json_path,
            audio_path=_find_audio_for_stem(incoming_dir, json_path.stem),
            unmapped=labels,
        ))
    return pending
