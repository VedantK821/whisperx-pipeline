"""Speaker-renaming subsystem.

Owns: example building, persistence, post-hoc rewrites, cross-platform
non-blocking input, ffplay playback, and the interactive rename UI loop.
The rest of the app calls into here; this module knows nothing about the
transcription pipeline beyond the JSON segment shape.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_SPEAKER_RE = re.compile(r"^SPEAKER_\d+$")

PERSIST_VERSION = 1


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


def play_snippet(audio_path: Path, start: float, duration: float = 10.0) -> None:
    """Play `[start, start+duration]` of `audio_path` via ffplay. Blocking.
    Ctrl-C kills the subprocess and returns. Subprocess errors are swallowed
    so the UI never crashes mid-rename.
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
        subprocess.run(cmd, check=False)
    except (FileNotFoundError, KeyboardInterrupt, OSError) as e:
        log.debug("ffplay invocation failed/interrupted: %s", e)


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
