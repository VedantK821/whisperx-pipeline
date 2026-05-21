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
