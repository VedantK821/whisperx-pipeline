"""Speaker-renaming subsystem.

Owns: example building, persistence, post-hoc rewrites, cross-platform
non-blocking input, ffplay playback, and the interactive rename UI loop.
The rest of the app calls into here; this module knows nothing about the
transcription pipeline beyond the JSON segment shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_DEFAULT_SPEAKER_RE = re.compile(r"^SPEAKER_\d+$")


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
