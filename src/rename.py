"""Speaker-renaming subsystem.

Owns: example building, persistence, post-hoc rewrites, cross-platform
non-blocking input, ffplay playback, and the interactive rename UI loop.
The rest of the app calls into here; this module knows nothing about the
transcription pipeline beyond the JSON segment shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field


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
