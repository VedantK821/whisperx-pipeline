import json
from pathlib import Path

import pytest

from src.rename import Snippet, SpeakerExample, build_speaker_examples

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["segments"]


def test_build_examples_returns_speakers_sorted_by_total_speaking_time():
    segments = _load("transcript_two_speakers.json")
    examples = build_speaker_examples(segments)

    assert [e.label for e in examples] == ["SPEAKER_00", "SPEAKER_01"]
    # SPEAKER_00: (5-0) + (15-5) + (35-30) = 20.0
    # SPEAKER_01: (22-15) + (30-22) = 15.0
    assert examples[0].total_seconds == pytest.approx(20.0)
    assert examples[1].total_seconds == pytest.approx(15.0)


def test_build_examples_counts_segments():
    segments = _load("transcript_two_speakers.json")
    examples = build_speaker_examples(segments)

    by_label = {e.label: e for e in examples}
    assert by_label["SPEAKER_00"].segment_count == 3
    assert by_label["SPEAKER_01"].segment_count == 2


def test_build_examples_snippets_sorted_by_text_length_desc():
    segments = _load("transcript_two_speakers.json")
    examples = build_speaker_examples(segments)

    speaker0 = next(e for e in examples if e.label == "SPEAKER_00")
    assert speaker0.snippets[0].text.startswith("this is a longer segment")
    lengths = [len(s.text) for s in speaker0.snippets]
    assert lengths == sorted(lengths, reverse=True)


def test_build_examples_skips_segments_without_speaker():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "anon", "speaker": None},
        {"start": 1.0, "end": 2.0, "text": "has speaker", "speaker": "SPEAKER_00"},
    ]
    examples = build_speaker_examples(segments)
    assert len(examples) == 1
    assert examples[0].label == "SPEAKER_00"


def test_build_examples_empty_returns_empty():
    assert build_speaker_examples([]) == []
