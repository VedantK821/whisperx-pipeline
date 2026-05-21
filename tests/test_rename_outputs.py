import json
from pathlib import Path

import pytest

from src.rename import rewrite_outputs_with_mapping


def _write_minimal_transcript(transcript_dir: Path, stem: str) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    json_path = transcript_dir / f"{stem}.json"
    data = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "text": "hi", "speaker": "SPEAKER_01"},
        ],
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")
    (transcript_dir / f"{stem}.txt").write_text("stale", encoding="utf-8")
    (transcript_dir / f"{stem}.srt").write_text("stale", encoding="utf-8")
    return json_path


def test_rewrite_updates_json_labels(tmp_path: Path):
    transcript_dir = tmp_path / "meeting"
    json_path = _write_minimal_transcript(transcript_dir, "meeting")

    rewrite_outputs_with_mapping(json_path, {"SPEAKER_00": "Alice"})

    new = json.loads(json_path.read_text(encoding="utf-8"))
    speakers = [s["speaker"] for s in new["segments"]]
    assert speakers == ["Alice", "SPEAKER_01"]


def test_rewrite_regenerates_txt(tmp_path: Path):
    transcript_dir = tmp_path / "meeting"
    json_path = _write_minimal_transcript(transcript_dir, "meeting")
    rewrite_outputs_with_mapping(json_path, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})

    txt = (transcript_dir / "meeting.txt").read_text(encoding="utf-8")
    assert "stale" not in txt
    assert "Alice: hello" in txt
    assert "Bob: hi" in txt


def test_rewrite_regenerates_srt(tmp_path: Path):
    transcript_dir = tmp_path / "meeting"
    json_path = _write_minimal_transcript(transcript_dir, "meeting")
    rewrite_outputs_with_mapping(json_path, {"SPEAKER_00": "Alice"})

    srt = (transcript_dir / "meeting.srt").read_text(encoding="utf-8")
    assert "stale" not in srt
    assert "Alice: hello" in srt


def test_rewrite_no_mapping_is_noop_on_labels(tmp_path: Path):
    transcript_dir = tmp_path / "meeting"
    json_path = _write_minimal_transcript(transcript_dir, "meeting")
    rewrite_outputs_with_mapping(json_path, {})

    new = json.loads(json_path.read_text(encoding="utf-8"))
    speakers = [s["speaker"] for s in new["segments"]]
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert "stale" not in (transcript_dir / "meeting.txt").read_text(encoding="utf-8")
