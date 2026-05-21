import json
from pathlib import Path

import pytest

from src.rename import RenamePending, find_rename_pending


def _make_transcript(dir_path: Path, stem: str, segments: list[dict]) -> None:
    transcript_dir = dir_path / stem
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{stem}.json").write_text(
        json.dumps({"segments": segments, "language": "en"}),
        encoding="utf-8",
    )


def test_finds_transcripts_with_default_labels(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    incoming = tmp_path / "incoming"
    incoming.mkdir()

    _make_transcript(transcripts, "alpha", [{"speaker": "SPEAKER_00", "text": "hi"}])
    _make_transcript(transcripts, "beta", [{"speaker": "Vedant", "text": "hi"}])
    _make_transcript(transcripts, "gamma", [
        {"speaker": "Vedant", "text": "hi"},
        {"speaker": "SPEAKER_02", "text": "yo"},
    ])

    pending = find_rename_pending(transcripts, incoming)
    stems = sorted(p.transcript_json.stem for p in pending)
    assert stems == ["alpha", "gamma"]


def test_pending_carries_unmapped_labels(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _make_transcript(transcripts, "x", [
        {"speaker": "SPEAKER_00"},
        {"speaker": "SPEAKER_01"},
        {"speaker": "SPEAKER_00"},
    ])
    pending = find_rename_pending(transcripts, incoming)
    assert len(pending) == 1
    assert sorted(pending[0].unmapped) == ["SPEAKER_00", "SPEAKER_01"]


def test_audio_path_present_when_source_exists(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "alpha.m4a").write_bytes(b"")

    _make_transcript(transcripts, "alpha", [{"speaker": "SPEAKER_00"}])
    pending = find_rename_pending(transcripts, incoming)
    assert pending[0].audio_path is not None
    assert pending[0].audio_path.name == "alpha.m4a"


def test_audio_path_none_when_source_gone(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    _make_transcript(transcripts, "alpha", [{"speaker": "SPEAKER_00"}])
    pending = find_rename_pending(transcripts, incoming)
    assert pending[0].audio_path is None


def test_ignores_malformed_json(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    incoming = tmp_path / "incoming"
    transcripts.mkdir()
    incoming.mkdir()
    bad_dir = transcripts / "broken"
    bad_dir.mkdir()
    (bad_dir / "broken.json").write_text("not json", encoding="utf-8")
    assert find_rename_pending(transcripts, incoming) == []


def test_empty_transcripts_dir_returns_empty(tmp_path: Path):
    transcripts = tmp_path / "transcripts"
    incoming = tmp_path / "incoming"
    transcripts.mkdir()
    incoming.mkdir()
    assert find_rename_pending(transcripts, incoming) == []
