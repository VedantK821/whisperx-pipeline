import json
from pathlib import Path

import pytest

from src.rename import load_persisted_mapping, save_persisted_mapping


def test_save_then_load_roundtrip(tmp_path: Path):
    save_persisted_mapping(
        tmp_path, "meeting", "meeting.m4a",
        {"SPEAKER_00": "Carol", "SPEAKER_01": "Dave"},
    )
    out = tmp_path / "meeting.speakers.json"
    assert out.exists()

    loaded = load_persisted_mapping(tmp_path, "meeting")
    assert loaded == {"SPEAKER_00": "Carol", "SPEAKER_01": "Dave"}


def test_save_writes_versioned_schema(tmp_path: Path):
    save_persisted_mapping(tmp_path, "x", "x.wav", {"SPEAKER_00": "A"})
    data = json.loads((tmp_path / "x.speakers.json").read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["audio_filename"] == "x.wav"
    assert data["mapping"] == {"SPEAKER_00": "A"}


def test_load_missing_returns_none(tmp_path: Path):
    assert load_persisted_mapping(tmp_path, "absent") is None


def test_load_malformed_returns_none(tmp_path: Path):
    (tmp_path / "bad.speakers.json").write_text("not json", encoding="utf-8")
    assert load_persisted_mapping(tmp_path, "bad") is None


def test_load_wrong_schema_returns_none(tmp_path: Path):
    (tmp_path / "x.speakers.json").write_text(
        json.dumps({"version": 999, "mapping": {}}), encoding="utf-8",
    )
    assert load_persisted_mapping(tmp_path, "x") is None


def test_save_overwrites_existing(tmp_path: Path):
    save_persisted_mapping(tmp_path, "x", "x.wav", {"SPEAKER_00": "First"})
    save_persisted_mapping(tmp_path, "x", "x.wav", {"SPEAKER_00": "Second"})
    assert load_persisted_mapping(tmp_path, "x") == {"SPEAKER_00": "Second"}
