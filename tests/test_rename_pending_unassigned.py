"""find_rename_pending must also surface older transcripts that have SPEAKER_??
lines — even when every real speaker is already named — so they can be cleaned
up from the startup 'r' flow.
"""
import json

import src.rename as r


def write_transcript(tmp_path, stem, segments):
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")


def seg(speaker=None, start=0.0, end=1.0):
    s = {"text": "x", "start": start, "end": end}
    if speaker is not None:
        s["speaker"] = speaker
    return s


def test_fully_named_transcript_with_unassigned_lines_is_pending(tmp_path):
    write_transcript(tmp_path, "meeting", [seg("Alice"), seg(None), seg(None)])
    pending = r.find_rename_pending(tmp_path, tmp_path / "incoming")
    assert len(pending) == 1
    assert pending[0].unmapped == []
    assert pending[0].unassigned_count == 2


def test_clean_transcript_is_not_pending(tmp_path):
    write_transcript(tmp_path, "clean", [seg("Alice"), seg("Bob")])
    assert r.find_rename_pending(tmp_path, tmp_path / "incoming") == []


def test_unmapped_speakers_still_detected_with_count(tmp_path):
    write_transcript(tmp_path, "raw", [seg("SPEAKER_00"), seg("SPEAKER_01"), seg(None)])
    pending = r.find_rename_pending(tmp_path, tmp_path / "incoming")
    assert len(pending) == 1
    assert set(pending[0].unmapped) == {"SPEAKER_00", "SPEAKER_01"}
    assert pending[0].unassigned_count == 1
