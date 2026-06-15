import json
from pathlib import Path

from src.export_speakers import (
    ffmpeg_clip_cmd,
    export_clips,
    export_speaker_artifacts,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["segments"]


def test_ffmpeg_clip_cmd_shape():
    cmd = ffmpeg_clip_cmd(Path("a.m4a"), 12.5, 6.0, Path("out/SPEAKER_00.ogg"))
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and "12.5" in cmd
    assert "-t" in cmd and "6.0" in cmd
    assert "-c:a" in cmd and "libopus" in cmd
    assert cmd[-1].endswith("SPEAKER_00.ogg")


def test_export_clips_cuts_best_snippet_per_speaker(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"OggS")  # pretend ffmpeg produced a clip
    monkeypatch.setattr("src.export_speakers.subprocess.run", fake_run)

    segments = _load("transcript_two_speakers.json")
    result = export_clips(segments, Path("meeting.m4a"), tmp_path / "clips")

    assert set(result) == {"SPEAKER_00", "SPEAKER_01"}
    assert result["SPEAKER_00"] == "SPEAKER_00.ogg"
    assert (tmp_path / "clips" / "SPEAKER_00.ogg").exists()
    assert len(calls) == 2
    # SPEAKER_00's best snippet starts at 30.0 ("speaker zero again briefly")
    spk0_cmd = next(c for c in calls if c[-1].endswith("SPEAKER_00.ogg"))
    assert "30.0" in spk0_cmd


def test_export_speaker_artifacts_writes_embeddings_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.export_speakers.subprocess.run",
        lambda cmd, check: Path(cmd[-1]).write_bytes(b"OggS"),
    )
    segments = _load("transcript_two_speakers.json")
    embeddings = {"SPEAKER_00": [0.1, 0.2, 0.3], "SPEAKER_01": [0.4, 0.5, 0.6]}

    export_speaker_artifacts(segments, embeddings, Path("meeting.m4a"), tmp_path)

    out = json.loads((tmp_path / "meeting.embeddings.json").read_text(encoding="utf-8"))
    assert out["embeddings"] == embeddings
    assert set(out["clips"]) == {"SPEAKER_00", "SPEAKER_01"}


def test_export_speaker_artifacts_tolerates_none_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.export_speakers.subprocess.run",
        lambda cmd, check: Path(cmd[-1]).write_bytes(b"OggS"),
    )
    segments = _load("transcript_two_speakers.json")

    export_speaker_artifacts(segments, None, Path("meeting.m4a"), tmp_path)

    out = json.loads((tmp_path / "meeting.embeddings.json").read_text(encoding="utf-8"))
    assert out["embeddings"] == {}
    assert set(out["clips"]) == {"SPEAKER_00", "SPEAKER_01"}
