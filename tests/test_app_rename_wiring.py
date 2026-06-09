"""_run_rename_for_transcript must, in one pass: apply the speaker names, fold
the SPEAKER_?? lines into the chosen speaker, and re-emit outputs once so both
land. Interactions are canned; the write path is real.
"""
import json
import types

import src.app as app
import src.rename as rename


def _make_transcript(tmp_path, stem, segments):
    d = tmp_path / stem
    d.mkdir(parents=True)
    j = d / f"{stem}.json"
    j.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return j


def test_run_rename_applies_names_and_unassigned(tmp_path, monkeypatch):
    segs = [
        {"speaker": "SPEAKER_00", "text": "hello", "start": 0.0, "end": 1.0},
        {"text": "thanks", "start": 1.0, "end": 2.0},          # SPEAKER_?? (no speaker)
        {"speaker": "SPEAKER_00", "text": "bye", "start": 2.0, "end": 3.0},
    ]
    j = _make_transcript(tmp_path, "meet", segs)

    monkeypatch.setattr(rename, "interactive_rename", lambda *a, **k: {"SPEAKER_00": "Alice"})
    # The lone ?? run (index 0) -> Alice.
    monkeypatch.setattr(rename, "reassign_unassigned", lambda runs, cands, *a, **k: {0: "Alice"})

    cfg = types.SimpleNamespace(transcripts_dir=tmp_path)
    app._run_rename_for_transcript(j, None, cfg, ffplay_avail=False)

    txt = (tmp_path / "meet" / "meet.txt").read_text(encoding="utf-8")
    assert "Alice: hello" in txt
    assert "Alice: thanks" in txt        # the ?? line was folded into Alice
    assert "SPEAKER_??" not in txt
    assert "SPEAKER_00" not in txt

    data = json.loads(j.read_text(encoding="utf-8"))
    assert all(s["speaker"] == "Alice" for s in data["segments"])


def test_run_rename_handles_unassigned_only_when_speakers_already_named(tmp_path, monkeypatch):
    # All real speakers already named; only ?? lines remain.
    segs = [
        {"speaker": "Alice", "text": "hello", "start": 0.0, "end": 1.0},
        {"text": "thanks", "start": 1.0, "end": 2.0},          # ??
    ]
    j = _make_transcript(tmp_path, "named", segs)

    called = {"rename": False}

    def _no_rename(*a, **k):
        called["rename"] = True
        return {}

    monkeypatch.setattr(rename, "interactive_rename", _no_rename)
    monkeypatch.setattr(rename, "reassign_unassigned", lambda runs, cands, *a, **k: {0: "Alice"})

    cfg = types.SimpleNamespace(transcripts_dir=tmp_path)
    app._run_rename_for_transcript(j, None, cfg, ffplay_avail=False)

    assert called["rename"] is False  # no SPEAKER_xx left, so no naming step
    txt = (tmp_path / "named" / "named.txt").read_text(encoding="utf-8")
    assert "Alice: thanks" in txt
    assert "SPEAKER_??" not in txt
