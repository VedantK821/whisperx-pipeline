from pathlib import Path
from unittest.mock import MagicMock, patch

from src.app import _parse_pending_choice, _pick_and_rename_pending, _render_pending_list
from src.rename import RenamePending


def mkpending(n):
    return [
        RenamePending(
            transcript_json=Path(f"transcripts/f{i}/f{i}.json"),
            audio_path=(Path(f"incoming/f{i}.m4a") if i % 2 == 0 else None),
            unmapped=[f"SPEAKER_0{i}"],
        )
        for i in range(n)
    ]


# ── pure parser ──────────────────────────────────────────────────────────────

def test_parse_all():
    assert _parse_pending_choice("a", 3) == "all"
    assert _parse_pending_choice("ALL", 3) == "all"


def test_parse_quit():
    assert _parse_pending_choice("q", 3) == "quit"
    assert _parse_pending_choice("", 3) == "quit"


def test_parse_index_is_zero_based():
    assert _parse_pending_choice("1", 3) == 0
    assert _parse_pending_choice("3", 3) == 2


def test_parse_out_of_range_is_none():
    assert _parse_pending_choice("0", 3) is None
    assert _parse_pending_choice("4", 3) is None


def test_parse_garbage_is_none():
    assert _parse_pending_choice("xyz", 3) is None


# ── renderer ─────────────────────────────────────────────────────────────────

def test_render_lists_each_pending_with_index_and_count():
    from rich.console import Console

    console = Console(width=80, record=True)
    console.print(_render_pending_list(mkpending(2)))
    out = console.export_text()
    assert "f0" in out and "f1" in out
    assert "1" in out and "2" in out
    assert "unnamed" in out


# ── interactive picker ───────────────────────────────────────────────────────

def test_quit_renames_nothing():
    pend = mkpending(3)
    with patch("src.app.Prompt.ask", return_value="q"), \
         patch("src.app._run_rename_for_transcript") as run:
        _pick_and_rename_pending(pend, MagicMock(), MagicMock(), ffplay_avail=False)
    run.assert_not_called()


def test_pick_specific_renames_only_that_one():
    pend = mkpending(3)
    with patch("src.app.Prompt.ask", side_effect=["2", "q"]), \
         patch("src.app._run_rename_for_transcript") as run:
        _pick_and_rename_pending(pend, MagicMock(), MagicMock(), ffplay_avail=False)
    run.assert_called_once()
    assert run.call_args.args[0] == pend[1].transcript_json


def test_pick_all_renames_everything():
    pend = mkpending(3)
    with patch("src.app.Prompt.ask", return_value="a"), \
         patch("src.app._run_rename_for_transcript") as run:
        _pick_and_rename_pending(pend, MagicMock(), MagicMock(), ffplay_avail=False)
    assert run.call_count == 3
