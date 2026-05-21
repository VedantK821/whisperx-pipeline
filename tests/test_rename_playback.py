from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.rename import play_snippet, ffplay_available


def test_play_snippet_invokes_ffplay_with_correct_args(tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"")
    with patch("src.rename.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        play_snippet(audio, start=12.5, duration=10.0)

    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "ffplay"
    assert "-nodisp" in cmd
    assert "-autoexit" in cmd
    assert "-ss" in cmd
    assert "12.5" in cmd
    assert "-t" in cmd
    assert "10.0" in cmd
    assert str(audio) in cmd


def test_play_snippet_swallows_subprocess_errors(tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"")
    with patch("src.rename.subprocess.run", side_effect=FileNotFoundError("no ffplay")):
        play_snippet(audio, start=0.0, duration=10.0)


def test_ffplay_available_returns_bool():
    assert isinstance(ffplay_available(), bool)


def test_ffplay_available_false_when_not_on_path():
    with patch("src.rename.shutil.which", return_value=None):
        assert ffplay_available() is False


def test_ffplay_available_true_when_on_path():
    with patch("src.rename.shutil.which", return_value="/usr/bin/ffplay"):
        assert ffplay_available() is True
