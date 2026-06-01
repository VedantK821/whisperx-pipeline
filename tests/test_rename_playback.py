import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.rename import play_snippet, ffplay_available


def test_play_snippet_invokes_ffplay_with_correct_args(tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"")
    with patch("src.rename.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        play_snippet(audio, start=12.5, duration=10.0)

    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == "ffplay"
    assert "-nodisp" in cmd
    assert "-autoexit" in cmd
    assert "-ss" in cmd
    assert "12.5" in cmd
    assert "-t" in cmd
    assert "10.0" in cmd
    assert str(audio) in cmd


def test_play_snippet_is_non_blocking_and_detaches_stdin(tmp_path: Path):
    # Non-blocking: returns a process handle, never waits. stdin must be
    # detached so ffplay can't steal the keystrokes we're reading in the loop.
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"")
    with patch("src.rename.subprocess.Popen") as mock_popen:
        handle = MagicMock()
        mock_popen.return_value = handle
        result = play_snippet(audio, start=0.0, duration=3.0)

    assert result is handle
    _, kwargs = mock_popen.call_args
    assert kwargs.get("stdin") == subprocess.DEVNULL


def test_play_snippet_returns_none_on_error(tmp_path: Path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"")
    with patch("src.rename.subprocess.Popen", side_effect=FileNotFoundError("no ffplay")):
        assert play_snippet(audio, start=0.0, duration=10.0) is None


def test_ffplay_available_returns_bool():
    assert isinstance(ffplay_available(), bool)


def test_ffplay_available_false_when_not_on_path():
    with patch("src.rename.shutil.which", return_value=None):
        assert ffplay_available() is False


def test_ffplay_available_true_when_on_path():
    with patch("src.rename.shutil.which", return_value="/usr/bin/ffplay"):
        assert ffplay_available() is True
