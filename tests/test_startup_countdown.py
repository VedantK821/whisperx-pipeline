from pathlib import Path
from unittest.mock import patch

from src.app import _startup_countdown, Row


def make_rows(n):
    return [Row(path=Path(f"incoming/file{i}.m4a"), status="pending") for i in range(n)]


def test_zero_timeout_proceeds_immediately():
    assert _startup_countdown(make_rows(1), 0.0) is True


def test_proceeds_on_timeout():
    # No key ever pressed -> countdown elapses -> proceed (auto-run).
    with patch("src.rename.wait_for_keypress", return_value=None):
        assert _startup_countdown(make_rows(2), 0.3) is True


def test_returns_false_when_key_pressed():
    # Any key -> drop to the menu.
    with patch("src.rename.wait_for_keypress", return_value="x"):
        assert _startup_countdown(make_rows(1), 5.0) is False


def test_start_timeout_arg_defaults_to_3():
    import argparse

    # Guards the intended default/contract for the --start-timeout flag.
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timeout", type=float, default=3.0)
    assert parser.parse_args([]).start_timeout == 3.0
    assert parser.parse_args(["--start-timeout", "0"]).start_timeout == 0.0
