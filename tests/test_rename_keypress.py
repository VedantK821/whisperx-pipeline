import time

import pytest

from src.rename import wait_for_keypress


def test_returns_none_on_timeout():
    start = time.monotonic()
    result = wait_for_keypress(0.2)
    elapsed = time.monotonic() - start
    assert result is None
    assert 0.15 <= elapsed < 0.6


def test_zero_timeout_returns_immediately():
    start = time.monotonic()
    result = wait_for_keypress(0.0)
    elapsed = time.monotonic() - start
    assert result is None
    assert elapsed < 0.1
