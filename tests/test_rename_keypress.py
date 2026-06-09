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


def test_consumes_full_arrow_sequence_so_nothing_leaks(monkeypatch):
    # Pressing an arrow to engage (countdown / review prompt) must consume the
    # trail code too. kbhit() cannot see the CRT-buffered trail byte (it stays
    # False), so the consume must be an unconditional blocking getwch() — or
    # the 'K' leaks into the next prompt and gets typed as a letter.
    import src.rename as r
    if r.sys.platform != "win32":
        pytest.skip("windows-specific console input path")
    import msvcrt

    getwch_seq = iter(["\xe0", "K"])
    hits = iter([True])  # one real console event: the arrow keypress itself
    monkeypatch.setattr(msvcrt, "getwch", lambda: next(getwch_seq))
    monkeypatch.setattr(msvcrt, "kbhit", lambda: next(hits, False))
    monkeypatch.setattr(r.time, "sleep", lambda s: None)

    assert wait_for_keypress(0.5) is not None
    with pytest.raises(StopIteration):
        next(getwch_seq)  # trail byte was consumed, not left to leak
