import pytest

from src.rename import (
    decode_key,
    KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN,
    KEY_ENTER, KEY_BACKSPACE, KEY_ESC, KEY_EOF, KEY_UNKNOWN,
)


def feed(chars):
    """Return a getch() callable that yields chars then "" (EOF) forever."""
    it = iter(chars)

    def getch():
        try:
            return next(it)
        except StopIteration:
            return ""

    return getch


def test_windows_arrows():
    assert decode_key(feed(["\xe0", "K"])) == KEY_LEFT
    assert decode_key(feed(["\xe0", "M"])) == KEY_RIGHT
    assert decode_key(feed(["\x00", "H"])) == KEY_UP
    assert decode_key(feed(["\x00", "P"])) == KEY_DOWN


def test_posix_arrows():
    assert decode_key(feed(["\x1b", "[", "D"])) == KEY_LEFT
    assert decode_key(feed(["\x1b", "[", "C"])) == KEY_RIGHT
    assert decode_key(feed(["\x1b", "[", "A"])) == KEY_UP
    assert decode_key(feed(["\x1b", "[", "B"])) == KEY_DOWN


def test_enter_backspace_esc():
    assert decode_key(feed(["\r"])) == KEY_ENTER
    assert decode_key(feed(["\n"])) == KEY_ENTER
    assert decode_key(feed(["\x7f"])) == KEY_BACKSPACE
    assert decode_key(feed(["\x08"])) == KEY_BACKSPACE
    assert decode_key(feed(["\x1b"])) == KEY_ESC  # bare ESC: no following byte


def test_printable_passthrough():
    assert decode_key(feed(["a"])) == "a"
    assert decode_key(feed(["Q"])) == "Q"
    assert decode_key(feed([" "])) == " "


def test_eof_on_closed_stream():
    assert decode_key(feed([""])) == KEY_EOF


def test_unknown_function_key_is_ignored_not_eof():
    # F1 on Windows arrives as \x00 ; — must NOT abort the loop.
    assert decode_key(feed(["\x00", ";"])) == KEY_UNKNOWN


def test_ctrl_c_raises_keyboardinterrupt():
    with pytest.raises(KeyboardInterrupt):
        decode_key(feed(["\x03"]))


from src.rename import read_key, raw_mode


def test_read_key_delegates_to_decoder():
    # Inject a getch so we don't touch a real terminal.
    seq = iter(["\xe0", "M"])
    assert read_key(_getch=lambda: next(seq, "")) == KEY_RIGHT


def test_raw_mode_is_a_noop_context_when_not_a_tty(monkeypatch):
    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with raw_mode():  # must not raise even with no real terminal
        pass


def test_read_key_waits_for_lagging_arrow_followup_byte(monkeypatch):
    # Windows race: the \xe0 arrow prefix arrives, kbhit() briefly lags, then
    # the code byte 'M' (right arrow) shows up. read_key must wait for it and
    # decode KEY_RIGHT — not drop it and let 'M' get typed as a letter later.
    import src.rename as r
    if r.sys.platform != "win32":
        import pytest
        pytest.skip("windows-specific console input path")
    import msvcrt

    getwch_seq = iter(["\xe0", "M"])
    kbhit_seq = iter([False, True])  # lag once, then the byte is ready
    monkeypatch.setattr(msvcrt, "getwch", lambda: next(getwch_seq))
    monkeypatch.setattr(msvcrt, "kbhit", lambda: next(kbhit_seq))
    monkeypatch.setattr(r.time, "sleep", lambda s: None)

    assert r.read_key() == KEY_RIGHT
