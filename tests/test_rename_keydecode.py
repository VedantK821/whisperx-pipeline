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
