from src.rename import (
    RenameNav, Snippet, SpeakerExample, PLAY,
    KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, KEY_ENTER, KEY_BACKSPACE, KEY_ESC, KEY_EOF,
)


def ex(label, snippets=("hello",), total=10.0, segs=3):
    return SpeakerExample(
        label=label,
        total_seconds=total,
        segment_count=segs,
        snippets=[Snippet(text=t, start=0.0, end=2.0) for t in snippets],
    )


def type_str(nav, s):
    for ch in s:
        nav.step(ch)


def test_type_then_enter_names_and_advances():
    nav = RenameNav(examples=[ex("SPEAKER_00"), ex("SPEAKER_01")])
    type_str(nav, "Sarah")
    assert nav.name_buffer == "Sarah"
    nav.step(KEY_ENTER)
    assert nav.mapping == {"SPEAKER_00": "Sarah"}
    assert nav.speaker_idx == 1
    assert not nav.finished


def test_enter_on_empty_keeps_label_and_advances():
    nav = RenameNav(examples=[ex("SPEAKER_00"), ex("SPEAKER_01")])
    nav.step(KEY_ENTER)
    assert nav.mapping == {}
    assert nav.speaker_idx == 1


def test_enter_past_last_finishes():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    nav.step(KEY_ENTER)
    assert nav.finished is True


def test_right_arrow_wraps_samples():
    nav = RenameNav(examples=[ex("SPEAKER_00", snippets=("a", "b", "c"))])
    assert nav.sample_idx == 0
    nav.step(KEY_RIGHT); assert nav.sample_idx == 1
    nav.step(KEY_RIGHT); assert nav.sample_idx == 2
    nav.step(KEY_RIGHT); assert nav.sample_idx == 0


def test_left_arrow_wraps_backwards():
    nav = RenameNav(examples=[ex("SPEAKER_00", snippets=("a", "b", "c"))])
    nav.step(KEY_LEFT); assert nav.sample_idx == 2


def test_arrows_flip_samples_even_while_typing():
    nav = RenameNav(examples=[ex("SPEAKER_00", snippets=("a", "b"))])
    type_str(nav, "Jo")
    nav.step(KEY_RIGHT)
    assert nav.sample_idx == 1
    assert nav.name_buffer == "Jo"  # name untouched


def test_space_plays_in_command_mode():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    assert nav.step(" ") == PLAY
    assert nav.name_buffer == ""  # did NOT start a name


def test_space_is_literal_while_typing():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    type_str(nav, "Jo")
    assert nav.step(" ") is None
    nav.step("e")
    assert nav.name_buffer == "Jo e"


def test_down_moves_speaker_clamped():
    nav = RenameNav(examples=[ex("a"), ex("b")])
    nav.step(KEY_DOWN); assert nav.speaker_idx == 1
    nav.step(KEY_DOWN); assert nav.speaker_idx == 1  # clamp at last


def test_up_clamped_at_zero():
    nav = RenameNav(examples=[ex("a"), ex("b")])
    nav.step(KEY_UP); assert nav.speaker_idx == 0


def test_updown_ignored_while_typing():
    nav = RenameNav(examples=[ex("a"), ex("b")])
    nav.step("X")
    nav.step(KEY_DOWN)
    assert nav.speaker_idx == 0
    assert nav.name_buffer == "X"


def test_esc_in_command_mode_finishes_and_keeps_partial():
    nav = RenameNav(examples=[ex("SPEAKER_00"), ex("SPEAKER_01")])
    type_str(nav, "Amy")
    nav.step(KEY_ENTER)   # commit Amy -> speaker 1
    nav.step(KEY_ESC)     # finish early
    assert nav.finished is True
    assert nav.mapping == {"SPEAKER_00": "Amy"}


def test_esc_while_typing_cancels_name_only():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    type_str(nav, "Bob")
    nav.step(KEY_ESC)
    assert nav.name_buffer == ""
    assert nav.finished is False
    assert nav.mapping == {}


def test_backspace_edits_and_returns_to_command_mode():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    type_str(nav, "Ab")
    nav.step(KEY_BACKSPACE); assert nav.name_buffer == "A"
    nav.step(KEY_BACKSPACE); assert nav.name_buffer == ""
    assert nav.step(" ") == PLAY  # back in command mode


def test_eof_aborts():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    nav.step(KEY_EOF)
    assert nav.aborted is True


def test_whitespace_only_name_is_not_committed():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    nav.step(" ")          # PLAY, not a name
    type_str(nav, "  ")    # only spaces while in command mode? first space plays.
    nav.step(KEY_ENTER)
    assert nav.mapping == {}


def test_current_snippet_none_when_no_snippets():
    nav = RenameNav(examples=[ex("SPEAKER_00", snippets=())])
    assert nav.current_snippet() is None
    nav.step(KEY_RIGHT)  # must not raise
    assert nav.sample_idx == 0
