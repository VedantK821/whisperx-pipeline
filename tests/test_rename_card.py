from rich.console import Console

from src.rename import RenameNav, Snippet, SpeakerExample, _render_nav_card, KEY_ENTER


def ex(label, snippets=("hello world",), total=252.0, segs=87):
    return SpeakerExample(
        label=label,
        total_seconds=total,
        segment_count=segs,
        snippets=[Snippet(text=t, start=0.0, end=8.0) for t in snippets],
    )


def render_to_text(nav, can_play):
    console = Console(width=80, record=True)
    console.print(_render_nav_card(nav, can_play))
    return console.export_text()


def test_card_shows_label_and_buffer():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    nav.name_buffer = "Sar"
    out = render_to_text(nav, can_play=True)
    assert "SPEAKER_00" in out
    assert "Sar" in out
    assert "Speaker 1 / 1" in out


def test_card_hides_play_hint_when_cannot_play():
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    out = render_to_text(nav, can_play=False)
    assert "play" not in out.lower()


def test_card_handles_speaker_with_no_samples():
    nav = RenameNav(examples=[ex("SPEAKER_00", snippets=())])
    out = render_to_text(nav, can_play=True)
    assert "no samples" in out.lower()


def test_render_after_all_named_does_not_crash():
    # Regression: committing the LAST speaker advances nav.current to None
    # (speaker_idx == len(examples)). Rendering that finished state must not
    # raise — previously AttributeError on None.label, which lost the names.
    nav = RenameNav(examples=[ex("SPEAKER_00")])
    nav.step("A")
    nav.step(KEY_ENTER)  # commit last speaker -> finished, index out of range
    assert nav.finished
    _render_nav_card(nav, can_play=False)  # must not raise
