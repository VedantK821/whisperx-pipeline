"""Speaker-pure, audio-matching sample selection (Bug D).

When word-level speaker tags + timings exist, each sample must be trimmed to a
contiguous run of the TARGET speaker's words, so the played window matches the
shown text and never contains another speaker.
"""
from src.rename import build_speaker_examples


def w(word, start, end, spk):
    return {"word": word, "start": start, "end": end, "speaker": spk}


def seg(spk, start, end, text, words):
    return {"start": start, "end": end, "text": text, "speaker": spk, "words": words}


def test_samples_exclude_other_speakers_words():
    # One segment attributed to SPEAKER_00 but with a SPEAKER_01 interjection
    # in the middle. No produced sample may contain the other speaker's word.
    words = [
        w("hello", 0.0, 0.7, "SPEAKER_00"),
        w("there", 0.7, 1.5, "SPEAKER_00"),
        w("yeah", 1.5, 2.0, "SPEAKER_01"),   # interjection by the OTHER speaker
        w("how", 2.0, 2.6, "SPEAKER_00"),
        w("are", 2.6, 3.0, "SPEAKER_00"),
        w("you", 3.0, 3.6, "SPEAKER_00"),
    ]
    segs = [seg("SPEAKER_00", 0.0, 3.6, "hello there yeah how are you", words)]
    examples = build_speaker_examples(segs)
    ex0 = next(e for e in examples if e.label == "SPEAKER_00")

    assert ex0.snippets, "expected at least one pure sample"
    for sn in ex0.snippets:
        assert "yeah" not in sn.text.split()


def test_sample_time_window_matches_its_text():
    # The sample's [start, end] must bound exactly the words in its text, so the
    # audio you hear corresponds to the text you read.
    words = [
        w("hello", 0.0, 0.7, "SPEAKER_00"),
        w("there", 0.7, 1.5, "SPEAKER_00"),
        w("friend", 1.5, 2.2, "SPEAKER_00"),
    ]
    segs = [seg("SPEAKER_00", 0.0, 2.2, "hello there friend", words)]
    ex0 = build_speaker_examples(segs)[0]
    sn = ex0.snippets[0]
    assert sn.text == "hello there friend"
    assert sn.start == 0.0
    assert sn.end == 2.2


def test_long_pure_run_is_trimmed_to_window():
    # A 20s monologue must be trimmed to roughly the sample window (~6s), not
    # shown in full while only the first seconds play.
    words = [w(f"word{i}", float(i), float(i) + 1.0, "SPEAKER_00") for i in range(20)]
    segs = [seg("SPEAKER_00", 0.0, 20.0, " ".join(f"word{i}" for i in range(20)), words)]
    ex0 = build_speaker_examples(segs)[0]
    sn = ex0.snippets[0]
    assert (sn.end - sn.start) <= 7.0  # trimmed near the ~6s target, not 20s


def test_falls_back_to_whole_segment_without_word_data():
    # No word timings (e.g. alignment failed) -> whole-segment samples, longest
    # first (preserves the pre-Bug-D behavior).
    segs = [
        {"start": 0.0, "end": 2.0, "text": "short", "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 9.0, "text": "a much longer line of speech", "speaker": "SPEAKER_00"},
    ]
    ex0 = build_speaker_examples(segs)[0]
    assert ex0.snippets[0].text == "a much longer line of speech"
    assert ex0.snippets[0].start == 2.0
    assert ex0.snippets[0].end == 9.0
