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


def test_falls_back_to_segment_snippets_without_word_data():
    # No word timings (e.g. alignment failed) -> per-segment samples are still
    # produced, anchored at the segment start.
    segs = [
        {"start": 0.0, "end": 2.0, "text": "short", "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 9.0, "text": "a much longer line of speech", "speaker": "SPEAKER_00"},
    ]
    ex0 = build_speaker_examples(segs)[0]
    assert len(ex0.snippets) == 2
    assert {sn.start for sn in ex0.snippets} == {0.0, 2.0}


def test_fallback_trims_long_segments_to_the_sample_window():
    # A 30s segment with no word data must not be shown as a 30s wall of text
    # while only the first seconds play: the text is cut near the sample
    # window (marked with an ellipsis) and the window end matches the cut.
    text = " ".join(f"word{i:02d}" for i in range(60))  # 60 words, ~30s of speech
    segs = [{"start": 10.0, "end": 40.0, "text": text, "speaker": "SPEAKER_00"}]
    ex0 = build_speaker_examples(segs)[0]
    sn = ex0.snippets[0]
    assert sn.start == 10.0
    assert (sn.end - sn.start) <= 6.5          # ~_SAMPLE_MAX_S, not 30s
    assert sn.text.endswith("…")
    assert len(sn.text) <= len(text) * 0.4     # roughly 6/30 of the text shown


def test_fallback_prefers_short_clean_segments_over_long_blobs():
    # Sweet-spot segments (~3-6s) make better voice samples than trimmed
    # 30-second blobs — they're far more likely to be a single speaker.
    blob = " ".join(f"blob{i}" for i in range(60))
    segs = [
        {"start": 0.0, "end": 30.0, "text": blob, "speaker": "SPEAKER_00"},
        {"start": 50.0, "end": 54.5, "text": "a clean short line from one person",
         "speaker": "SPEAKER_00"},
    ]
    ex0 = build_speaker_examples(segs)[0]
    assert ex0.snippets[0].text == "a clean short line from one person"


def test_fallback_caps_sample_count_like_the_pure_path():
    segs = [
        {"start": float(i * 10), "end": float(i * 10 + 4), "text": f"line {i}",
         "speaker": "SPEAKER_00"}
        for i in range(20)
    ]
    ex0 = build_speaker_examples(segs)[0]
    assert len(ex0.snippets) <= 6  # _SAMPLES_PER_SPEAKER


def test_run_with_a_time_gap_is_split_so_audio_stays_tight():
    # Same speaker, two bursts separated by an 8s hole (someone else talking,
    # untranscribed). One window spanning the hole would play the other
    # speaker's audio — the run must split at the gap.
    words = [w(f"a{i}", i * 0.5, i * 0.5 + 0.5, "SPEAKER_00") for i in range(4)]
    words += [w(f"b{i}", 10.0 + i * 0.5, 10.5 + i * 0.5, "SPEAKER_00") for i in range(4)]
    segs = [seg("SPEAKER_00", 0.0, 12.0, " ".join(x["word"] for x in words), words)]
    ex0 = build_speaker_examples(segs)[0]
    assert ex0.snippets
    for sn in ex0.snippets:
        assert (sn.end - sn.start) <= 3.0  # never a window spanning the hole


def test_degenerate_alignment_durations_are_dropped():
    # Real artifact: 'minister?' aligned across 9.5s. Playing 9.5s of audio
    # for a one-word quote is guaranteed mismatch — a sane chars/sec floor
    # must drop it in favor of normal samples.
    segs = [
        seg("SPEAKER_00", 0.0, 9.5, "minister?", [w("minister?", 0.0, 9.5, "SPEAKER_00")]),
        seg("SPEAKER_00", 20.0, 23.0, "a perfectly normal sample here",
            [w("a", 20.0, 20.3, "SPEAKER_00"), w("perfectly", 20.3, 21.0, "SPEAKER_00"),
             w("normal", 21.0, 21.6, "SPEAKER_00"), w("sample", 21.6, 22.3, "SPEAKER_00"),
             w("here", 22.3, 23.0, "SPEAKER_00")]),
    ]
    ex0 = build_speaker_examples(segs)[0]
    assert ex0.snippets
    for sn in ex0.snippets:
        assert len(sn.text) / (sn.end - sn.start) >= 3.0
