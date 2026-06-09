"""Pure-data helpers behind the SPEAKER_?? (unassigned-line) reassignment pass.

SPEAKER_?? is the output fallback for a segment whisperx left with no `speaker`
field (diarization attributed it to no cluster). These helpers find those gaps,
group adjacent ones into runs, list the speakers you could assign them to, and
write a chosen speaker onto the gap segments + their words.
"""
import src.rename as r


def seg(speaker=None, text="", start=0.0, end=0.0, words=None):
    d = {"text": text, "start": start, "end": end}
    if speaker is not None:
        d["speaker"] = speaker
    if words is not None:
        d["words"] = words
    return d


def test_unassigned_segments_returns_indices_with_no_speaker():
    segs = [seg("SPEAKER_00", "a"), seg(None, "b"), seg("", "c"), seg("Alice", "d")]
    # index 1 (None) and 2 ("") have no usable speaker; 0 and 3 do.
    assert r.unassigned_segments(segs) == [1, 2]


def test_group_contiguous_splits_runs_of_consecutive_indices():
    assert r.group_contiguous([0, 1, 2, 5, 6, 9]) == [[0, 1, 2], [5, 6], [9]]
    assert r.group_contiguous([]) == []
    assert r.group_contiguous([3]) == [[3]]


def test_build_unassigned_runs_groups_and_collects_text_and_span():
    segs = [
        seg("Alice", "hi", 0.0, 1.0),
        seg(None, "thank", 2.0, 3.0),
        seg(None, "you", 3.0, 4.0),
        seg("Alice", "bye", 5.0, 6.0),
        seg(None, "end", 7.0, 8.0),
    ]
    runs = r.build_unassigned_runs(segs)
    assert [run.indices for run in runs] == [[1, 2], [4]]
    assert runs[0].start == 2.0 and runs[0].end == 4.0
    assert "thank" in runs[0].text and "you" in runs[0].text
    assert runs[1].indices == [4]
    assert runs[1].start == 7.0 and runs[1].end == 8.0


def test_candidate_speakers_distinct_ordered_by_speaking_time_desc():
    segs = [
        seg("A", "", 0.0, 1.0),   # A: 1s
        seg("B", "", 0.0, 5.0),   # B: 5s
        seg("A", "", 0.0, 3.0),   # A: +3 -> 4s
        seg(None, "", 0.0, 1.0),  # ?? excluded
    ]
    assert r.candidate_speakers(segs) == ["B", "A"]


def test_assign_segments_speaker_sets_segment_and_nested_words():
    segs = [
        seg("Alice", "hi", 0.0, 1.0, words=[{"word": "hi"}]),
        seg(None, "thanks", 2.0, 3.0, words=[{"word": "thanks"}]),
    ]
    r.assign_segments_speaker(segs, [1], "Bob")
    assert segs[1]["speaker"] == "Bob"
    assert segs[1]["words"][0]["speaker"] == "Bob"
    # untouched
    assert segs[0]["speaker"] == "Alice"
    assert segs[0]["words"][0].get("speaker") is None
