"""Key-driven state machine for the SPEAKER_?? reassignment pass.

Pure: feed it key tokens, assert decisions/cursor. Mirrors test_rename_nav for
the speaker-naming RenameNav. No terminal, no audio.
"""
import src.rename as r


def mkruns(n):
    return [
        r.UnassignedRun(indices=[i], start=float(i), end=float(i) + 1, text=f"t{i}", line_count=1)
        for i in range(n)
    ]


def nav(n_runs=3, candidates=("Alice", "Bob")):
    return r.UnassignedNav(runs=mkruns(n_runs), candidates=list(candidates))


def test_digit_assigns_current_run_and_advances():
    nv = nav(3)
    nv.step("1")
    assert nv.decisions == {0: "Alice"}
    assert nv.run_idx == 1


def test_skip_advances_without_a_decision():
    nv = nav(3)
    nv.step("s")
    assert nv.decisions == {}
    assert nv.run_idx == 1


def test_assigning_the_last_run_finishes():
    nv = nav(1)
    nv.step("2")
    assert nv.decisions == {0: "Bob"}
    assert nv.finished is True


def test_new_name_flow_types_then_commits():
    nv = nav(2)
    nv.step("n")
    for ch in "Bob":
        nv.step(ch)
    nv.step(r.KEY_ENTER)
    assert nv.decisions == {0: "Bob"}
    assert nv.run_idx == 1


def test_new_name_empty_enter_cancels_and_stays():
    nv = nav(2)
    nv.step("n")
    nv.step(r.KEY_ENTER)
    assert nv.decisions == {}
    assert nv.run_idx == 0


def test_all_remaining_assigns_the_rest_and_finishes():
    nv = nav(3)
    nv.step("s")          # skip run 0 -> stays undecided
    nv.step("a")          # enter all-remaining mode
    nv.step("1")          # assign all remaining (runs 1,2) -> Alice
    assert nv.decisions == {1: "Alice", 2: "Alice"}
    assert nv.finished is True


def test_space_returns_play_in_command_mode():
    nv = nav(2)
    assert nv.step(" ") == r.PLAY
    assert nv.run_idx == 0


def test_space_is_a_typed_char_while_naming():
    nv = nav(2)
    nv.step("n")
    assert nv.step(" ") is None     # not PLAY while typing
    nv.step("x")
    nv.step(r.KEY_ENTER)
    assert nv.decisions[0] == "x"   # leading space stripped on commit


def test_up_down_navigate_without_deciding():
    nv = nav(3)
    nv.step(r.KEY_DOWN)
    assert nv.run_idx == 1
    nv.step(r.KEY_DOWN)
    nv.step(r.KEY_DOWN)             # clamps at last
    assert nv.run_idx == 2
    nv.step(r.KEY_UP)
    assert nv.run_idx == 1
    assert nv.decisions == {}


def test_esc_finishes_and_eof_aborts():
    nv = nav(3)
    nv.step(r.KEY_ESC)
    assert nv.finished is True and nv.aborted is False
    nv2 = nav(3)
    nv2.step(r.KEY_EOF)
    assert nv2.aborted is True


def test_out_of_range_digit_is_ignored():
    nv = nav(2, candidates=("Alice",))   # only one candidate
    nv.step("2")
    assert nv.decisions == {}
    assert nv.run_idx == 0
