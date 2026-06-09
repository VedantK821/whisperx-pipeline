"""The reassign_unassigned interactive loop, driven with injected keys and a
fake ffplay process. Confirms it records decisions, cuts audio the instant the
displayed run changes (the behavior the existing rename loop is missing), and
skips cleanly when there's no TTY.
"""
import io

import pytest
from rich.console import Console

import src.rename as r


class FakeProc:
    def __init__(self):
        self._alive = True
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False


class FakeStdin:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def _runs(n=3):
    return [r.UnassignedRun(indices=[i], start=float(i), end=float(i) + 1, text=f"t{i}", line_count=1)
            for i in range(n)]


def _drive(monkeypatch, tmp_path, keys, *, runs=None, tty=True, can_play=True):
    runs = runs if runs is not None else _runs(3)
    seq = list(keys)

    def fake_read_key():
        return seq.pop(0) if seq else r.KEY_EOF

    plays = []

    def fake_play(audio_path, start, duration=10.0):
        p = FakeProc()
        plays.append((p, start))
        return p

    monkeypatch.setattr(r, "read_key", fake_read_key)
    monkeypatch.setattr(r, "play_snippet", fake_play)
    monkeypatch.setattr(r.sys, "stdin", FakeStdin(tty))

    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    console = Console(file=io.StringIO())
    result = r.reassign_unassigned(runs, ["Alice", "Bob"], audio, console, ffplay_available=can_play)
    return result, plays


def test_loop_records_assignments(monkeypatch, tmp_path):
    result, _ = _drive(monkeypatch, tmp_path, ["1", "2", "s"])
    assert result == {0: "Alice", 1: "Bob"}


def test_loop_cuts_audio_when_advancing(monkeypatch, tmp_path):
    result, plays = _drive(monkeypatch, tmp_path, [" ", "1", "s", "s"])
    assert len(plays) == 1
    proc, start = plays[0]
    assert start == 0.0            # played the current run's audio
    assert proc.terminated is True  # cut when we advanced past it
    assert result == {0: "Alice"}


def test_loop_returns_none_on_eof_abort(monkeypatch, tmp_path):
    result, _ = _drive(monkeypatch, tmp_path, [r.KEY_EOF])
    assert result is None


def test_loop_skips_without_a_tty(monkeypatch, tmp_path):
    result, plays = _drive(monkeypatch, tmp_path, ["1"], tty=False)
    assert result == {}
    assert plays == []
