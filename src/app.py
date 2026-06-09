"""Cyberpunk terminal UI for WhisperX transcription.

Default flow:
  1. Scan `incoming/` and show a status table.
  2. Skip files that already have a transcript (unless --force).
  3. Prompt to transcribe / force-all / watch / quit.
  4. Optionally keep watching for new files.

Usage:
  python -m src.app                 # interactive
  python -m src.app -y              # auto-transcribe new files, then exit
  python -m src.app -y --watch      # transcribe new, then keep watching
  python -m src.app --force         # retranscribe everything
  python -m src.app --files a.m4a   # restrict to these files
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any

# Force UTF-8 stdout/stderr so the cyberpunk box-drawing glyphs render on
# legacy Windows terminals (default codepage is cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _silence_noisy_libraries() -> None:
    """Mute the chatty libraries that punch through Rich's Live region.

    We keep ERROR-level messages so genuine failures still surface, and we
    strip the cosmetic UserWarnings whisperx + pyannote emit on every import.
    """
    for name in (
        "whisperx", "whisperx.asr", "whisperx.alignment", "whisperx.diarize",
        "whisperx.vads", "whisperx.vads.pyannote",
        "pyannote", "pyannote.audio",
        "lightning", "lightning.pytorch",
        "speechbrain", "transformers", "huggingface_hub",
        "torch", "torio",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Module-level reference so the faulthandler dump file is not garbage-collected
# (and thus closed) for the lifetime of the process.
_fault_log = None


def _setup_logging(verbose: bool) -> Path:
    """Configure console logging plus a persistent file log and crash dump.

    The console keeps its old behaviour (WARNING, or DEBUG with -v). On top of
    that we always attach a RotatingFileHandler that writes DEBUG-level detail to
    logs/whisperx.log. The file handler writes straight to the file descriptor,
    so — unlike the console stream — the Rich Live TUI's `redirect_stderr=True`
    cannot swallow it. That is what lets us see *where* a batch run died.

    faulthandler dumps a native stack to logs/crash.log on a fatal signal
    (SIGSEGV/SIGABRT/…). A CUDA/torch crash bypasses Python's try/except entirely,
    so this is the only way such a death leaves a trail.

    Returns the path of the main log file.
    """
    global _fault_log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "whisperx.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers decide what actually gets emitted
    root.handlers.clear()         # drop any basicConfig/prior-run handlers

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG if verbose else logging.WARNING)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _fault_log = open(LOG_DIR / "crash.log", "a", encoding="utf-8")
    faulthandler.enable(file=_fault_log, all_threads=True)

    return log_file


from rich.console import Console, Group
from rich.live import Live
from rich.prompt import Prompt
from watchdog.observers import Observer

from . import ui
from .config import Config
from .transcribe import STAGES, AUDIO_EXTS, transcribe_file
from .watcher import IncomingHandler, is_candidate, wait_until_stable

console = Console()
log = logging.getLogger("app")

PENDING, RUNNING, DONE, FAILED, SKIPPED, QUEUED = (
    "pending", "running", "done", "failed", "skipped", "queued"
)


# ─── State ──────────────────────────────────────────────────────────────────

@dataclass
class Row:
    path: Path
    status: str
    started: float | None = None
    finished: float | None = None
    error: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    @property
    def elapsed_text(self) -> str:
        if self.started is None:
            return ""
        end = self.finished if self.finished is not None else time.monotonic()
        return f"{end - self.started:.1f}s"


@dataclass
class LiveState:
    """Shared mutable state read by the renderer on every tick."""
    rows: list[Row] = field(default_factory=list)
    current_file: str | None = None
    current_size_text: str = ""
    current_duration_s: float | None = None
    current_stage: str | None = None
    current_stage_progress: float = 0.0   # 0..1, real callback fraction
    completed_stages: set[str] = field(default_factory=set)
    current_started: float | None = None
    frame: int = 0
    # REVIEW stage state — armed when on_review is waiting on a keypress.
    review_examples: list[Any] | None = None   # list[SpeakerExample]
    review_deadline: float | None = None       # monotonic deadline
    rename_pending: list[Row] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def begin_file(self, row: Row) -> None:
        with self.lock:
            self.current_file = row.name
            self.current_size_text = ui._fmt_size(row.size)
            self.current_duration_s = None
            self.current_stage = None
            self.current_stage_progress = 0.0
            self.completed_stages = set()
            self.current_started = time.monotonic()

    def end_file(self) -> None:
        with self.lock:
            self.current_file = None
            self.current_size_text = ""
            self.current_duration_s = None
            self.current_stage = None
            self.current_stage_progress = 0.0
            self.completed_stages = set()
            self.current_started = None
            self.review_examples = None
            self.review_deadline = None

    def on_stage(self, stage: str, info: dict[str, Any]) -> None:
        with self.lock:
            if self.current_stage and self.current_stage != stage:
                self.completed_stages.add(self.current_stage)
            self.current_stage = stage
            self.current_stage_progress = 0.0
            dur = info.get("audio_duration")
            if dur:
                self.current_duration_s = float(dur)

    def on_progress(self, stage: str, fraction: float) -> None:
        with self.lock:
            # Only accept progress for the active stage to avoid out-of-order updates.
            if self.current_stage == stage:
                self.current_stage_progress = max(self.current_stage_progress, fraction)


# ─── GPU helpers ────────────────────────────────────────────────────────────

def _gpu_info() -> tuple[str | None, float | None]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated(0)
        pct = used / total * 100.0 if total else None
        return name, pct
    except Exception:
        return None, None


# ─── Renderer ───────────────────────────────────────────────────────────────

def render(state: LiveState, cfg: Config) -> Group:
    gpu_name, vram_pct = _gpu_info()
    with state.lock:
        elapsed = (time.monotonic() - state.current_started) if state.current_started else 0.0
        current_file = state.current_file
        current_size_text = state.current_size_text
        current_duration_s = state.current_duration_s
        current_stage = state.current_stage
        current_progress = state.current_stage_progress
        completed = set(state.completed_stages)
        rows_snapshot = list(state.rows)
        frame = state.frame
        review_deadline = state.review_deadline

    review_seconds_remaining = None
    if review_deadline is not None:
        review_seconds_remaining = max(0.0, review_deadline - time.monotonic())

    done = sum(1 for r in rows_snapshot if r.status == DONE)
    failed = sum(1 for r in rows_snapshot if r.status == FAILED)
    total_actionable = sum(
        1 for r in rows_snapshot if r.status in (DONE, FAILED, RUNNING, PENDING, QUEUED)
    )

    return Group(
        ui.banner(frame=frame),
        ui.system_panel(cfg, gpu_name=gpu_name, vram_pct=vram_pct),
        ui.now_transcribing_panel(
            filename=current_file,
            size_text=current_size_text,
            duration_s=current_duration_s,
            current_stage=current_stage,
            stage_progress=current_progress,
            completed=completed,
            elapsed_s=elapsed,
            frame=frame,
            stages=STAGES,
            review_seconds_remaining=review_seconds_remaining,
        ),
        ui.queue_table(rows_snapshot),
        ui.batch_progress(done, failed, total_actionable),
    )


# ─── Helpers ────────────────────────────────────────────────────────────────

def upsert(state: LiveState, path: Path, cfg: Config) -> Row:
    with state.lock:
        for r in state.rows:
            if r.path == path:
                return r
        nested = cfg.transcripts_dir / path.stem / f"{path.stem}.json"
        legacy = cfg.transcripts_dir / f"{path.stem}.json"
        status = SKIPPED if (nested.exists() or legacy.exists()) else PENDING
        row = Row(path=path, status=status)
        state.rows.append(row)
        return row


def scan_incoming(state: LiveState, cfg: Config) -> None:
    for f in sorted(cfg.incoming_dir.rglob("*")):
        if is_candidate(f):
            upsert(state, f, cfg)


def run_one(
    row: Row,
    state: LiveState,
    cfg: Config,
    *,
    on_review=None,
) -> None:
    row.status = RUNNING
    row.started = time.monotonic()
    state.begin_file(row)
    try:
        transcribe_file(
            row.path, cfg,
            on_stage=state.on_stage,
            on_progress=state.on_progress,
            on_review=on_review,
        )
        row.status = DONE
        # If the resulting transcript still has SPEAKER_XX labels, flag this
        # file for the post-batch catch-up prompt.
        from . import rename
        out_json = cfg.transcripts_dir / row.path.stem / f"{row.path.stem}.json"
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
            segs = data.get("segments") or []
            if rename.unmapped_speakers(segs) or rename.unassigned_segments(segs):
                with state.lock:
                    state.rename_pending.append(row)
        except (OSError, json.JSONDecodeError):
            pass
    except Exception as e:
        row.status = FAILED
        row.error = str(e).splitlines()[0][:200]
        log.exception("Transcription failed for %s", row.path)
    finally:
        row.finished = time.monotonic()
        state.end_file()


def _run_rename_for_transcript(
    transcript_json: Path,
    audio_path: Path | None,
    cfg: Config,
    *,
    ffplay_avail: bool,
) -> None:
    """Post-hoc rename for a single transcript: name the diarized speakers, then
    fold any SPEAKER_?? lines into a known speaker, and re-emit outputs once.
    Used by both the post-batch catch-up and the startup `r` option.
    """
    from . import rename
    from .transcribe import write_outputs

    stem = transcript_json.stem
    out_dir = transcript_json.parent

    if not transcript_json.exists():
        console.print(f"[{ui.NEON_RED}]Transcript missing: {transcript_json}[/{ui.NEON_RED}]")
        return

    data = json.loads(transcript_json.read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    persisted = rename.load_persisted_mapping(out_dir, stem)
    if persisted:
        rename.apply_speaker_mapping(segments, persisted)

    remaining = rename.unmapped_speakers(segments)
    runs = rename.build_unassigned_runs(segments)
    if not remaining and not runs:
        console.print(f"[{ui.DIM}]{stem}: nothing to rename.[/{ui.DIM}]")
        return

    audio_for_play = audio_path if (audio_path and audio_path.exists()) else None
    can_play = ffplay_avail and audio_for_play is not None
    console.rule(f"[{ui.NEON_MAGENTA} bold]Renaming {stem}[/{ui.NEON_MAGENTA} bold]")

    # 1) Name the diarized speakers.
    new_mapping: dict[str, str] = {}
    if remaining:
        examples = [e for e in rename.build_speaker_examples(segments) if e.label in remaining]
        mapping = rename.interactive_rename(
            examples, audio_for_play, console, ffplay_available=can_play,
        )
        if mapping is None:   # aborted
            return
        new_mapping = mapping
        rename.apply_speaker_mapping(segments, new_mapping)

    # 2) Fold the SPEAKER_?? runs into a known speaker.
    decided = 0
    if runs:
        candidates = rename.candidate_speakers(segments)
        decisions = rename.reassign_unassigned(
            runs, candidates, audio_for_play, console, ffplay_available=can_play,
        )
        if decisions:
            for run_idx, name in decisions.items():
                rename.assign_segments_speaker(segments, runs[run_idx].indices, name)
            decided = len(decisions)

    if not new_mapping and not decided:
        console.print(f"[{ui.DIM}]{stem}: no changes.[/{ui.DIM}]")
        return

    # 3) Persist the label->name mapping and re-emit all outputs once.
    if new_mapping:
        merged = dict(persisted or {})
        merged.update(new_mapping)
        audio_filename = audio_path.name if audio_path else f"{stem}.audio"
        rename.save_persisted_mapping(out_dir, stem, audio_filename, merged)
    write_outputs(data, transcript_json.with_suffix(""))
    console.print(f"[{ui.NEON_GREEN}]✓ Rewrote outputs for {stem}.[/{ui.NEON_GREEN}]")


def _parse_pending_choice(raw: str, count: int) -> "str | int | None":
    """Parse a rename-pending picker choice. Returns 'all', 'quit', a 0-based
    index into the list, or None for unrecognised input."""
    s = raw.strip().lower()
    if s in ("a", "all"):
        return "all"
    if s in ("", "q", "quit"):
        return "quit"
    if s.isdigit():
        n = int(s)
        if 1 <= n <= count:
            return n - 1
    return None


def _render_pending_list(pending: list) -> "Any":
    """Build the numbered [ RENAME PENDING ] panel."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    t = Table.grid(padding=(0, 3))
    t.add_column(justify="right", style=ui.DIM, no_wrap=True)
    t.add_column(overflow="fold")
    t.add_column(style=ui.NEON_YELLOW, no_wrap=True)
    t.add_column(style=ui.DIM, no_wrap=True)
    for i, p in enumerate(pending, 1):
        bits = []
        if p.unmapped:
            bits.append(f"{len(p.unmapped)} unnamed")
        if p.unassigned_count:
            bits.append(f"{p.unassigned_count} ??")
        status = " · ".join(bits) or "—"
        audio = "♪ audio" if p.audio_path else "— no audio"
        t.add_row(str(i), p.transcript_json.stem, status, audio)

    return Panel(
        t,
        title=Text("[ RENAME PENDING ]", style=f"{ui.NEON_MAGENTA} bold"),
        title_align="left",
        border_style=ui.GRID,
        padding=(0, 1),
    )


def _pick_and_rename_pending(pending: list, cfg: Config, console, *, ffplay_avail: bool) -> None:
    """Let the user pick which pending transcript(s) to rename — one at a time,
    'a' for all, or 'q' to cancel. Renamed picks drop off the list so they can
    keep choosing until done."""
    remaining = list(pending)
    while remaining:
        console.print(_render_pending_list(remaining))
        raw = Prompt.ask(
            f"[bold]Pick[/bold] 1-{len(remaining)} · "
            f"[{ui.NEON_MAGENTA}]a[/{ui.NEON_MAGENTA}]ll · "
            f"[{ui.DIM}]q[/{ui.DIM}]uit",
            default="q",
            show_default=False,
        )
        choice = _parse_pending_choice(raw, len(remaining))
        if choice == "quit":
            return
        if choice == "all":
            for p in remaining:
                _run_rename_for_transcript(
                    p.transcript_json, p.audio_path, cfg, ffplay_avail=ffplay_avail
                )
            return
        if choice is None:
            console.print(
                f"[{ui.NEON_RED}]Enter a number 1-{len(remaining)}, 'a', or 'q'.[/{ui.NEON_RED}]"
            )
            continue
        p = remaining.pop(choice)
        _run_rename_for_transcript(
            p.transcript_json, p.audio_path, cfg, ffplay_avail=ffplay_avail
        )
    console.print(f"[{ui.NEON_GREEN}]No more pending transcripts.[/{ui.NEON_GREEN}]")


def _post_batch_catchup(state: LiveState, cfg: Config, *, ffplay_avail: bool) -> None:
    """Prompt to name remaining unnamed speakers across just-finished batch."""
    pending = list(state.rename_pending)
    state.rename_pending.clear()
    if not pending:
        return

    prompt_msg = (f"\n[bold]{len(pending)} file(s) still need speaker cleanup. "
                  f"Rename now?[/bold] "
                  f"[[{ui.NEON_CYAN}]Y[/{ui.NEON_CYAN}]]es / "
                  f"[[{ui.DIM}]n[/{ui.DIM}]]o")
    choice = Prompt.ask(prompt_msg, default="y", show_default=False).strip().lower()
    if choice in ("n", "no"):
        return

    for row in pending:
        stem = row.path.stem
        transcript_json = cfg.transcripts_dir / stem / f"{stem}.json"
        _run_rename_for_transcript(transcript_json, row.path, cfg, ffplay_avail=ffplay_avail)


def _make_on_review(
    state: LiveState,
    live: "Live",
    args: argparse.Namespace,
    ffplay_avail: bool,
):
    """Return a `ReviewCallback` closure suitable for transcribe_file.

    1. Arms a countdown by setting state.review_examples / review_deadline.
    2. Waits up to args.review_timeout for any key (non-blocking).
    3. If a key arrives: pauses Live, runs interactive_rename, resumes Live,
       returns the resulting mapping (or None on abort).
    4. If timeout: returns None.
    """
    from . import rename

    def _on_review(examples, audio_path):
        if args.no_review:
            return None

        with state.lock:
            state.review_examples = examples
            state.review_deadline = time.monotonic() + args.review_timeout

        key = rename.wait_for_keypress(args.review_timeout)

        with state.lock:
            state.review_examples = None
            state.review_deadline = None

        if key is None:
            return None

        # User engaged — pause Live for blocking prompts.
        live.stop()
        try:
            return rename.interactive_rename(
                examples, audio_path, console,
                ffplay_available=ffplay_avail,
            )
        finally:
            live.start(refresh=True)

    return _on_review


def _clear_terminal() -> None:
    """Wipe visible screen and scrollback so the Live dashboard starts clean.

    ESC[2J = clear visible screen, ESC[3J = clear scrollback, ESC[H = home cursor.
    Without this, the snapshot printed before the y/n prompt stays in scrollback
    above the Live region, making the dashboard look duplicated.
    """
    console.file.write("\033[H\033[2J\033[3J")
    console.file.flush()


def _startup_countdown(new_rows: list[Row], timeout: float) -> bool:
    """Show the new recordings found and a short countdown. Return True to
    proceed (auto-transcribe), or False if the user pressed a key (wants the
    options menu). A timeout <= 0 proceeds immediately with no countdown.
    """
    from . import rename
    from rich.live import Live
    from rich.text import Text

    if timeout <= 0:
        return True

    listing = "\n".join(f"   • {r.name}" for r in new_rows)
    console.print(
        f"\n[{ui.NEON_CYAN}]▶ {len(new_rows)} new recording(s) found:[/{ui.NEON_CYAN}]\n"
        f"[{ui.DIM}]{listing}[/{ui.DIM}]\n"
    )

    width = 24

    def line(remaining: float) -> Text:
        frac = max(0.0, min(1.0, 1.0 - remaining / timeout))
        filled = int(frac * width)
        secs = int(remaining) + 1
        t = Text(f"  Transcribing in {secs}s…  press any key for options  ", style=ui.DIM)
        t.append("█" * filled, style=ui.NEON_CYAN)
        t.append("░" * (width - filled), style=ui.GRID)
        return t

    deadline = time.monotonic() + timeout
    with Live(line(timeout), console=console, refresh_per_second=20, transient=True) as live:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            live.update(line(remaining))
            if rename.wait_for_keypress(min(0.08, remaining)) is not None:
                return False


def process_batch(
    rows_to_run: list[Row],
    state: LiveState,
    cfg: Config,
    *,
    args: argparse.Namespace,
    ffplay_avail: bool,
) -> None:
    stop = Event()

    _clear_terminal()

    with Live(
        render(state, cfg),
        console=console,
        refresh_per_second=10,
        screen=False,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as live:
        on_review = _make_on_review(state, live, args=args, ffplay_avail=ffplay_avail)

        def _ticker():
            while not stop.is_set():
                with state.lock:
                    state.frame += 1
                live.update(render(state, cfg))
                time.sleep(0.1)

        ticker = Thread(target=_ticker, daemon=True)
        ticker.start()
        try:
            for row in rows_to_run:
                run_one(row, state, cfg, on_review=on_review)
                live.update(render(state, cfg))
        finally:
            stop.set()
            ticker.join()
            live.update(render(state, cfg))


def watch_loop(
    state: LiveState,
    cfg: Config,
    *,
    args: argparse.Namespace,
    ffplay_avail: bool,
) -> None:
    queue: Queue[Path] = Queue()

    def on_new(p: Path) -> None:
        queue.put(p)

    observer = Observer()
    observer.schedule(IncomingHandler(on_new), str(cfg.incoming_dir), recursive=True)
    observer.start()

    stop = Event()

    _clear_terminal()
    console.rule(f"[{ui.NEON_MAGENTA} bold]◉ WATCHING incoming/ — Ctrl-C to stop[/{ui.NEON_MAGENTA} bold]")
    with Live(
        render(state, cfg),
        console=console,
        refresh_per_second=10,
        screen=False,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as live:
        on_review = _make_on_review(state, live, args=args, ffplay_avail=ffplay_avail)

        def _ticker():
            while not stop.is_set():
                with state.lock:
                    state.frame += 1
                live.update(render(state, cfg))
                time.sleep(0.1)

        ticker = Thread(target=_ticker, daemon=True)
        ticker.start()
        try:
            while True:
                try:
                    p = queue.get(timeout=0.25)
                except Empty:
                    live.update(render(state, cfg))
                    continue
                row = upsert(state, p, cfg)
                if row.status in (DONE, SKIPPED, RUNNING):
                    continue
                row.status = QUEUED
                live.update(render(state, cfg))
                if not wait_until_stable(p):
                    row.status = FAILED
                    row.error = "vanished before stable"
                    live.update(render(state, cfg))
                    continue
                run_one(row, state, cfg, on_review=on_review)
                live.update(render(state, cfg))
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            ticker.join()
            observer.stop()
            observer.join()
            live.update(render(state, cfg))


# ─── Entry point ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Retranscribe files that already have a transcript.")
    parser.add_argument("--watch", action="store_true",
                        help="After the initial pass, keep watching incoming/.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the prompt; just process new files.")
    parser.add_argument("--files", nargs="+", type=Path,
                        help="Restrict to these files (relative to incoming/ or absolute).")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-review", action="store_true",
                        help="Disable the inline REVIEW prompt (transcribe without name prompts).")
    parser.add_argument("--review-timeout", type=float, default=7.0,
                        help="Seconds the inline REVIEW prompt waits for a keypress (default: 7).")
    parser.add_argument("--start-timeout", type=float, default=3.0,
                        help="Seconds the launch countdown waits before auto-transcribing "
                             "new files (0 = start immediately; default: 3).")
    args = parser.parse_args(argv)

    log_file = _setup_logging(args.verbose)
    if not args.verbose:
        _silence_noisy_libraries()
    log.info("=== run start: argv=%s ===", argv if argv is not None else sys.argv[1:])
    console.print(f"[{ui.DIM}]Logging to {log_file}[/{ui.DIM}]")

    cfg = Config.load()
    state = LiveState()

    from . import rename
    ffplay_avail = rename.ffplay_available()

    if args.files:
        for f in args.files:
            p = f if f.is_absolute() else cfg.incoming_dir / f
            if not p.exists():
                console.print(f"[{ui.NEON_RED}]Not found:[/{ui.NEON_RED}] {p}")
                return 2
            upsert(state, p, cfg)
    else:
        scan_incoming(state, cfg)

    console.print(render(state, cfg))

    if not state.rows and not args.watch:
        console.print(f"\n[{ui.DIM}]Drop audio files into {cfg.incoming_dir} and re-run.[/{ui.DIM}]")
        return 0

    new = [r for r in state.rows if r.status == PENDING]
    skipped = [r for r in state.rows if r.status == SKIPPED]

    rename_pending_existing = rename.find_rename_pending(cfg.transcripts_dir, cfg.incoming_dir)
    rename_chip = (
        f" / [[{ui.NEON_GREEN}]r[/{ui.NEON_GREEN}]]ename pending ({len(rename_pending_existing)})"
        if rename_pending_existing else ""
    )

    if args.force:
        for r in state.rows:
            r.status = PENDING
            r.started = r.finished = None
            r.error = None
        to_run = list(state.rows)
    elif args.yes:
        to_run = new
    else:
        if new and _startup_countdown(new, args.start_timeout):
            # New recordings found and the countdown elapsed — just go.
            to_run = new
        else:
            if new and skipped:
                prompt = (f"\n[bold]Transcribe {len(new)} new?[/bold]  "
                          f"[[{ui.NEON_CYAN}]Y[/{ui.NEON_CYAN}]]es / "
                          f"[[{ui.NEON_MAGENTA}]a[/{ui.NEON_MAGENTA}]]ll force"
                          f"{rename_chip}"
                          f" / [[{ui.NEON_YELLOW}]w[/{ui.NEON_YELLOW}]]atch / "
                          f"[[{ui.DIM}]q[/{ui.DIM}]]uit")
                default = "y"
            elif new:
                prompt = (f"\n[bold]Transcribe {len(new)} new?[/bold]  "
                          f"[[{ui.NEON_CYAN}]Y[/{ui.NEON_CYAN}]]es"
                          f"{rename_chip}"
                          f" / [[{ui.NEON_YELLOW}]w[/{ui.NEON_YELLOW}]]atch / "
                          f"[[{ui.DIM}]q[/{ui.DIM}]]uit")
                default = "y"
            elif skipped:
                prompt = (f"\nAll caught up. "
                          f"[[{ui.NEON_MAGENTA}]a[/{ui.NEON_MAGENTA}]]ll force"
                          f"{rename_chip}"
                          f" / [[{ui.NEON_YELLOW}]w[/{ui.NEON_YELLOW}]]atch / "
                          f"[[{ui.DIM}]Q[/{ui.DIM}]]uit")
                default = "q"
            else:
                prompt = (f"\nNothing here."
                          f"{rename_chip}"
                          f" / [[{ui.NEON_YELLOW}]w[/{ui.NEON_YELLOW}]]atch / "
                          f"[[{ui.DIM}]Q[/{ui.DIM}]]uit")
                default = "q"

            choice = Prompt.ask(prompt, default=default, show_default=False).strip().lower()
            if choice in ("q", "n", ""):
                return 0
            if choice == "r" and rename_pending_existing:
                _pick_and_rename_pending(
                    rename_pending_existing, cfg, console, ffplay_avail=ffplay_avail
                )
                return 0
            if choice == "a":
                for r in state.rows:
                    r.status = PENDING
                    r.started = r.finished = None
                    r.error = None
                to_run = list(state.rows)
            elif choice == "w":
                to_run = new
                args.watch = True
            else:
                to_run = new

    if to_run:
        process_batch(to_run, state, cfg, args=args, ffplay_avail=ffplay_avail)
        if state.rename_pending and not args.no_review:
            _post_batch_catchup(state, cfg, ffplay_avail=ffplay_avail)

    failed = sum(1 for r in state.rows if r.status == FAILED)

    if args.watch:
        watch_loop(state, cfg, args=args, ffplay_avail=ffplay_avail)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
