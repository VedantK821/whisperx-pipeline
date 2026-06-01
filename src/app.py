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
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
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
            if rename.unmapped_speakers(data.get("segments") or []):
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
    """Post-hoc rename for a single transcript. Used by both
    post-batch catch-up and the startup `r` option.
    """
    from . import rename

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
    if not remaining:
        console.print(f"[{ui.DIM}]{stem}: nothing to rename.[/{ui.DIM}]")
        return

    examples = [e for e in rename.build_speaker_examples(segments) if e.label in remaining]
    audio_for_play = audio_path if (audio_path and audio_path.exists()) else None
    console.rule(f"[{ui.NEON_MAGENTA} bold]Renaming {stem}[/{ui.NEON_MAGENTA} bold]")
    mapping = rename.interactive_rename(
        examples, audio_for_play, console,
        ffplay_available=ffplay_avail and audio_for_play is not None,
    )
    if not mapping:
        return

    merged = dict(persisted or {})
    merged.update(mapping)
    audio_filename = audio_path.name if audio_path else f"{stem}.audio"
    rename.save_persisted_mapping(out_dir, stem, audio_filename, merged)
    rename.rewrite_outputs_with_mapping(transcript_json, merged)
    console.print(f"[{ui.NEON_GREEN}]✓ Rewrote outputs for {stem}.[/{ui.NEON_GREEN}]")


def _post_batch_catchup(state: LiveState, cfg: Config, *, ffplay_avail: bool) -> None:
    """Prompt to name remaining unnamed speakers across just-finished batch."""
    pending = list(state.rename_pending)
    state.rename_pending.clear()
    if not pending:
        return

    prompt_msg = (f"\n[bold]{len(pending)} file(s) still have unnamed speakers. "
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        _silence_noisy_libraries()

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
            for p in rename_pending_existing:
                _run_rename_for_transcript(
                    p.transcript_json, p.audio_path, cfg,
                    ffplay_avail=ffplay_avail,
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
