"""Cyberpunk Rich UI primitives shared by the app."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─── Palette ────────────────────────────────────────────────────────────────
NEON_CYAN    = "bright_cyan"
NEON_MAGENTA = "magenta1"
NEON_YELLOW  = "bright_yellow"
NEON_GREEN   = "spring_green2"
NEON_RED     = "bright_red"
NEON_BLUE    = "deep_sky_blue1"
GRID         = "grey39"
DIM          = "grey50"


# ─── Status icons (kept ASCII-safe-ish so Windows terms render cleanly) ─────
STATUS_GLYPHS = {
    "pending": ("◇", NEON_YELLOW),
    "queued":  ("◈", NEON_MAGENTA),
    "running": ("◉", NEON_CYAN),
    "done":    ("◆", NEON_GREEN),
    "failed":  ("✗", NEON_RED),
    "skipped": ("─", DIM),
}


# ─── Banner ─────────────────────────────────────────────────────────────────
BANNER_LINES = (
    "██╗    ██╗██╗  ██╗██╗███████╗██████╗ ███████╗██████╗ ██╗  ██╗",
    "██║    ██║██║  ██║██║██╔════╝██╔══██╗██╔════╝██╔══██╗╚██╗██╔╝",
    "██║ █╗ ██║███████║██║███████╗██████╔╝█████╗  ██████╔╝ ╚███╔╝ ",
    "██║███╗██║██╔══██║██║╚════██║██╔═══╝ ██╔══╝  ██╔══██╗ ██╔██╗ ",
    "╚███╔███╔╝██║  ██║██║███████║██║     ███████╗██║  ██║██╔╝ ██╗",
    " ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝",
)

# Gradient line colors (hex) — cycles top→bottom
BANNER_COLORS = ("#00f0ff", "#00d6ff", "#7d4dff", "#c026ff", "#ff2bd6", "#ff14a3")
TAGLINE = "▰▰▰  diarize · transcribe · stream  ▰▰▰"


def banner(frame: int = 0) -> Panel:
    lines: list[Text] = []
    for i, raw in enumerate(BANNER_LINES):
        color = BANNER_COLORS[(i + frame // 6) % len(BANNER_COLORS)]
        lines.append(Text(raw, style=color))
    tag = Text(TAGLINE, style=f"{NEON_MAGENTA} bold")
    body = Group(*[Align.center(l) for l in lines], Align.center(tag))
    return Panel(
        body,
        box=box.DOUBLE_EDGE,
        border_style=NEON_CYAN,
        padding=(0, 2),
    )


# ─── System / config panel ──────────────────────────────────────────────────

def system_panel(cfg, gpu_name: str | None = None, vram_pct: float | None = None) -> Panel:
    rows = Table.grid(padding=(0, 2), expand=True)
    rows.add_column(style=DIM, no_wrap=True)
    rows.add_column(style=f"{NEON_CYAN} bold")
    rows.add_column(style=DIM, no_wrap=True)
    rows.add_column(style=f"{NEON_CYAN} bold")

    rows.add_row("MODEL",   cfg.model,        "DEVICE",  cfg.device.upper())
    rows.add_row("COMPUTE", cfg.compute_type, "LANG",    cfg.language or "auto")

    if gpu_name:
        vram_text = f"{vram_pct:.0f}% used" if vram_pct is not None else "—"
        rows.add_row("GPU", gpu_name, "VRAM", vram_text)

    rows.add_row("IN",  str(cfg.incoming_dir),    "OUT", str(cfg.transcripts_dir))

    return Panel(
        rows,
        title=Text("[ SYSTEM ]", style=f"{NEON_MAGENTA} bold"),
        title_align="left",
        box=box.HEAVY,
        border_style=GRID,
        padding=(0, 1),
    )


# ─── Now-transcribing panel ─────────────────────────────────────────────────

PULSE_FRAMES = ("▰▱▱▱▱▱▱▱", "▰▰▱▱▱▱▱▱", "▰▰▰▱▱▱▱▱", "▰▰▰▰▱▱▱▱",
                "▱▰▰▰▰▱▱▱", "▱▱▰▰▰▰▱▱", "▱▱▱▰▰▰▰▱", "▱▱▱▱▰▰▰▰",
                "▱▱▱▱▱▰▰▰", "▱▱▱▱▱▱▰▰", "▱▱▱▱▱▱▱▰", "▱▱▱▱▱▱▱▱")


def _pulse(frame: int) -> str:
    return PULSE_FRAMES[frame % len(PULSE_FRAMES)]


def _stage_chips(stages: Iterable[str], current: str | None, completed: set[str]) -> Text:
    out = Text()
    for s in stages:
        if s in completed:
            out.append(f" ✓ {s} ", style=f"{NEON_GREEN} bold")
        elif s == current:
            out.append(f" ▶ {s} ", style=f"black on {NEON_CYAN} bold")
        else:
            out.append(f" · {s} ", style=DIM)
        out.append(" ", style=DIM)
    return out


def _bar(pct: float, width: int = 30, filled_style: str = NEON_CYAN, empty_style: str = GRID) -> Text:
    pct = max(0.0, min(1.0, pct))
    filled = int(round(pct * width))
    t = Text()
    t.append("█" * filled, style=filled_style)
    t.append("░" * (width - filled), style=empty_style)
    return t


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def now_transcribing_panel(
    *,
    filename: str | None,
    size_text: str,
    duration_s: float | None,
    current_stage: str | None,
    stage_progress: float,
    completed: set[str],
    elapsed_s: float,
    frame: int,
    stages: tuple[str, ...],
) -> Panel:
    if not filename:
        body = Align.center(
            Text("◇ idle — no file in pipeline", style=DIM),
            vertical="middle",
        )
        return Panel(
            body,
            title=Text("[ NOW TRANSCRIBING ]", style=f"{NEON_MAGENTA} bold"),
            title_align="left",
            box=box.HEAVY,
            border_style=GRID,
            padding=(1, 2),
            height=9,
        )

    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style=DIM, no_wrap=True, width=10)
    grid.add_column(ratio=1)

    duration_text = _fmt_mmss(duration_s) if duration_s else "—"
    file_line = Text()
    file_line.append("▶ ", style=NEON_CYAN)
    file_line.append(filename, style=f"{NEON_CYAN} bold")
    file_line.append(f"   {size_text} · {duration_text}", style=DIM)

    pulse = Text(_pulse(frame), style=NEON_MAGENTA)
    pulse_label = Text(current_stage or "READY", style=f"{NEON_CYAN} bold")
    pulse_line = Text.assemble(pulse, "  ", pulse_label)

    chips = _stage_chips(stages, current_stage, completed)

    # Real progress for stages WhisperX exposes; indeterminate pulse otherwise.
    has_real_progress = current_stage in ("ASR", "ALIGN", "DIARIZE")
    if has_real_progress:
        bar = _bar(stage_progress, width=40, filled_style=NEON_CYAN)
        pct_text = Text(f"{stage_progress * 100:5.1f}%", style=f"{NEON_CYAN} bold")
    else:
        # Cosmetic indeterminate bar — slides a chunk across.
        width = 40
        pos = frame % (width + 8)
        cells = []
        for i in range(width):
            d = (pos - i) % (width + 8)
            cells.append("█" if 0 <= d < 6 else "░")
        bar = Text("".join(cells), style=NEON_MAGENTA)
        pct_text = Text(" ····", style=DIM)
    elapsed_text = Text(_fmt_mmss(elapsed_s), style=f"{NEON_YELLOW} bold")

    grid.add_row("FILE",     file_line)
    grid.add_row("PULSE",    pulse_line)
    grid.add_row("STAGES",   chips)
    grid.add_row("PROGRESS", Text.assemble(bar, "  ", pct_text, "  ", elapsed_text))

    return Panel(
        grid,
        title=Text("[ NOW TRANSCRIBING ]", style=f"{NEON_MAGENTA} bold"),
        title_align="left",
        box=box.HEAVY,
        border_style=NEON_CYAN,
        padding=(1, 2),
    )


# ─── Queue table ────────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    x = float(n)
    for unit in ("B ", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:5.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def queue_table(rows) -> Panel:
    t = Table(
        box=box.SIMPLE_HEAVY,
        header_style=f"{NEON_MAGENTA} bold",
        expand=True,
        show_edge=False,
        padding=(0, 1),
    )
    t.add_column("#",      width=4, justify="right", style=DIM)
    t.add_column("FILE",   overflow="fold")
    t.add_column("SIZE",   width=10, justify="right", style=DIM)
    t.add_column("STATUS", width=14)
    t.add_column("TIME",   width=8, justify="right", style=DIM)

    if not rows:
        t.add_row("", Text("no audio files in incoming/", style=DIM), "", "", "")
    else:
        for i, r in enumerate(rows, 1):
            glyph, color = STATUS_GLYPHS.get(r.status, ("?", DIM))
            status_text = Text()
            status_text.append(f"{glyph} ", style=color)
            status_text.append(r.status.upper(), style=color)
            t.add_row(str(i), r.name, _fmt_size(r.size), status_text, r.elapsed_text)

    return Panel(
        t,
        title=Text("[ QUEUE ]", style=f"{NEON_MAGENTA} bold"),
        title_align="left",
        box=box.HEAVY,
        border_style=GRID,
        padding=(0, 1),
    )


# ─── Batch progress bar ─────────────────────────────────────────────────────

def batch_progress(done: int, failed: int, total: int) -> Text:
    if total <= 0:
        return Text("")
    pct = (done + failed) / total
    bar = _bar(pct, width=40)
    return Text.assemble(
        Text("BATCH  ", style=f"{NEON_MAGENTA} bold"),
        bar,
        Text(f"  {done + failed}/{total}", style=f"{NEON_CYAN} bold"),
        Text(f"   ✓{done}", style=NEON_GREEN),
        Text(f"  ✗{failed}", style=NEON_RED) if failed else Text(""),
    )
