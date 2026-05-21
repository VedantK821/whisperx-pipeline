"""Watchdog helpers shared by the app."""
from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler

from .transcribe import AUDIO_EXTS

STABILITY_SECONDS = 5.0
POLL_INTERVAL = 1.0
TMP_PREFIXES = (".syncthing.", "~syncthing~.")
TMP_SUFFIXES = (".tmp", ".part", ".partial")


def is_candidate(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() not in AUDIO_EXTS:
        return False
    name = p.name
    if name.startswith(TMP_PREFIXES) or name.endswith(TMP_SUFFIXES):
        return False
    return True


def wait_until_stable(p: Path) -> bool:
    """Block until `p`'s size stops changing for STABILITY_SECONDS.

    Returns False if the file disappears before stabilising.
    """
    last_size = -1
    stable_since = 0.0
    while True:
        try:
            size = p.stat().st_size
        except FileNotFoundError:
            return False
        now = time.monotonic()
        if size != last_size:
            last_size = size
            stable_since = now
        elif now - stable_since >= STABILITY_SECONDS:
            return True
        time.sleep(POLL_INTERVAL)


class IncomingHandler(FileSystemEventHandler):
    """Forwards filesystem events to a callback for any audio candidate."""

    def __init__(self, on_new) -> None:
        self.on_new = on_new

    def _maybe(self, path_str: str) -> None:
        p = Path(path_str)
        if is_candidate(p):
            self.on_new(p)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe(event.dest_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe(event.src_path)
