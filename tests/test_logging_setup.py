"""The persistent file log must actually capture pipeline records.

Without this, the "re-run the file and read the log" diagnosis loop reads an
empty file and wastes a cycle. The risk is real: the Rich Live TUI sets
redirect_stderr=True, so the console StreamHandler loses everything during a
batch run, and the noisy-library silencing raises some loggers to ERROR. The
file handler must sit below all of that and still record our own INFO/DEBUG
lifecycle lines (e.g. "Loading alignment model (xx)" — the marker that tells us
the run hung inside align).
"""
import logging

import src.app as app


def _cleanup_root_handlers():
    root = logging.getLogger()
    for h in list(root.handlers):
        h.close()
    root.handlers.clear()


def test_file_log_captures_info_when_console_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "LOG_DIR", tmp_path)
    # Don't touch the global faulthandler / pytest's faulthandler plugin.
    monkeypatch.setattr(app.faulthandler, "enable", lambda *a, **k: None)

    try:
        log_file = app._setup_logging(verbose=False)
        assert log_file == tmp_path / "whisperx.log"

        # Mirror the real signal: an INFO line from the transcribe logger, which
        # is NOT among the silenced libraries, emitted while the console is WARNING.
        logging.getLogger("src.transcribe").info("Loading alignment model (%s)", "hi")
        for h in logging.getLogger().handlers:
            h.flush()

        assert "Loading alignment model (hi)" in log_file.read_text(encoding="utf-8")
    finally:
        _cleanup_root_handlers()
