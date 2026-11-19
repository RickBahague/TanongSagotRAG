"""Rotating log file in the user-data dir (P4 "Logging").

The packaged app has no console, so print() output, warnings and
tracebacks from server threads were lost. setup() sends them to
<user-data>/logs/anyrag.log (1 MB, 3 backups):

  - the "anyrag" logger tree (encoder, rag, serve, …) at INFO,
  - print() / stderr when there is no usable console (frozen app, or
    stdout is None), line by line,
  - uncaught exceptions, in the main thread and in worker threads.

In dev (a console exists) output still goes to the terminal as well.
"""
import logging
import logging.handlers
import sys
import threading

import paths

LOG_NAME = "anyrag.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
_handler = None


def log_dir():
    d = paths.user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_file():
    return log_dir() / LOG_NAME


class _StreamToLog:
    """File-like object that writes complete lines to a logger and,
    when given one, to the original stream as well."""

    def __init__(self, logger, level, echo=None):
        self.logger, self.level, self.echo = logger, level, echo
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s):
        if self.echo is not None:
            try:
                self.echo.write(s)
            except Exception:
                pass
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        for line in lines:
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(s)

    def flush(self):
        if self.echo is not None:
            try:
                self.echo.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup(capture_output=None):
    """Install the file handler once. capture_output=None captures print()
    and stderr only when there is no console (sys.stdout is None or the
    app is frozen); True/False force it."""
    global _handler
    if _handler is not None:
        return log_file()
    _handler = logging.handlers.RotatingFileHandler(
        log_file(), maxBytes=MAX_BYTES, backupCount=BACKUPS,
        encoding="utf-8")
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
    root = logging.getLogger("anyrag")
    root.setLevel(logging.INFO)
    root.addHandler(_handler)

    if capture_output is None:
        capture_output = paths.FROZEN or sys.stdout is None
    if capture_output:
        out = logging.getLogger("anyrag.stdout")
        err = logging.getLogger("anyrag.stderr")
        sys.stdout = _StreamToLog(out, logging.INFO, sys.stdout)
        sys.stderr = _StreamToLog(err, logging.ERROR, sys.stderr)

    crash = logging.getLogger("anyrag.crash")
    # chain to the previous hooks: the traceback still reaches a console
    # (dev), and is also in the log (packaged app)
    prev_hook, prev_thread_hook = sys.excepthook, threading.excepthook

    def excepthook(tp, val, tb):
        crash.error("uncaught exception", exc_info=(tp, val, tb))
        if not capture_output:
            prev_hook(tp, val, tb)

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        crash.error("uncaught exception in thread %s",
                    getattr(args.thread, "name", "?"),
                    exc_info=(args.exc_type, args.exc_value,
                              args.exc_traceback))
        if not capture_output:
            prev_thread_hook(args)

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    logging.getLogger("anyrag").info("AnyRAG started; log at %s", log_file())
    return log_file()


def tail(n=200):
    """Last n lines of the current log ('' when there is none)."""
    f = log_file()
    if not f.exists():
        return ""
    with f.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(size - 256_000, 0))
        data = fh.read().decode("utf-8", "replace")
    return "\n".join(data.splitlines()[-n:])


def open_folder():
    """Open the log folder in the system file manager. Returns the path."""
    import os
    import subprocess
    d = log_dir()
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(d)])
    elif os.name == "nt":
        os.startfile(str(d))                     # noqa: S606 (local folder)
    else:
        subprocess.Popen(["xdg-open", str(d)])
    return d
