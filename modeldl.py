"""First-run download of the embedded GGUF (P4 "Bundle size").

The desktop bundle no longer ships the 1.1 GB model. On first use the app
downloads it into the user-data dir (<data>/models/), with progress,
resume after an interruption (HTTP Range on the .part file), cancel, and a
SHA-256 check before the file is put in place. A model found in the bundle
(dev checkout, or a build made with ANYRAG_BUNDLE_MODEL=1) is used as is.

One job per process; state is polled by GET /model/download.
"""
import hashlib
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

log = logging.getLogger("anyrag.modeldl")

MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_URL = ("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
             "resolve/main/" + MODEL_NAME)
MODEL_BYTES = 1_117_320_736
MODEL_SHA256 = ("6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434"
                "e9407e")
CHUNK = 1 << 20                # hashing block
READ_SIZE = 1 << 16            # network read
TIMEOUT = 30

_lock = threading.Lock()
_job = None


def bundled_path():
    return Path(config.ROOT) / "models" / MODEL_NAME


def user_path():
    return Path(config.USER_ROOT) / "models" / MODEL_NAME


def default_model_path():
    """The bundled GGUF when present, else where the download puts it."""
    b = bundled_path()
    return b if b.is_file() else user_path()


def is_download_target(path):
    """True when `path` is the file this module downloads (so the UI may
    offer a download for it; a user-chosen GGUF elsewhere is never
    fetched)."""
    try:
        return Path(path).resolve() == user_path().resolve()
    except OSError:
        return False


class Job:
    def __init__(self, url, dest, size, sha256):
        self.url, self.dest, self.size, self.sha256 = url, Path(dest), size, sha256
        self.part = self.dest.with_name(self.dest.name + ".part")
        self.state = "starting"       # downloading|verifying|done|error|cancelled
        self.done = self.part.stat().st_size if self.part.exists() else 0
        self.total = size
        self.error = None
        self._got = 0
        self.started = time.time()
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, name="model-download",
                                       daemon=True)

    def status(self):
        el = max(time.time() - self.started, 1e-6)
        return {"state": self.state, "done": self.done, "total": self.total,
                "error": self.error, "path": str(self.dest),
                "rate": self._rate(el)}

    def _rate(self, elapsed):
        return self._got / elapsed

    def _run(self):
        try:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            self._download()
            if self.cancel.is_set():
                self.state = "cancelled"
                return
            self.state = "verifying"
            if self.sha256 and _sha256(self.part) != self.sha256:
                self.part.unlink(missing_ok=True)
                raise RuntimeError("downloaded file failed the SHA-256 check "
                                   "(deleted; try again)")
            self.part.replace(self.dest)
            self.state = "done"
            log.info("model downloaded to %s", self.dest)
        except Exception as e:
            self.state, self.error = "error", str(e)
            log.warning("model download failed: %s", e)

    def _download(self):
        self.state = "downloading"
        have = self.part.stat().st_size if self.part.exists() else 0
        if self.size and have > self.size:
            self.part.unlink()
            have = 0
        if self.size and have == self.size:
            self.done = have
            return
        req = urllib.request.Request(self.url, headers={
            "User-Agent": "AnyRAG", **({"Range": f"bytes={have}-"} if have else {})})
        opener = urllib.request.build_opener()
        with opener.open(req, timeout=TIMEOUT) as r:
            if have and r.status != 206:      # server ignored Range: restart
                have = 0
            length = r.headers.get("Content-Length")
            if length and not self.size:
                self.total = have + int(length)
            self.done, self._got = have, 0
            with self.part.open("ab" if have else "wb") as f:
                while not self.cancel.is_set():
                    buf = r.read(READ_SIZE)    # small reads: cancel is prompt
                    if not buf:
                        break
                    f.write(buf)
                    self.done += len(buf)
                    self._got += len(buf)
        if not self.cancel.is_set() and self.total and self.done != self.total:
            raise RuntimeError(f"download ended early ({self.done:,} of "
                               f"{self.total:,} bytes); start it again to "
                               "resume")


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()


def status():
    """Current job status, or the file's state when no job ran."""
    with _lock:
        if _job is not None:
            return _job.status()
    dest = user_path()
    part = dest.with_name(dest.name + ".part")
    return {"state": "done" if dest.is_file() else "idle",
            "done": part.stat().st_size if part.exists() else 0,
            "total": MODEL_BYTES, "error": None, "path": str(dest),
            "rate": 0.0}


def start(url=None, dest=None, size=None, sha256=None):
    """Start (or return the running) download job."""
    global _job
    with _lock:
        if _job is not None and _job.state in ("starting", "downloading",
                                               "verifying"):
            return _job.status()
        dest = Path(dest or user_path())
        if dest.is_file():
            return {**status(), "state": "done"}
        _job = Job(url or MODEL_URL, dest,
                   MODEL_BYTES if size is None else size,
                   MODEL_SHA256 if sha256 is None else sha256)
        _job.thread.start()
        return _job.status()


def cancel():
    with _lock:
        job = _job
    if job is not None:
        job.cancel.set()
    return status()


def reset():
    """Forget the job (tests)."""
    global _job
    with _lock:
        _job = None
