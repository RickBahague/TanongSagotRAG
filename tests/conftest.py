"""Shared test harness.

IMPORTANT — import-time paths: config.USER_ROOT, library.REGISTRY and
library.COLLECTIONS_DIR are computed when those modules are first imported.
So ANYRAG_HOME is set HERE, at conftest import, before any project module
is imported; `_guard_user_root` then asserts every path really points into
the temp dir. Never import project modules at the top of this file above
the env assignment.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="anyrag-test-"))
os.environ["ANYRAG_HOME"] = str(_HOME)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import socket  # noqa: E402

import pytest  # noqa: E402

# HTTPServer.server_bind() calls socket.getfqdn(), which can stall for many
# seconds on macOS (reverse DNS). Test servers only bind 127.0.0.1.
socket.getfqdn = lambda name="": name or "localhost"

import config  # noqa: E402
import library  # noqa: E402

from helpers import (CLEAN_FILES, CORPUS, CORPUS_FILES, FIXTURES,  # noqa: E402
                     UNREACHABLE_LLM)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_HOME, ignore_errors=True)


def _ensure_corpus():
    if not all((CORPUS / f).exists() for f in CORPUS_FILES):
        sys.path.insert(0, str(FIXTURES))
        import build
        build.build(CORPUS)


# --------------------------------------------------------------- isolation --

@pytest.fixture(autouse=True, scope="session")
def _guard_user_root():
    home = _HOME.resolve()
    for p in (config.USER_ROOT, library.REGISTRY, library.COLLECTIONS_DIR):
        assert home in Path(p).resolve().parents or Path(p).resolve() == home, \
            f"test data path escaped the temp home: {p}"
    _ensure_corpus()
    yield


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Fresh registry per test, caches cleared, LLM pointed at a dead port
    so nothing ever reaches a real Ollama/LM Studio on this machine."""
    monkeypatch.setattr(config, "LLM_BACKEND", "openai")
    monkeypatch.setattr(config, "LLM_URL", UNREACHABLE_LLM)
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")
    yield
    shutil.rmtree(library.COLLECTIONS_DIR, ignore_errors=True)
    for f in (library.REGISTRY, config.USER_ROOT / "settings.json"):
        Path(f).unlink(missing_ok=True)
    import rag
    rag._PARA_MAPS.clear()
    if "serve" in sys.modules:
        sys.modules["serve"]._RUNTIMES.clear()


# ------------------------------------------------------------- collections --

@pytest.fixture
def corpus_dir(tmp_path):
    """A private, mutable copy of the fixture corpus (tests may delete or
    edit files in it — it plays the user's linked folder)."""
    d = tmp_path / "docs"
    d.mkdir()
    for f in CORPUS_FILES:
        shutil.copy2(CORPUS / f, d / f)
    return d


@pytest.fixture
def make_collection(tmp_path):
    """make_collection(name, files=CLEAN_FILES, settings=None, ingest=True,
    linked=True) → Collection. Files are copied from the fixture corpus into
    a fresh folder (linked) or the collection's sources/ (managed).
    `col.ingest_events` holds the progress events of the ingest run."""
    counter = {"n": 0}

    def make(name="Fixtures", files=CLEAN_FILES, settings=None, ingest=True,
             linked=True, **create_kw):
        import ingest as ingest_mod
        counter["n"] += 1
        if linked:
            folder = tmp_path / f"linked-{counter['n']}"
            folder.mkdir()
            for f in files:
                shutil.copy2(CORPUS / f, folder / f)
            col = library.create(name, link=str(folder), **create_kw)
        else:
            col = library.create(name, **create_kw)
            for f in files:
                shutil.copy2(CORPUS / f, col.sources_dir / f)
        if settings:
            col.settings_file.write_text(json.dumps(settings))
            col = library.get(col.id)
        col.ingest_events = []
        if ingest:
            ingest_mod.run_pipeline(col, summaries=False, workers=1,
                                    progress=col.ingest_events.append)
        return col
    return make


# ---------------------------------------------------------------- fake LLM --

class FakeLLM:
    """Stdlib OpenAI-compatible chat server (streaming SSE). `reply` is the
    canned answer; `requests` records every parsed request body."""

    def __init__(self):
        self.reply = ("The documents describe this in detail "
                      "[Hive Structure, p. 1].")
        self.requests = []
        self.delay = 0.0
        fake = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                words = fake.reply.split(" ")
                for i, w in enumerate(words):
                    piece = w if i == 0 else " " + w
                    ev = {"choices": [{"delta": {"content": piece}}]}
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                    if fake.delay:
                        time.sleep(fake.delay)
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever,
                         kwargs={"poll_interval": 0.05}, daemon=True).start()

    @property
    def last(self):
        return self.requests[-1]

    def prompt_text(self, body=None):
        body = body or self.last
        return "\n".join(m["content"] for m in body["messages"])

    def prompt_words(self, body=None):
        return len(self.prompt_text(body).split())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(config, "LLM_BACKEND", "openai")
    monkeypatch.setattr(config, "LLM_URL", fake.url)
    yield fake
    fake.close()


# ------------------------------------------------------------------ server --

class Client:
    """Tiny HTTP client for the app server (stdlib http.client)."""

    def __init__(self, port):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"

    def request(self, method, path, body=None, headers=None, timeout=30):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=timeout)
        hdrs = dict(headers or {})
        data = body
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        return resp.status, raw

    def json(self, method, path, body=None, **kw):
        status, raw = self.request(method, path, body, **kw)
        return status, json.loads(raw) if raw else None

    def ndjson(self, method, path, body=None, **kw):
        status, raw = self.request(method, path, body, **kw)
        return status, [json.loads(l) for l in raw.decode().splitlines()
                        if l.strip()]


@pytest.fixture
def server():
    import serve
    httpd = serve.make_server(0)
    threading.Thread(target=httpd.serve_forever,
                     kwargs={"poll_interval": 0.05}, daemon=True).start()
    client = Client(httpd.server_address[1])
    client.httpd = httpd
    yield client
    httpd.shutdown()
    httpd.server_close()
