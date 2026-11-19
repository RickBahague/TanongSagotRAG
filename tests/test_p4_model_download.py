"""P4: first-run GGUF download (modeldl.py) and the embedded GPU setting."""
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import config
import modeldl
import rag

BLOB = bytes(range(256)) * 4096                        # 1 MB


class _Files(BaseHTTPRequestHandler):
    blob = BLOB
    honor_range = True
    delay = 0.0

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng and self.honor_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = self.blob[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for i in range(0, len(body), 65536):
            if self.delay:
                time.sleep(self.delay)
            try:
                self.wfile.write(body[i:i + 65536])
            except (BrokenPipeError, ConnectionResetError):
                return

    def log_message(self, *a):
        pass


@pytest.fixture
def files():
    _Files.honor_range, _Files.delay = True, 0.0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/model.gguf"
    httpd.shutdown()
    httpd.server_close()
    modeldl.reset()


def _wait(states=("done", "error", "cancelled"), t=20):
    end = time.time() + t
    while time.time() < end:
        s = modeldl.status()
        if s["state"] in states:
            return s
        time.sleep(0.02)
    raise AssertionError(f"timed out: {modeldl.status()}")


SHA = hashlib.sha256(BLOB).hexdigest()


def test_download_verifies_and_installs(files, tmp_path):
    dest = tmp_path / "m.gguf"
    modeldl.start(files, dest, len(BLOB), SHA)
    s = _wait()
    assert s["state"] == "done" and dest.read_bytes() == BLOB
    assert not dest.with_name("m.gguf.part").exists()


def test_bad_checksum_rejected(files, tmp_path):
    dest = tmp_path / "m.gguf"
    modeldl.start(files, dest, len(BLOB), "0" * 64)
    s = _wait()
    assert s["state"] == "error" and "SHA-256" in s["error"]
    assert not dest.exists() and not dest.with_name("m.gguf.part").exists()


def test_resume_from_partial_file(files, tmp_path):
    dest = tmp_path / "m.gguf"
    dest.with_name("m.gguf.part").write_bytes(BLOB[:300_000])
    modeldl.start(files, dest, len(BLOB), SHA)
    assert _wait()["state"] == "done" and dest.read_bytes() == BLOB


def test_server_ignoring_range_restarts(files, tmp_path):
    _Files.honor_range = False
    dest = tmp_path / "m.gguf"
    dest.with_name("m.gguf.part").write_bytes(b"junk" * 1000)
    modeldl.start(files, dest, len(BLOB), SHA)
    assert _wait()["state"] == "done" and dest.read_bytes() == BLOB


def test_cancel_keeps_part_for_resume(files, tmp_path):
    _Files.delay = 0.02
    dest = tmp_path / "m.gguf"
    modeldl.start(files, dest, len(BLOB), SHA)
    _wait(("downloading",))
    time.sleep(0.1)
    modeldl.cancel()
    s = _wait(("cancelled",))
    part = dest.with_name("m.gguf.part")
    assert s["state"] == "cancelled" and 0 < part.stat().st_size < len(BLOB)
    _Files.delay = 0.0
    modeldl.start(files, dest, len(BLOB), SHA)
    assert _wait()["state"] == "done" and dest.read_bytes() == BLOB


def test_default_path_prefers_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path / "bundle")
    assert modeldl.default_model_path() == modeldl.user_path()
    assert modeldl.is_download_target(modeldl.user_path())
    b = tmp_path / "bundle" / "models" / modeldl.MODEL_NAME
    b.parent.mkdir(parents=True)
    b.write_bytes(b"x")
    assert modeldl.default_model_path() == b
    assert not modeldl.is_download_target(b)


def test_status_and_routes(server, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_BACKEND", "embedded")
    monkeypatch.setattr(config, "LLM_MODEL_PATH", modeldl.user_path())
    monkeypatch.setattr(rag, "llama_cpp_error", lambda: None)
    st, d = server.json("GET", "/backend/status")
    assert st == 200 and d["downloadable"] is True
    assert "not downloaded" in d["error"]
    started = []
    monkeypatch.setattr(modeldl, "start", lambda: started.append(1) or {"state": "starting"})
    st, _ = server.request("POST", "/model/download")          # not JSON
    assert st == 415 and not started
    st, d = server.json("POST", "/model/download", {})
    assert st == 200 and started and d["state"] == "starting"
    st, d = server.json("GET", "/model/download")
    assert st == 200 and "state" in d


def test_missing_download_target_message(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL_PATH", modeldl.user_path())
    monkeypatch.setattr(rag, "llama_cpp_error", lambda: None)
    monkeypatch.setattr(rag, "_EMBEDDED_LLM", None)
    monkeypatch.setattr(rag, "_EMBEDDED_KEY", None)
    import sys
    import types
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=None))
    with pytest.raises(RuntimeError, match="Download model"):
        rag._embedded_llm()


def test_gpu_layers_passed_and_cpu_fallback(monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    calls = []

    class FakeLlama:
        def __init__(self, **kw):
            calls.append(kw)
            if kw.get("n_gpu_layers"):
                raise RuntimeError("no GPU memory")
    import sys
    import types
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr(rag, "llama_cpp_error", lambda: None)
    monkeypatch.setattr(rag, "_EMBEDDED_LLM", None)
    monkeypatch.setattr(rag, "_EMBEDDED_KEY", None)
    monkeypatch.setattr(config, "LLM_MODEL_PATH", gguf)
    monkeypatch.setattr(config, "LLM_GPU_LAYERS", -1)
    assert isinstance(rag._embedded_llm(), FakeLlama)
    assert calls[0]["n_gpu_layers"] == -1 and "n_gpu_layers" not in calls[1]
    assert "LLM_GPU_LAYERS" in config.SETTINGS_SCHEMA
