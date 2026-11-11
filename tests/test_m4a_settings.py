"""M4-A (P3-1): global + per-collection settings routes validated against
config.SETTINGS_SCHEMA, applied without a restart, and /backend/status."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import config
import library

QUESTION = "How do worker bees build the wax comb?"


@pytest.fixture(autouse=True)
def _reset_settings():
    """Global settings written by a test must not leak into the next one:
    remove the file and re-apply (restores the pre-override values)."""
    yield
    config.SETTINGS_FILE.unlink(missing_ok=True)
    config.reload_settings()


def _ask_sources(server, cid):
    st, evs = server.ndjson("POST", f"/c/{cid}/ask",
                            {"question": QUESTION, "mode": "scholar"})
    assert st == 200 and evs[0]["type"] == "meta", evs
    assert evs[-1]["type"] == "done", evs[-1]
    return evs[0]["sources"]


# ------------------------------------------------------------- global --

def test_get_settings_shape(server):
    st, s = server.json("GET", "/settings")
    assert st == 200
    # legacy fields kept for older UI code
    assert s["backend"] == config.LLM_BACKEND and s["model"] == "test-model"
    top = s["settings"]["TOP_K"]
    assert top["value"] == config.TOP_K and top["default"] == config.TOP_K
    assert top["overridden"] is False and top["type"] == "int"
    assert top["scope"] == "collection" and top["min"] == 1
    be = s["settings"]["LLM_BACKEND"]
    assert be["type"] == "enum" and be["scope"] == "global"
    assert set(be["choices"]) == {"ollama", "openai", "embedded"}
    for k in ("LLM_MODEL", "LLM_URL", "LLM_CTX", "LLM_MAX_TOKENS",
              "TEMPERATURE", "MIN_SCORE", "MIN_COVERAGE", "MIN_DENSE_SIM",
              "DENSE_SOLO_SIM", "MAX_PER_WORK", "POOL_FACTOR", "HIERARCHICAL",
              "RERANK", "QUOTES_TOP_K", "QUOTES_TEMPERATURE",
              "QUOTE_MATCH_RATIO", "CHUNK_WORDS", "CHUNK_OVERLAP_WORDS",
              "FOLLOWUP_TURNS", "HISTORY_ANSWER_WORDS",
              "REWRITE_MAX_TOKENS"):
        assert k in s["settings"], k
    assert s["overrides"] == {}


@pytest.mark.parametrize("body,key", [
    ({"NOT_A_SETTING": 1}, "NOT_A_SETTING"),
    ({"TOP_K": "5"}, "TOP_K"),
    ({"TOP_K": 0}, "TOP_K"),
    ({"TOP_K": 2.5}, "TOP_K"),
    ({"TOP_K": True}, "TOP_K"),
    ({"TEMPERATURE": 3}, "TEMPERATURE"),
    ({"LLM_BACKEND": "gpt"}, "LLM_BACKEND"),
    ({"LLM_URL": "ftp://x"}, "LLM_URL"),
    ({"LLM_MODEL": ""}, "LLM_MODEL"),
    ({"HIERARCHICAL": "yes"}, "HIERARCHICAL"),
    ({"CHUNK_OVERLAP_WORDS": 500}, "CHUNK_OVERLAP_WORDS"),   # >= 450 words
    ({"LLM_MAX_TOKENS": 9000}, "LLM_MAX_TOKENS"),           # >= LLM_CTX
])
def test_invalid_global_settings_rejected(server, body, key):
    before = config.TOP_K
    st, res = server.json("POST", "/settings", dict(body, MIN_SCORE=1.0))
    assert st == 400, res
    assert key in res["errors"] and res["errors"][key]
    assert key in res["error"]
    # nothing written, nothing applied (not even the valid MIN_SCORE)
    assert not config.SETTINGS_FILE.exists()
    assert config.TOP_K == before and config.MIN_SCORE == 5.0


def test_settings_require_json(server):
    st, _ = server.request("POST", "/settings", b'{"TOP_K": 3}',
                           headers={"Content-Type": "text/plain"})
    assert st == 415
    st, res = server.json("POST", "/settings", [1, 2])
    assert st == 400


def test_override_and_null_removes(server):
    st, s = server.json("POST", "/settings", {"TOP_K": 3, "TEMPERATURE": 1})
    assert st == 200, s
    assert s["settings"]["TOP_K"] == {**s["settings"]["TOP_K"],
                                      "value": 3, "default": 10,
                                      "overridden": True}
    assert s["settings"]["TEMPERATURE"]["value"] == 1.0
    assert json.loads(config.SETTINGS_FILE.read_text()) == \
        {"TOP_K": 3, "TEMPERATURE": 1.0}
    assert config.TOP_K == 3 and config.get("TOP_K") == 3

    st, s = server.json("POST", "/settings", {"TOP_K": None})
    assert st == 200
    assert s["settings"]["TOP_K"]["value"] == 10
    assert s["settings"]["TOP_K"]["overridden"] is False
    assert s["overrides"] == {"TEMPERATURE": 1.0}
    assert config.TOP_K == 10
    assert json.loads(config.SETTINGS_FILE.read_text()) == \
        {"TEMPERATURE": 1.0}


def test_top_k_applied_without_restart(server, fake_llm, make_collection):
    col = make_collection()
    assert len(_ask_sources(server, col.id)) > 3
    st, _ = server.json("POST", "/settings", {"TOP_K": 2})
    assert st == 200
    assert len(_ask_sources(server, col.id)) == 2
    # a per-collection override beats the global one, also live
    st, res = server.json("POST", f"/c/{col.id}/settings", {"TOP_K": 3})
    assert st == 200, res
    assert len(_ask_sources(server, col.id)) == 3
    st, _ = server.json("POST", f"/c/{col.id}/settings", {"TOP_K": None})
    assert len(_ask_sources(server, col.id)) == 2


def test_llm_url_applied_without_restart(server, make_collection):
    """The backend URL is read at call time: a saved LLM_URL is used by the
    very next question (conftest points the default at a dead port)."""
    from conftest import FakeLLM
    col = make_collection()
    st, evs = server.ndjson("POST", f"/c/{col.id}/ask",
                            {"question": QUESTION})
    assert evs[-1]["type"] == "error"              # dead port
    fake = FakeLLM()
    try:
        st, _ = server.json("POST", "/settings", {"LLM_URL": fake.url,
                                                  "LLM_MODEL": "other-model"})
        assert st == 200
        st, evs = server.ndjson("POST", f"/c/{col.id}/ask",
                                {"question": QUESTION})
        assert evs[-1]["type"] == "done", evs[-1]
        assert fake.last["model"] == "other-model"
    finally:
        fake.close()


def test_removed_override_restores_previous_value():
    config.SETTINGS_FILE.write_text(json.dumps({"POOL_FACTOR": 9}))
    config.reload_settings()
    assert config.POOL_FACTOR == 9 and config.default_value("POOL_FACTOR") == 6
    config.SETTINGS_FILE.unlink()
    config.reload_settings()
    assert config.POOL_FACTOR == 6


def test_invalid_entries_in_file_are_ignored():
    config.SETTINGS_FILE.write_text(json.dumps({"TOP_K": -4, "BOGUS": 1,
                                                "MAX_PER_WORK": 4}))
    config.reload_settings()
    assert config.TOP_K == 10 and config.MAX_PER_WORK == 4


# --------------------------------------------------------- collection --

def test_collection_settings_routes(server, make_collection):
    col = make_collection(ingest=False)          # works before indexing
    st, s = server.json("GET", f"/c/{col.id}/settings")
    assert st == 200 and s["id"] == col.id and s["overrides"] == {}
    assert "LLM_BACKEND" not in s["settings"]     # global-only keys hidden
    assert s["settings"]["CHUNK_WORDS"]["value"] == config.CHUNK_WORDS

    st, res = server.json("POST", f"/c/{col.id}/settings",
                          {"LLM_BACKEND": "ollama"})
    assert st == 400 and "global only" in res["errors"]["LLM_BACKEND"]
    st, res = server.json("POST", f"/c/{col.id}/settings",
                          {"CHUNK_WORDS": 100, "CHUNK_OVERLAP_WORDS": 100})
    assert st == 400 and "CHUNK_OVERLAP_WORDS" in res["errors"]
    assert not col.settings_file.exists()

    st, s = server.json("POST", f"/c/{col.id}/settings",
                        {"CHUNK_WORDS": 200, "MIN_COVERAGE": 0.25})
    assert st == 200, s
    assert s["settings"]["CHUNK_WORDS"]["value"] == 200
    assert s["settings"]["CHUNK_WORDS"]["overridden"] is True
    assert s["settings"]["CHUNK_WORDS"]["default"] == config.CHUNK_WORDS
    assert library.get(col.id).setting("CHUNK_WORDS") == 200
    st, s = server.json("POST", f"/c/{col.id}/settings",
                        {"CHUNK_WORDS": None})
    assert s["overrides"] == {"MIN_COVERAGE": 0.25}
    assert library.get(col.id).setting("CHUNK_WORDS") == config.CHUNK_WORDS

    st, _ = server.json("GET", "/c/nope/settings")
    assert st == 404
    st, _ = server.json("POST", "/c/nope/settings", {"TOP_K": 3})
    assert st == 404


def test_global_change_seen_through_collection(server, make_collection):
    col = make_collection(ingest=False)
    server.json("POST", "/settings", {"MIN_COVERAGE": 0.3})
    st, s = server.json("GET", f"/c/{col.id}/settings")
    assert s["settings"]["MIN_COVERAGE"]["value"] == 0.3
    assert s["settings"]["MIN_COVERAGE"]["overridden"] is False
    assert library.get(col.id).setting("MIN_COVERAGE") == 0.3


# ------------------------------------------------------ backend status --

class _OllamaStub:
    def __init__(self, models):
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"models": [{"name": m} for m in models]})
                self.send_response(200 if self.path == "/api/tags" else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *a):
                pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever,
                         kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_backend_status_fake_llm_up(server, fake_llm):
    st, s = server.json("GET", "/backend/status")
    assert st == 200
    assert s["backend"] == "openai" and s["url"] == fake_llm.url
    assert s["reachable"] is True          # the stub answers (404 on GET)


def test_backend_status_dead_port(server):
    t0 = time.monotonic()
    st, s = server.json("GET", "/backend/status")
    assert st == 200 and time.monotonic() - t0 < 5
    assert s["reachable"] is False and "cannot reach" in s["error"]
    assert s["models"] is None


def test_backend_status_ollama_models(server, monkeypatch):
    stub = _OllamaStub(["llama3.1:8b", "qwen2.5:1.5b"])
    try:
        monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
        monkeypatch.setattr(config, "LLM_URL", stub.url)
        monkeypatch.setattr(config, "LLM_MODEL", "qwen2.5:1.5b")
        st, s = server.json("GET", "/backend/status")
        assert s["reachable"] and s["model_present"] is True
        assert s["models"] == ["llama3.1:8b", "qwen2.5:1.5b"]
        # "Test connection" with unsaved values: query overrides
        st, s = server.json("GET", "/backend/status?model=phi3:mini")
        assert s["reachable"] and s["model_present"] is False
        assert "ollama pull phi3:mini" in s["error"]
        st, s = server.json("GET", "/backend/status?model=llama3.1")
        assert s["model_present"] is False
        st, s = server.json("GET", "/backend/status?backend=ollama&url="
                            "http://127.0.0.1:9")
        assert s["reachable"] is False
    finally:
        stub.close()


def test_backend_status_embedded(server, monkeypatch, tmp_path):
    gguf = tmp_path / "m.gguf"
    monkeypatch.setattr(config, "LLM_BACKEND", "embedded")
    monkeypatch.setattr(config, "LLM_MODEL_PATH", gguf)
    st, s = server.json("GET", "/backend/status")
    assert s["backend"] == "embedded" and s["file_exists"] is False
    assert s["reachable"] is False and "missing" in s["error"]
    gguf.write_bytes(b"GGUF")
    st, s = server.json("GET", "/backend/status")
    assert s["file_exists"] is True and s["model_present"] is True
    assert s["reachable"] is s["llama_cpp"]
