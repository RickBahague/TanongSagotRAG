"""P4 logging (rotating log in the user-data dir) and the ONNX reranker."""
import logging
import sys
import threading
from pathlib import Path

import pytest

import applog
import config
import rag


@pytest.fixture
def fresh_log(monkeypatch):
    """applog.setup() as if for the first time; hooks and streams restored."""
    monkeypatch.setattr(applog, "_handler", None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    root = logging.getLogger("anyrag")
    before = list(root.handlers)
    yield
    for h in root.handlers[:]:
        if h not in before:
            root.removeHandler(h)
            h.close()


def _flush():
    for h in logging.getLogger("anyrag").handlers:
        h.flush()


def test_log_lives_in_user_data_dir(fresh_log):
    f = applog.setup(capture_output=False)
    assert Path(f).resolve().parent == (config.USER_ROOT / "logs").resolve()
    logging.getLogger("anyrag.rag").warning("zebra-warning")
    _flush()
    assert "zebra-warning" in applog.tail()
    assert applog.setup() == f                 # idempotent


def test_prints_and_thread_crashes_captured(fresh_log):
    applog.setup(capture_output=True)
    print("[rag] okapi-printed")
    t = threading.Thread(target=lambda: 1 / 0, name="worker-x")
    t.start()
    t.join()
    _flush()
    text = applog.tail()
    assert "okapi-printed" in text
    assert "worker-x" in text and "ZeroDivisionError" in text


def test_log_rotates(fresh_log, monkeypatch):
    monkeypatch.setattr(applog, "MAX_BYTES", 2000)
    applog.setup(capture_output=False)
    lg = logging.getLogger("anyrag.test")
    for i in range(200):
        lg.info("line %d %s", i, "x" * 40)
    _flush()
    assert (applog.log_dir() / (applog.LOG_NAME + ".1")).exists()


def test_logs_routes(server, fresh_log, monkeypatch):
    applog.setup(capture_output=False)
    logging.getLogger("anyrag.serve").warning("quokka-in-log")
    _flush()
    st, d = server.json("GET", "/logs")
    assert st == 200 and "quokka-in-log" in d["tail"]
    assert d["folder"] == str(applog.log_dir())
    opened = []
    monkeypatch.setattr(applog, "open_folder",
                        lambda: opened.append(1) or applog.log_dir())
    st, _ = server.request("POST", "/logs/open")            # not JSON
    assert st == 415 and not opened
    st, d = server.json("POST", "/logs/open", {})
    assert st == 200 and opened


def test_unexpected_error_logged_and_500(server, fresh_log, monkeypatch):
    applog.setup(capture_output=False)
    import serve

    def boom(*a, **k):
        raise RuntimeError("wombat-crash")
    monkeypatch.setattr(serve, "global_settings", boom)
    st, d = server.json("GET", "/settings")
    _flush()
    assert st == 500 and "log" in d["error"]
    assert "wombat-crash" in applog.tail()


# ------------------------------------------------------------ reranker --

class FakeReranker:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return [len(p[1]) for p in self.pairs]     # longest text first


def test_rerank_sees_section_heading(monkeypatch):
    fake = FakeReranker()
    monkeypatch.setattr(rag, "_RERANKER", fake)
    hits = [(1.0, {"essay": "Deep Sea Mining", "text": "short"}),
            (0.5, {"essay": "", "text": "a longer body"})]
    q = "What is deep sea mining?"
    out = rag._maybe_rerank(q, hits, "auto")
    assert fake.pairs == [(q, "Deep Sea Mining. short"), (q, "a longer body")]
    assert [c["text"] for _, c in out] == ["short", "a longer body"]
    assert rag._maybe_rerank(q, hits, "off") is hits


def test_get_reranker_prefers_onnx(monkeypatch, tmp_path):
    import encoder
    made = []
    monkeypatch.setattr(config, "ONNX_RERANKER_DIR", tmp_path)
    monkeypatch.setattr(encoder, "OnnxReranker", lambda d: made.append(d) or "onnx")
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "tokenizer.json").write_text("{}")
    assert encoder.get_reranker() == "onnx" and made == [tmp_path]


@pytest.mark.skipif(not (Path(config.ONNX_RERANKER_DIR) / "model.onnx").exists(),
                    reason="reranker not fetched (scripts/fetch_reranker.py)")
def test_onnx_reranker_orders_relevance():
    import encoder
    rr = encoder.OnnxReranker(config.ONNX_RERANKER_DIR)
    q = "How do worker bees build the wax comb?"
    good, bad, long_ = rr.predict([
        (q, "Worker bees secrete wax from glands and shape it into "
            "hexagonal comb cells."),
        (q, "The lighthouse keeper trimmed the lamp wick each evening."),
        (q, "bees " * 2000)])                         # truncated, no error
    assert good > bad
    assert isinstance(long_, float)


def test_rerank_pool_limits_scored_hits(monkeypatch):
    fake = FakeReranker()
    monkeypatch.setattr(rag, "_RERANKER", fake)
    hits = [(9 - i, {"essay": "", "text": "t" * (i + 1), "id": i})
            for i in range(6)]
    out = rag._maybe_rerank("What is the longest text?", hits, "auto", pool=3)
    assert len(fake.pairs) == 3
    assert [c["id"] for _, c in out] == [2, 1, 0, 3, 4, 5]
    scores = [s for s, _ in out]
    assert scores == sorted(scores, reverse=True)


def test_crash_still_reaches_console_in_dev(fresh_log, capsys):
    applog.setup(capture_output=False)
    try:
        raise ValueError("platypus-crash")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    _flush()
    assert "platypus-crash" in applog.tail()
    assert "platypus-crash" in capsys.readouterr().err
