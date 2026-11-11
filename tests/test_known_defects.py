"""Known defects from plan-enhancements.md, pinned as strict xfail tests.

Each test asserts the CORRECT behaviour and fails on today's code. With
xfail_strict (pytest.ini), a fix makes the test XPASS → suite error, which
is the signal to delete that `@xfail` marker. Milestone ownership is in
implem-plan-enhancements.md; `scripts/verify.sh M<n>` refuses to pass while
any test it lists still carries a marker here.
"""
import json
import os
import re
import threading
import time
from pathlib import Path

import pytest

import config
import library
import rag
from helpers import CORPUS, REPO, reingest


def xfail(item, why):
    return pytest.mark.xfail(reason=f"{item}: {why}", strict=True)


def _sources(col):
    return {c["source"] for c in rag.load_chunks(library.get(col.id))}


# ------------------------------------------------------------- M1-A ingest

def test_prune_removed_source(make_collection):
    col = make_collection(files=("volcanoes.txt", "beekeeping.md"))
    (col.sources_dir / "volcanoes.txt").unlink()
    reingest(col)
    assert _sources(col) == {"beekeeping.md"}
    assert not list(col.data_dir.rglob("volcanoes*"))
    from ingest import load_manifest
    assert "volcanoes.txt" not in load_manifest(col)


def test_same_stem_sources_both_indexed(make_collection):
    col = make_collection(files=("report.md", "report.txt"))
    assert _sources(col) == {"report.md", "report.txt"}
    text = " ".join(c["text"] for c in rag.load_chunks(col))
    assert "sourdough" in text and "derailleur" in text


def test_short_document_is_chunked(make_collection):
    col = make_collection(files=("tiny.txt",))
    chunks = rag.load_chunks(col)
    assert len(chunks) == 1 and "ferns" in chunks[0]["text"].lower()


def test_chunk_tail_not_dropped(make_collection):
    col = make_collection(files=("volcanoes.txt",),
                          settings={"CHUNK_WORDS": 40,
                                    "CHUNK_OVERLAP_WORDS": 10})
    last_words = (CORPUS / "volcanoes.txt").read_text().split()[-5:]
    text = " ".join(c["text"] for c in rag.load_chunks(col))
    assert " ".join(last_words) in text


def test_duplicate_section_titles_kept(make_collection):
    col = make_collection(files=("astronomy.pdf",))
    chunks = rag.load_chunks(col)
    tele = {c["essay"] for c in chunks if c["essay"].startswith("Telescopes")}
    assert len(tele) == 2
    assert not any(c["page_start"] == 1 and c["page_end"] == 3 for c in chunks)


def test_rechunk_on_content_change_not_mtime(make_collection):
    col = make_collection(files=("volcanoes.txt",))
    src = col.sources_dir / "volcanoes.txt"

    # 1. touch without change → no re-chunk
    os.utime(src, (time.time() + 60, time.time() + 60))
    ev = [e for e in reingest(col) if e["stage"] == "chunk"]
    assert ev[0]["status"] == "up to date"

    # 2. content change with an OLD mtime → must re-chunk
    src.write_text("Lava lakes " * 60)
    os.utime(src, (1_000_000_000, 1_000_000_000))
    reingest(col)
    assert "lava lakes" in rag.load_chunks(library.get(col.id))[0]["text"].lower()


def test_json_writes_atomic(make_collection, monkeypatch):
    replaced = []
    real = os.replace

    def spy(a, b, *args, **kw):
        replaced.append(Path(b).name)
        return real(a, b, *args, **kw)
    monkeypatch.setattr(os, "replace", spy)
    make_collection(files=("volcanoes.txt",))
    assert {"collections.json", "collection.json", "manifest.json"} \
        <= set(replaced)


# ------------------------------------------------------------- M1-B server

@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_unindexed_collection_returns_409(server):
    _, col = server.json("POST", "/collections", {"name": "Empty"})
    st, body = server.json("GET", f"/c/{col['id']}/stats")
    assert st == 409 and "error" in body


def test_runtime_cache_is_lru(make_collection):
    import serve
    ids = [make_collection(n, files=("volcanoes.txt",)).id
           for n in ("a", "b", "c", "d")]
    for cid in ids[:3]:
        serve.runtime(cid)
    serve.runtime(ids[0])                 # touch a → b is now oldest
    serve.runtime(ids[3])
    assert ids[0] in serve._RUNTIMES and ids[1] not in serve._RUNTIMES


def test_foreign_host_rejected(server):
    st, _ = server.request("GET", "/collections",
                           headers={"Host": "evil.example"})
    assert st == 403


def test_foreign_origin_rejected(server):
    st, _ = server.json("POST", "/collections", {"name": "Pwned"},
                        headers={"Origin": "http://evil.example"})
    assert st == 403
    assert library.list_all() == []


def test_post_requires_json_content_type(server):
    st, _ = server.request("POST", "/collections",
                           json.dumps({"name": "Simple"}).encode(),
                           headers={"Content-Type": "text/plain"})
    assert st in (400, 415)
    assert library.list_all() == []


def test_oversized_body_rejected(server):
    """Contract: JSON bodies above 10 MB are refused with 413 before the
    body is read (the test only sends 1 KB of a declared 64 MB)."""
    import http.client
    server.httpd.handle_error = lambda *a: None   # expected broken pipe
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
    conn.putrequest("POST", "/collections")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(64 * 1024 * 1024))
    conn.endheaders()
    conn.send(b" " * 1024)
    try:
        assert conn.getresponse().status == 413
    finally:
        conn.close()


@pytest.mark.parametrize("path,pattern", [
    ("rag.py", r"martial(\\s\+| )law"),
    ("conversations.py", r"author himself"),
    ("scripts/export_onnx.py", r"land reform|national democracy"),
])
def test_no_askj_leftovers(path, pattern):
    assert not re.search(pattern, (REPO / path).read_text(), re.I)


# ----------------------------------------------------------- M1-C generation

def _all_chunks_index(col):
    """Index stub returning every chunk, so the context is as large as k."""
    chunks = rag.load_chunks(col)

    class Everything:
        def __init__(self):
            self.chunks = chunks

        def search(self, q, k=10):
            return [(1.0, c) for c in self.chunks][:k]
    return Everything()


def test_context_fits_budget(make_collection, fake_llm, monkeypatch):
    col = make_collection()
    monkeypatch.setattr(config, "LLM_CTX", 900)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 200)
    monkeypatch.setattr(rag, "gate_ok", lambda *a: True)
    res = rag.ask_stream(col, "Tell me everything", index=_all_chunks_index(col),
                         k=10)
    "".join(res["stream"])
    est = fake_llm.prompt_words() * 1.35          # estimator from the plan
    assert est + config.LLM_MAX_TOKENS <= config.LLM_CTX
    assert res.get("dropped", 0) > 0


def test_max_tokens_sent_openai(make_collection, fake_llm):
    col = make_collection(files=("beekeeping.md",))
    res = rag.ask_stream(col, "How do worker bees build the wax comb?")
    "".join(res["stream"])
    assert fake_llm.last.get("max_tokens") == config.LLM_MAX_TOKENS


def test_embedded_generation_serialized(monkeypatch):
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    class FakeLlama:
        def create_chat_completion(self, **kw):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            try:
                for t in ("a", "b", "c"):
                    time.sleep(0.05)
                    yield {"choices": [{"delta": {"content": t}}]}
            finally:
                with lock:
                    state["active"] -= 1

    monkeypatch.setattr(config, "LLM_BACKEND", "embedded")
    monkeypatch.setattr(rag, "_EMBEDDED_LLM", FakeLlama())
    out = []
    threads = [threading.Thread(target=lambda: out.append(
        "".join(rag.llm_stream("s", "u")))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == ["abc", "abc"]
    assert state["max"] == 1


# ----------------------------------------------------------- M2-A encoder

@xfail("P0-1", "embed stage only knows sentence-transformers")
def test_embed_stage_uses_onnx_encoder(make_collection):
    if not (config.ONNX_ENCODER_DIR / "model.onnx").exists():
        pytest.skip("ONNX encoder not present (scripts/export_onnx.py)")
    pytest.importorskip("onnxruntime")
    col = make_collection(files=("volcanoes.txt",))
    vec = col.vectors_dir / "volcanoes.txt.npz"
    assert vec.exists()
    import numpy as np
    assert str(np.load(vec)["model"]) == config.EMBED_MODEL
