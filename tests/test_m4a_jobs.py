"""M4-A (P3-4): ingest as a background job per collection — status, per
stage done/total, cooperative cancel, and a second client re-attaching to
the running job's NDJSON stream."""
import http.client
import json
import threading
import time

import pytest

import ingest
import library
from helpers import CLEAN_FILES


@pytest.fixture(autouse=True)
def _jobs_settle():
    """No ingest thread may outlive its test (conftest wipes the data dir)."""
    yield
    import serve
    for job in list(serve._JOBS.values()):
        job.cancel_event.set()
        job.thread.join(timeout=30)
    serve._JOBS.clear()


@pytest.fixture
def gate(monkeypatch):
    """Blocks ingest.chunk_source on its FIRST call until released, so a job
    can be observed (and cancelled / attached to) while it runs."""
    real = ingest.chunk_source
    g = {"entered": threading.Event(), "release": threading.Event(),
         "calls": []}

    def blocking(col, path, *a, **kw):
        g["calls"].append(path.name)
        if len(g["calls"]) == 1:
            g["entered"].set()
            assert g["release"].wait(30)
        return real(col, path, *a, **kw)

    monkeypatch.setattr(ingest, "chunk_source", blocking)
    yield g
    g["release"].set()


def _stream(port, path):
    """POST an ingest stream; returns (headers, events)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("POST", path)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    evs = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    return resp.status, dict(resp.getheaders()), evs


def _bg_stream(server, path):
    out = {}

    def run():
        out["status"], out["headers"], out["events"] = _stream(server.port,
                                                                path)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    out["thread"] = t
    return out


def _wait_status(server, cid, pred, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st, s = server.json("GET", f"/c/{cid}/ingest/status")
        if pred(s):
            return s
        time.sleep(0.05)
    raise AssertionError(f"status never matched: {s}")


def test_status_idle_then_complete(server, make_collection):
    col = make_collection(ingest=False)
    st, s = server.json("GET", f"/c/{col.id}/ingest/status")
    assert st == 200 and s == {"job": None, "state": "idle",
                               "running": False}
    st, evs = server.ndjson("POST", f"/c/{col.id}/ingest?summaries=0")
    assert evs[-1]["stage"] == "complete", evs[-1]
    chunk = [e for e in evs if e["stage"] == "chunk"]
    assert [e["done"] for e in chunk] == list(range(1, len(CLEAN_FILES) + 1))
    assert {e["total"] for e in chunk} == {len(CLEAN_FILES)}
    assert not [e for e in evs if e["stage"] == "summarize"]  # summaries=0
    s = _wait_status(server, col.id, lambda s: not s["running"])
    assert s["state"] == "complete" and s["stage"] == "complete"
    assert s["stages"]["chunk"] == {"done": len(CLEAN_FILES),
                                    "total": len(CLEAN_FILES)}
    assert s["stages"]["dates"] == {"done": 1, "total": 1}
    assert s["result"]["sources"] == len(CLEAN_FILES)
    assert s["events"] == len(evs) and s["summaries"] is False
    for e in evs:
        if e["stage"] == "embed" and "i" in e:
            assert 1 <= e["i"] <= e["of"] == len(CLEAN_FILES)
    if "embed" in s["stages"]:
        assert s["stages"]["embed"] == {"done": len(CLEAN_FILES),
                                        "total": len(CLEAN_FILES)}
    st, _ = server.json("GET", "/c/nope/ingest/status")
    assert st == 404


def test_summaries_stage_when_enabled(server, make_collection):
    col = make_collection(ingest=False)
    st, evs = server.ndjson("POST", f"/c/{col.id}/ingest")
    assert evs[-1]["stage"] == "complete"
    summ = [e for e in evs if e["stage"] == "summarize"]
    assert summ and "skipped" in summ[-1]["status"]      # dead LLM port


def test_cancel(server, make_collection, gate):
    col = make_collection(ingest=False)
    first = _bg_stream(server, f"/c/{col.id}/ingest?summaries=0")
    assert gate["entered"].wait(30)
    s = _wait_status(server, col.id, lambda s: s["running"])
    assert s["state"] == "running"
    st, res = server.json("POST", f"/c/{col.id}/ingest/cancel")
    assert st == 200 and res["cancelling"] is True
    assert res["cancel_requested"] is True
    gate["release"].set()
    first["thread"].join(30)
    evs = first["events"]
    assert evs[-1]["stage"] == "cancelled", evs[-1]
    assert not [e for e in evs if e["stage"] in ("complete", "embed")]
    s = _wait_status(server, col.id, lambda s: not s["running"])
    assert s["state"] == "cancelled"
    # consistent: only the finished source is recorded / has a chunk file
    done = [e["source"] for e in evs if e["stage"] == "chunk"]
    assert len(done) == 1
    assert set(ingest.load_manifest(col)) == set(done)
    assert sorted(p.name for p in col.chunks_dir.glob("*.jsonl")) == \
        [col.artifact_key(done[0]) + ".jsonl"]
    assert not list(col.chunks_dir.glob("*.tmp"))
    # the next run picks up the rest
    st, evs = server.ndjson("POST", f"/c/{col.id}/ingest?summaries=0")
    assert evs[-1]["stage"] == "complete"
    assert set(ingest.load_manifest(col)) == set(CLEAN_FILES)
    up = {e["source"] for e in evs if e["stage"] == "chunk"
          and e["status"] == "up to date"}
    assert up == set(done)

    st, res = server.json("POST", f"/c/{col.id}/ingest/cancel")
    assert st == 200 and res["cancelling"] is False      # nothing running


def test_second_client_attaches(server, make_collection, gate):
    col = make_collection(ingest=False)
    first = _bg_stream(server, f"/c/{col.id}/ingest?summaries=0")
    assert gate["entered"].wait(30)
    second = _bg_stream(server, f"/c/{col.id}/ingest?summaries=0&force=1")
    third = _bg_stream(server, f"/c/{col.id}/ingest?from=1")
    s = _wait_status(server, col.id, lambda s: s["clients"] == 3)
    assert s["running"]
    assert s["force"] is False                 # the attach did not restart
    # while running: import and collection delete are refused
    st, _ = server.request("POST", f"/c/{col.id}/import", b"x",
                           headers={"X-Filename": "n.txt"})
    assert st == 409
    st, _ = server.json("DELETE", f"/collections/{col.id}")
    assert st == 409 and library.get(col.id)
    gate["release"].set()
    first["thread"].join(60)
    second["thread"].join(60)
    third["thread"].join(60)
    assert first["headers"]["X-Ingest-Attached"] == "0"
    assert second["headers"]["X-Ingest-Attached"] == "1"
    assert first["headers"]["X-Ingest-Job"] == \
        second["headers"]["X-Ingest-Job"]
    # replay from the start: both clients see the identical event sequence
    assert first["events"] == second["events"]
    assert third["events"] == first["events"][1:]      # ?from=1
    assert first["events"][-1]["stage"] == "complete"
    # one job ran: every source chunked exactly once
    assert sorted(gate["calls"]) == sorted(CLEAN_FILES)


def test_attach_from_index(server, make_collection):
    col = make_collection(ingest=False)
    st, evs = server.ndjson("POST", f"/c/{col.id}/ingest?summaries=0")
    s = _wait_status(server, col.id, lambda s: not s["running"])
    assert s["events"] == len(evs)
    # a finished job is not re-attached: a new POST starts a new run
    st, again = server.ndjson("POST", f"/c/{col.id}/ingest?summaries=0")
    assert again[-1]["stage"] == "complete"
    s2 = server.json("GET", f"/c/{col.id}/ingest/status")[1]
    assert s2["job"] != s["job"]
    st, res = server.json("POST", f"/c/{col.id}/ingest?from=x")
    assert st == 400


def test_stream_disconnect_does_not_stop_job(server, make_collection, gate):
    col = make_collection(ingest=False)
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    conn.request("POST", f"/c/{col.id}/ingest?summaries=0")
    resp = conn.getresponse()
    assert resp.status == 200
    assert gate["entered"].wait(30)
    conn.close()                                  # the tab is closed
    gate["release"].set()
    s = _wait_status(server, col.id, lambda s: not s["running"])
    assert s["state"] == "complete"
    st, stats = server.json("GET", f"/c/{col.id}/stats")
    assert st == 200 and len(stats["works"]) == len(CLEAN_FILES)
    # the job is over: a new POST starts a fresh run (?from is ignored)
    st, again = server.ndjson("POST", f"/c/{col.id}/ingest?summaries=0"
                                      f"&from={s['events'] - 1}")
    assert again[-1]["stage"] == "complete" and len(again) > 1


def test_queued_behind_other_collection(server, make_collection, gate):
    a = make_collection(name="A", ingest=False)
    b = make_collection(name="B", files=("volcanoes.txt",), ingest=False)
    first = _bg_stream(server, f"/c/{a.id}/ingest?summaries=0")
    assert gate["entered"].wait(30)
    second = _bg_stream(server, f"/c/{b.id}/ingest?summaries=0")
    s = _wait_status(server, b.id, lambda s: s["state"] == "queued")
    assert s["running"] is True
    gate["release"].set()
    first["thread"].join(60)
    second["thread"].join(60)
    assert second["events"][0]["stage"] == "queued"
    assert second["events"][-1]["stage"] == "complete"
    assert first["events"][-1]["stage"] == "complete"
