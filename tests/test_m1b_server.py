"""M1-B server: per-launch token, Origin/Content-Type rules, 409 for
un-indexed collections, and the runtime cache's single concurrent load."""
import http.client
import json
import threading
import time

import pytest

import library
import rag


@pytest.fixture
def token_server():
    import serve
    httpd = serve.make_server(0, token="s3cret-token")
    threading.Thread(target=httpd.serve_forever,
                     kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = dict(headers or {})
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp, raw


# ------------------------------------------------------------ token flow

def test_token_sets_cookie_and_redirects(token_server):
    port = token_server.server_address[1]
    resp, _ = _req(port, "GET", "/?token=s3cret-token")
    assert resp.status == 303 and resp.getheader("Location") == "/"
    cookie = resp.getheader("Set-Cookie")
    assert cookie.startswith(f"anyrag_token_{port}=s3cret-token")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    pair = cookie.split(";")[0]

    # with the cookie: the page and the API work
    resp, raw = _req(port, "GET", "/", headers={"Cookie": pair})
    assert resp.status == 200 and b"<html" in raw.lower()
    resp, raw = _req(port, "GET", "/collections", headers={"Cookie": pair})
    assert resp.status == 200 and json.loads(raw) == []
    resp, raw = _req(port, "POST", "/collections", {"name": "Tok"},
                     headers={"Cookie": pair})
    assert resp.status == 200, raw
    cid = json.loads(raw)["id"]
    resp, _ = _req(port, "GET", f"/c/{cid}/read?id=x",
                   headers={"Cookie": pair})
    assert resp.status == 409          # reader route is reachable (not 403)


def test_token_required_without_cookie(token_server):
    port = token_server.server_address[1]
    for method, path, body in (("GET", "/", None),
                               ("GET", "/collections", None),
                               ("POST", "/collections", {"name": "X"})):
        resp, raw = _req(port, method, path, body)
        assert resp.status == 403 and "error" in json.loads(raw)
    assert library.list_all() == []
    resp, _ = _req(port, "GET", "/?token=wrong")
    assert resp.status == 403 and resp.getheader("Set-Cookie") is None
    resp, _ = _req(port, "GET", "/collections",
                   headers={"Cookie": f"anyrag_token_{port}=wrong"})
    assert resp.status == 403


def test_no_token_mode_unchanged(server):
    st, raw = server.request("GET", "/")
    assert st == 200 and b"<html" in raw.lower()
    st, raw = server.request("GET", "/?token=anything")
    assert st == 200
    assert server.json("GET", "/collections") == (200, [])


# ------------------------------------------------------- origin / content

def test_own_origin_allowed(server):
    for origin in (f"http://127.0.0.1:{server.port}",
                   f"http://localhost:{server.port}"):
        st, body = server.json("POST", "/collections", {"name": origin[7:16]},
                               headers={"Origin": origin})
        assert st == 200, body
    st, _ = server.json("GET", "/collections",
                        headers={"Host": f"localhost:{server.port}"})
    assert st == 200


def test_other_port_origin_rejected(server):
    st, _ = server.json("POST", "/collections", {"name": "P"},
                        headers={"Origin": "http://127.0.0.1:1"})
    assert st == 403 and library.list_all() == []


def test_json_content_type_with_charset_ok(server):
    st, body = server.json("POST", "/collections", {"name": "Charset"},
                           headers={"Content-Type":
                                    "application/json; charset=utf-8"})
    assert st == 200, body
    st, body = server.json("POST", f"/c/{body['id']}/ask",
                           {"question": "x"},
                           headers={"Content-Type": "text/plain"})
    assert st == 415


# ----------------------------------------------------------- not indexed

def test_unindexed_ask_and_passage_409(server):
    _, col = server.json("POST", "/collections", {"name": "Empty"})
    cid = col["id"]
    st, body = server.json("POST", f"/c/{cid}/ask", {"question": "anything"})
    assert st == 409 and "not indexed" in body["error"]
    assert body["type"] == "error"        # also a valid NDJSON error event
    st, body = server.json("GET", f"/c/{cid}/passage?id=x")
    assert st == 409 and "not indexed" in body["error"]
    st, events = server.ndjson("POST", f"/c/{cid}/ingest?summaries=0")
    assert st == 200 and events[-1]["stage"] == "error"
    # the server is still alive and well
    assert server.json("GET", "/collections")[0] == 200


# ---------------------------------------------------------- runtime cache

def test_concurrent_runtime_loads_once(make_collection, monkeypatch):
    import serve
    cid = make_collection(files=("volcanoes.txt",)).id
    calls = []
    real = rag.Runtime

    def slow_runtime(col):
        calls.append(col.id)
        time.sleep(0.3)
        return real(col)
    monkeypatch.setattr(rag, "Runtime", slow_runtime)

    out = []
    threads = [threading.Thread(target=lambda: out.append(serve.runtime(cid)))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [cid]
    assert len(out) == 6 and all(rt is out[0] for rt in out)


def test_slow_load_does_not_block_other_collections(make_collection,
                                                    monkeypatch):
    import serve
    slow_id = make_collection("slow", files=("volcanoes.txt",)).id
    fast_id = make_collection("fast", files=("beekeeping.md",)).id
    serve.runtime(fast_id)                       # cached
    real = rag.Runtime
    started = threading.Event()

    def slow_runtime(col):
        started.set()
        time.sleep(1.0)
        return real(col)
    monkeypatch.setattr(rag, "Runtime", slow_runtime)

    t = threading.Thread(target=serve.runtime, args=(slow_id,))
    t.start()
    assert started.wait(5)
    t0 = time.monotonic()
    serve.runtime(fast_id)
    assert time.monotonic() - t0 < 0.5
    t.join()
    assert slow_id in serve._RUNTIMES


# ------------------------------------------------------------ transcript

def test_markdown_persona_disclaimer_neutral():
    import conversations
    turns = [{"ts": "2026-01-01T00:00:00", "question": "Q?", "mode": "persona",
              "answer": "*[old tag]* I wrote this.", "sources": []}]
    md = conversations.to_markdown(turns)
    assert "not the author." in md and "himself" not in md
    assert "old tag" not in md and "I wrote this." in md
    md = conversations.to_markdown(turns, author="Ada Lovelace")
    assert "collected writings of Ada Lovelace — not the author." in md
