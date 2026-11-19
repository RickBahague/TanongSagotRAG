"""Headless self-test of a build (CI's "Phase 5 smoke test").

  AnyRAG --smoke <docs_dir>             (frozen app)
  .venv/bin/python app.py --smoke <docs_dir>

app.py points ANYRAG_HOME at a temporary folder before anything is
imported, then calls run(). With the generated fixture corpus
(tests/fixtures/build.py <docs_dir>) it checks, inside the real bundle:
ingest of every format, search, the relevance gate, the dense encoder, the
reranker and llama.cpp's native library when they are bundled, and one HTTP
round trip through the server with the per-launch token. Exit code 0 = OK.
"""
import json
import sys
import threading
import urllib.error
import urllib.request

QUESTION = "How do worker bees build the wax comb?"
OFF_TOPIC = "Who won the 2018 FIFA World Cup?"


def run(docs):
    import config
    import encoder
    import ingest
    import library
    import rag
    import serve

    failures = []

    def check(ok, what):
        print(("  ok    " if ok else "  FAIL  ") + what)
        if not ok:
            failures.append(what)

    col = library.create("Smoke test", link=str(docs))
    ingest.run_pipeline(col, summaries=False, workers=1)
    col = library.get(col.id)
    n_src = len(list(col.chunks_dir.glob("*.jsonl")))
    check(n_src >= 5, f"ingest: {n_src} sources chunked")

    index = rag.load_index(col)
    dense = getattr(index, "dense", None) is not None
    check(dense or not (config.ONNX_ENCODER_DIR / "model.onnx").exists(),
          f"dense encoder {'loaded' if dense else 'not bundled'}")

    res = rag.search_passages(col, QUESTION, index=index)
    top = res["results"][0] if res["results"] else {}
    check(res["covered"] and "beekeeping" in json.dumps(top),
          "search finds the beekeeping passage and the gate accepts it")
    check(not rag.gate_ok(index, OFF_TOPIC), "gate rejects an off-topic question")

    if (config.ONNX_RERANKER_DIR / "model.onnx").exists():
        try:
            rr = encoder.get_reranker()
            a, b = rr.predict([(QUESTION, "Bees shape wax into comb cells."),
                               (QUESTION, "A lighthouse lamp at night.")])
            check(a > b, "reranker loads and ranks")
        except Exception as e:
            check(False, f"reranker failed: {e}")

    err = rag.llama_cpp_error()
    check(err is None, "llama.cpp native library loads"
          + ("" if err is None else f": {err}"))

    token = "smoke-token"
    httpd = serve.make_server(0, token=token)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        op = urllib.request.build_opener(NoRedirect)
        try:
            op.open(f"http://127.0.0.1:{port}/?token={token}", timeout=10)
            cookie = None
        except urllib.error.HTTPError as e:
            cookie = (e.headers.get("Set-Cookie") or "").split(";")[0]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/collections",
                                     headers={"Cookie": cookie or ""})
        with urllib.request.urlopen(req, timeout=10) as r:
            cols = json.loads(r.read())
        check(any(c["id"] == col.id for c in cols),
              "HTTP: token cookie accepted, /collections lists the collection")
    except Exception as e:
        check(False, f"HTTP round trip failed: {e}")
    finally:
        httpd.shutdown()

    print("SMOKE " + ("OK" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


def main(argv):
    if len(argv) < 1:
        print("usage: --smoke <docs_dir>", file=sys.stderr)
        return 2
    return run(argv[0])
