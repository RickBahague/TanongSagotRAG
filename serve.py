#!/usr/bin/env python3
"""AnyRAG web UI — stdlib only. Run: python serve.py → http://127.0.0.1:8000

All corpus routes are collection-scoped under /c/<id>/…:

  GET    /collections               [{id, name, sources, chunks, …}]
  POST   /collections               {name, link?, description?, author?,
                                     persona?} → create
  DELETE /collections/<id>          remove collection + derived data
  GET    /c/<id>/stats              capabilities + per-source table
  POST   /c/<id>/ask                NDJSON stream: meta, delta, verify, done
  POST   /c/<id>/ingest             (re)ingest; NDJSON progress events
  POST   /c/<id>/import             file upload (X-Filename header)
  GET    /c/<id>/passage?id=…       windowed/full stitched passage (reader)
  GET    /c/<id>/read?id=…          full-section reader page
  GET    /c/<id>/timeline           dated sections (when dates exist)
  GET    /c/<id>/quote              random verbatim quote
  GET    /c/<id>/conversations      history; /conversation/<cid>[/export]
  GET    /c/<id>/source-meta        editable per-source metadata
  POST   /c/<id>/source-meta        {filename, title?, years?}
"""
import json
import re
import threading
import urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import config
import conversations
import library
import rag

INDEX_HTML_FILE = Path(__file__).parent / "static" / "index.html"

_RUNTIMES = {}                 # collection id → rag.Runtime (LRU-ish, max 3)
_RUNTIME_LOCK = threading.Lock()
_INGEST_LOCK = threading.Lock()   # one ingest at a time (embed/LLM heavy)


def runtime(cid, reload=False):
    """Load-once query runtime per collection; evict oldest beyond 3."""
    with _RUNTIME_LOCK:
        if reload:
            _RUNTIMES.pop(cid, None)
        if cid not in _RUNTIMES:
            col = library.get(cid)                    # KeyError → 404
            _RUNTIMES[cid] = rag.Runtime(col)
            while len(_RUNTIMES) > 3:
                _RUNTIMES.pop(next(iter(_RUNTIMES)))
        return _RUNTIMES[cid]


READ_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AnyRAG — Reader</title><style>
 body{margin:0;font:17px/1.7 Georgia,serif;background:#faf7f2;color:#2b2620}
 main{max-width:680px;margin:0 auto;padding:32px 20px 90px}
 h1{font-size:22px;color:#1b5e8c;margin:0 0 2px}
 .meta{color:#8a8072;font-style:italic;margin-bottom:24px;
       border-bottom:1px solid #e5ddd0;padding-bottom:12px}
 mark{background:#f3dfa8;padding:1px 0}
 #text{white-space:pre-wrap}
 #loading{color:#8a8072;font-style:italic}
 #backTo{position:fixed;bottom:18px;right:18px;font:13px system-ui,sans-serif;
   padding:8px 14px;border:1px solid #d8cfc0;background:#fff;border-radius:18px;
   cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none}
 #backTo:hover{color:#1b5e8c;border-color:#1b5e8c}</style></head><body><main>
 <h1 id="essay"></h1><div class="meta" id="meta"></div>
 <div id="loading">Loading the full section…</div>
 <div id="text"></div>
</main>
<button id="backTo" title="Scroll back to the cited passage">◎ cited passage</button>
<script>
const qs = new URLSearchParams(location.search);
const id = qs.get('id');
const base = location.pathname.replace(/\\/read$/, '');
let markEl = null;
async function load() {
  const r = await fetch(base + '/passage?id=' + encodeURIComponent(id) + '&full=1');
  const p = await r.json();
  document.getElementById('loading').remove();
  if (p.error) { document.getElementById('text').textContent = p.error; return; }
  document.getElementById('essay').textContent = p.essay;
  document.title = 'AnyRAG — ' + p.essay;
  document.getElementById('meta').textContent = p.work + ' · pp. ' +
      p.page_start + '–' + p.page_end;
  const t = document.getElementById('text');
  if (p.before) t.appendChild(document.createTextNode(p.before + ' '));
  markEl = document.createElement('mark');
  markEl.textContent = p.cited;
  t.appendChild(markEl);
  if (p.after) t.appendChild(document.createTextNode(' ' + p.after));
  markEl.scrollIntoView({block: 'center'});
}
document.getElementById('backTo').onclick = () =>
  markEl && markEl.scrollIntoView({behavior: 'smooth', block: 'center'});
addEventListener('scroll', () => {
  if (!markEl) return;
  const r = markEl.getBoundingClientRect();
  const off = r.bottom < 0 || r.top > innerHeight;
  document.getElementById('backTo').style.display = off ? 'block' : 'none';
});
load();
</script></body></html>"""


def import_work(col, filename, data):
    """Save an uploaded source into a MANAGED collection, run the cheap
    pipeline stages, and rebuild the runtime. Returns a summary dict."""
    import ingest
    name = re.sub(r"[^\w .\-()\[\]]", "_", Path(filename).name)
    if Path(name).suffix.lower() not in library.SUPPORTED_SUFFIXES:
        raise ValueError("Supported: " + ", ".join(library.SUPPORTED_SUFFIXES))
    if col.linked:
        raise ValueError("This collection links a folder on disk — add the "
                         "file there and press Rescan instead.")
    col.ensure_dirs()
    src = col.sources_dir / name
    src.write_bytes(data)

    # stage 1 — chunk (always first)
    res = ingest.chunk_source(col, src, force=True)
    _, n, rung = res
    ingest.detect_source_meta(col, src)
    # stage 1c/1d — dates + quote bank (hash-idempotent, cheap)
    import dates
    dates.run_all(col)
    import quotes_bank
    quotes_bank.run_all(col)
    # stage 2 — embed, only after chunking, if a query encoder is available
    embedded = False
    try:
        import numpy as np
        enc = rag._query_encoder()
        chunk_file = col.chunks_dir / (src.stem + ".jsonl")
        chunks = [json.loads(l) for l in open(chunk_file)]
        vecs = enc.encode([c["text"] for c in chunks],
                          normalize_embeddings=True, show_progress_bar=False)
        col.vectors_dir.mkdir(parents=True, exist_ok=True)
        np.savez(col.vectors_dir / (chunk_file.stem + ".npz"),
                 vectors=np.asarray(vecs, dtype="float32"),
                 model=np.array(config.EMBED_MODEL),
                 src_hash=np.array(ingest.file_hash(chunk_file)))
        embedded = True
    except ImportError:
        pass  # BM25 will cover it

    rt = runtime(col.id, reload=True)
    return {"work": col.work_title(name), "chunks": n, "embedded": embedded,
            "segmentation": rung, "retrieval": type(rt.index).__name__}


def collection_stats(rt):
    """Capabilities + per-source table for the UI."""
    col = rt.col
    import ingest
    manifest = ingest.load_manifest(col)
    sources = {}
    for c in rt.index.chunks:
        s = sources.setdefault(c["source"], {"chunks": 0, "essays": set()})
        s["chunks"] += 1
        s["essays"].add(c["essay"])
    works = [{"source": name, "title": col.work_title(name),
              "years": (col.source_meta.get(name, {}) or {}).get("years"),
              "chunks": s["chunks"], "essays": len(s["essays"]),
              "segmentation": manifest.get(name, {}).get("segmentation"),
              "embed_status": manifest.get(name, {}).get("embed_status")}
             for name, s in sorted(sources.items())]
    dated = sum(1 for c in rt.index.chunks
                if c.get("date") and c.get("date_precision") != "work")
    date_coverage = dated / max(len(rt.index.chunks), 1)
    return {"id": col.id, "name": col.name, "description": col.description,
            "author": col.author, "linked": col.linked,
            "folder": str(col.sources_dir),
            "works": works, "total_chunks": len(rt.index.chunks),
            "retrieval": type(rt.index).__name__,
            "modes": rag.available_modes(col),
            "has_summaries": bool(rag.load_summaries(col)),
            "has_quotes": bool(rt.quote_bank),
            "date_coverage": round(date_coverage, 3),
            "show_timeline": date_coverage >= 0.4,
            "backend": config.LLM_BACKEND, "model": config.LLM_MODEL}


def timeline(rt):
    """Sections with dates: precise ones on the timeline, work-precision
    grouped at the end."""
    seen, dated, coarse = set(), [], []
    for c in rt.index.chunks:
        key = (c["source"], c["essay"])
        if key in seen:
            continue
        seen.add(key)
        entry = {"essay": c["essay"], "work": c.get("work"),
                 "id": c["id"], "page_start": c["page_start"],
                 "date": c.get("date"), "precision": c.get("date_precision")}
        if c.get("date") and c.get("date_precision") != "work":
            entry["year"] = int(c["date"][:4])
            dated.append(entry)
        else:
            coarse.append(entry)
    dated.sort(key=lambda e: e["date"])
    coarse.sort(key=lambda e: (e["work"] or "", e["essay"]))
    return {"dated": dated, "undated": coarse}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _route(self):
        """(collection_id, tail_path, query) for /c/<id>/… paths."""
        from urllib.parse import urlparse, parse_qs
        url = urlparse(self.path)
        m = re.match(r"^/c/([\w-]+)(/.*)?$", url.path)
        if not m:
            return None, url.path, parse_qs(url.query)
        return m.group(1), m.group(2) or "/", parse_qs(url.query)

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        cid, path, qs = self._route()
        try:
            if cid is None:
                return self._get_global(path)
            rt = runtime(cid)
        except KeyError:
            return self._err(404, "unknown collection")
        try:
            self._get_collection(rt, path, qs)
        except BrokenPipeError:
            pass
        except Exception as e:
            self._err(400, str(e))

    def _get_global(self, path):
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML_FILE.read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/collections":
            out = []
            for col in library.list_all():
                out.append({"id": col.id, "name": col.name,
                            "author": col.author, "linked": col.linked,
                            "folder": str(col.sources_dir),
                            "sources": len(col.source_files()),
                            "indexed": col.chunks_dir.exists() and
                            any(col.chunks_dir.glob("*.jsonl"))})
            self._json(out)
        elif path == "/settings":
            self._json({"backend": config.LLM_BACKEND,
                        "model": config.LLM_MODEL, "url": config.LLM_URL,
                        "retrieval": config.RETRIEVAL,
                        "embed_model": config.EMBED_MODEL})
        else:
            self._err(404, "not found")

    def _get_collection(self, rt, path, qs):
        col = rt.col
        if path == "/stats":
            self._json(collection_stats(rt))
        elif path == "/passage":
            try:
                cidq = qs["id"][0]
                if qs.get("full", ["0"])[0] == "1":
                    c = rt.by_id[cidq]
                    n = len(rt.groups[(c["source"], c["essay"])])
                    before = after = n
                else:
                    before = min(int(qs.get("before", ["1"])[0]), 50)
                    after = min(int(qs.get("after", ["1"])[0]), 50)
                self._json(rt.passage(cidq, before=before, after=after))
            except KeyError:
                self._err(404, "unknown passage id")
        elif path == "/read":
            self._send(200, READ_PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/timeline":
            self._json(timeline(rt))
        elif path == "/quote":
            era = None
            if qs.get("after"):
                era = (int(qs["after"][0]), 9999)
            elif qs.get("era"):
                m = re.match(r"(\d{4})(?:-(\d{4}))?", qs["era"][0])
                if m:
                    era = (int(m.group(1)), int(m.group(2) or m.group(1)))
            q = rt.random_quote(topic=(qs.get("topic") or [None])[0], era=era)
            if q:
                self._json(q)
            else:
                self._err(404, "quote bank empty — ingest first")
        elif path == "/conversations":
            self._json(conversations.list_all(col))
        elif path == "/source-meta":
            self._json(col.source_meta)
        elif path.startswith("/conversation/"):
            seg = path.split("/")
            try:
                turns = conversations.load(col, seg[2])
            except (KeyError, ValueError):
                return self._err(404, "unknown conversation")
            if len(seg) > 3 and seg[3] == "export":
                fmt = qs.get("format", ["md"])[0]
                if fmt == "json":
                    body = json.dumps(turns, ensure_ascii=False, indent=1).encode()
                    ctype, ext = "application/json", "json"
                else:
                    body = conversations.to_markdown(
                        turns, f"{col.name} conversation").encode()
                    ctype, ext = "text/markdown; charset=utf-8", "md"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{col.id}-{seg[2]}.{ext}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(turns)
        else:
            self._err(404, "not found")

    # --------------------------------------------------------------- DELETE
    def do_DELETE(self):
        cid, path, _ = self._route()
        try:
            if cid is None:
                m = re.match(r"^/collections/([\w-]+)$", path)
                if not m:
                    return self._err(404, "not found")
                with _RUNTIME_LOCK:
                    _RUNTIMES.pop(m.group(1), None)
                library.delete(m.group(1))
                return self._json({"ok": True})
            col = library.get(cid)
            if path.startswith("/conversation/"):
                conversations.delete(col, path.split("/")[2])
                return self._json({"ok": True})
            self._err(404, "not found")
        except (KeyError, ValueError) as e:
            self._err(400, str(e))

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        cid, path, qs = self._route()
        if cid is None:
            if path == "/collections":
                return self._create_collection()
            return self._err(404, "not found")
        try:
            col = library.get(cid)
        except KeyError:
            return self._err(404, "unknown collection")
        if path == "/ask":
            self._do_ask(cid)
        elif path == "/ingest":
            self._do_ingest(col, qs)
        elif path == "/import":
            self._do_import(col)
        elif path == "/source-meta":
            self._do_source_meta(col)
        else:
            self._err(404, "not found")

    def _create_collection(self):
        try:
            req = self._body()
            if not req.get("name", "").strip():
                raise ValueError("a name is required")
            col = library.create(req["name"].strip(),
                                 link=req.get("link") or None,
                                 description=req.get("description", ""),
                                 author=req.get("author", ""),
                                 persona=req.get("persona"))
            self._json({"id": col.id, "name": col.name,
                        "sources": len(col.source_files())})
        except Exception as e:
            self._err(400, str(e))

    def _do_source_meta(self, col):
        try:
            req = self._body()
            years = req.get("years")
            if years and not (isinstance(years, list) and len(years) == 2):
                raise ValueError("years must be [lo, hi]")
            col.set_source_meta(req["filename"], title=req.get("title"),
                                years=years)
            runtime(col.id, reload=True)   # titles are attached at load time
            self._json({"ok": True})
        except Exception as e:
            self._err(400, str(e))

    def _ndjson_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _w(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def _do_ingest(self, col, qs):
        """Run the pipeline, streaming progress as NDJSON."""
        import ingest
        force = qs.get("force", ["0"])[0] == "1"
        summaries = qs.get("summaries", ["1"])[0] != "0"
        self._ndjson_start()
        if not _INGEST_LOCK.acquire(blocking=False):
            self._w({"stage": "error",
                     "status": "another ingest is already running"})
            return
        try:
            res = ingest.run_pipeline(col, force=force, summaries=summaries,
                                      progress=self._w, workers=1)
            try:
                runtime(col.id, reload=True)
            except SystemExit as e:      # no chunks produced
                self._w({"stage": "error", "status": str(e)})
                return
            self._w({"stage": "complete", **res})
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._w({"stage": "error", "status": str(e)})
            except BrokenPipeError:
                pass
        finally:
            _INGEST_LOCK.release()

    def _do_ask(self, cid):
        try:
            req = self._body()
        except Exception as e:
            return self._err(400, str(e))
        self._ndjson_start()
        w = self._w
        try:
            rt = runtime(cid)
            col = rt.col
            conv = req.get("conversation_id") or conversations.new_id()
            res = rt.ask_stream(req["question"],
                                mode=req.get("mode", "scholar"))
            w({"type": "meta", "mode": res["mode"], "sources": res["sources"],
               "conversation_id": conv})
            verify = None
            if res["stream"] is None:
                answer = res["answer"]
                w({"type": "delta", "text": answer})
            else:
                parts = []
                for t in res["stream"]:
                    parts.append(t)
                    w({"type": "delta", "text": t})
                answer = "".join(parts)
                # quotes mode: mechanical fidelity check after generation
                if res["mode"] == "quotes" and res["sources"]:
                    chunks = [rt.by_id[s["id"]] for s in res["sources"]
                              if s["id"] in rt.by_id]
                    verify = rag.verify_quotes(answer, chunks)
                    w({"type": "verify", "quotes": verify})
            conversations.append_turn(col, conv, req["question"], res["mode"],
                                      answer, res["sources"], verify)
            w({"type": "done"})
        except urllib.error.URLError:
            w({"type": "error", "error":
               f"Cannot reach {config.LLM_BACKEND} backend at {config.LLM_URL} "
               f"(model {config.LLM_MODEL}). Check Settings and that the "
               f"server is running."})
        except BrokenPipeError:
            pass  # client disconnected mid-stream
        except Exception as e:
            try:
                w({"type": "error", "error": str(e)})
            except BrokenPipeError:
                pass

    def _do_import(self, col):
        try:
            filename = self.headers.get("X-Filename", "upload.pdf")
            length = int(self.headers.get("Content-Length", 0))
            if length > 200 * 1024 * 1024:
                raise ValueError("File too large (200 MB max)")
            data = self.rfile.read(length)
            summary = import_work(col, filename, data)
            self._json(summary)
        except Exception as e:
            self._err(400, str(e))

    def log_message(self, fmt, *args):
        pass


def make_server(port=config.PORT):
    return ThreadingHTTPServer((config.HOST, port), Handler)


if __name__ == "__main__":
    n = len(library.list_all())
    print(f"AnyRAG on http://{config.HOST}:{config.PORT}  "
          f"({n} collection(s), {config.LLM_BACKEND}/{config.LLM_MODEL})")
    make_server().serve_forever()
