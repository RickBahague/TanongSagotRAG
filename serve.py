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

Local-server hardening (every request): the Host header must be
127.0.0.1/localhost on the bound port; an Origin header, when present, must
be the app's own origin; JSON POSTs need Content-Type application/json and
a body of at most 10 MB. When the server is made with a per-launch `token`
(app.py does this), GET /?token=<token> sets an HttpOnly SameSite=Strict
cookie and redirects to /; every other request must carry that cookie.
"""
import hmac
import json
import re
import threading
import urllib.error
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import config
import conversations
import errors
import library
import rag

INDEX_HTML_FILE = Path(__file__).parent / "static" / "index.html"

MAX_RUNTIMES = 3
MAX_JSON_BODY = 10 * 1024 * 1024       # JSON POST bodies
MAX_UPLOAD = 200 * 1024 * 1024          # /import raw uploads
# POST routes whose body is NOT JSON (exempt from the Content-Type rule)
_NON_JSON_POSTS = ("/ingest", "/import")

# rag.load_sources raises errors.NotIndexed for a collection with no chunks
# (older rag.py raised SystemExit); both mean "not indexed" to the server.
NOT_INDEXED = (errors.NotIndexed, SystemExit)

_RUNTIMES = OrderedDict()      # collection id → rag.Runtime (LRU, max 3)
_RUNTIME_LOCK = threading.Lock()  # guards _RUNTIMES/_LOAD_LOCKS only (fast)
_LOAD_LOCKS = {}               # collection id → Lock held while loading it
_INGEST_LOCK = threading.Lock()   # one ingest at a time (embed/LLM heavy)


def not_indexed_msg(cid):
    return (f"Collection {cid!r} is not indexed yet — run ingest "
            f"(Index it now / Rescan) first.")


def runtime(cid, reload=False):
    """Load-once query runtime per collection, LRU-evicted beyond
    MAX_RUNTIMES. The (slow) index load runs outside _RUNTIME_LOCK under a
    per-collection lock, so other collections are never blocked and the
    same collection is never loaded twice concurrently. With reload=True
    the old runtime keeps serving other requests until the new one is
    ready. Raises KeyError (unknown collection) or NOT_INDEXED."""
    with _RUNTIME_LOCK:
        if not reload and cid in _RUNTIMES:
            _RUNTIMES.move_to_end(cid)
            return _RUNTIMES[cid]
        load_lock = _LOAD_LOCKS.setdefault(cid, threading.Lock())
    with load_lock:
        if not reload:
            with _RUNTIME_LOCK:               # loaded while we waited?
                if cid in _RUNTIMES:
                    _RUNTIMES.move_to_end(cid)
                    return _RUNTIMES[cid]
        try:
            col = library.get(cid)                    # KeyError → 404
            rt = rag.Runtime(col)
        except BaseException:
            if reload:                        # stale index must not linger
                with _RUNTIME_LOCK:
                    _RUNTIMES.pop(cid, None)
            raise
        with _RUNTIME_LOCK:
            _RUNTIMES[cid] = rt
            _RUNTIMES.move_to_end(cid)
            while len(_RUNTIMES) > MAX_RUNTIMES:
                _RUNTIMES.popitem(last=False)
        return rt


def forget_runtime(cid):
    with _RUNTIME_LOCK:
        _RUNTIMES.pop(cid, None)
        _LOAD_LOCKS.pop(cid, None)


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
    # a schema-1 (stem-keyed) collection is re-keyed first, or the new
    # source's full-filename files would sit next to stale stem-keyed ones
    ingest.migrate_layout(col)
    src = col.sources_dir / name
    src.write_bytes(data)

    # stage 1 — chunk (always first). The new file can change the chunk-id
    # prefix of a same-stem sibling (ingest.chunk_id_prefixes): re-chunk
    # those too, and record every result in the manifest so the next plain
    # ingest sees them as up to date.
    manifest = ingest.load_manifest(col)
    params = ingest.chunk_params(col)
    prefixes = ingest.chunk_id_prefixes(s.name for s in col.source_files())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = rung = None
    touched = []
    cut = Path(name).stem[:ingest.ID_PREFIX_LEN]
    for s in col.source_files():
        if s.name != name and Path(s.name).stem[:ingest.ID_PREFIX_LEN] != cut:
            continue                         # unrelated source: untouched
        sha = ingest.file_hash(s)
        r = ingest.chunk_source(col, s, force=(s.name == name),
                                prev=manifest.get(s.name), src_sha256=sha,
                                id_prefix=prefixes[s.name])
        if r:
            ingest.record_chunked(manifest.setdefault(s.name, {}), r, now,
                                  sha, params, prefixes[s.name])
        touched.append(s.name)
        if s.name == name:
            _, n, rung = r
    ingest.save_manifest(col, manifest)
    ingest.detect_source_meta(col, src)
    # stage 1c/1d — dates + quote bank (hash-idempotent, cheap)
    import dates
    dates.run_all(col)
    import quotes_bank
    quotes_bank.run_all(col)
    # stage 2 — embed (after chunking) the new source and any re-chunked
    # sibling, with the same shared encoder and vector format as the ingest
    # pipeline; skipped without an encoder (BM25 then covers the source).
    # embed_source is hash-based, so unchanged siblings stay "up to date".
    embedded = False
    for s_name in touched:
        status, _ = ingest.embed_source(col, col.chunk_file(s_name))
        entry = manifest.setdefault(s_name, {})
        entry["embed_status"] = status
        if status.startswith("embedded"):
            entry["embedded_at"] = now
        if s_name == name:
            embedded = (status.startswith("embedded")
                        or status == "up to date")
    ingest.save_manifest(col, manifest)

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


class RequestError(Exception):
    """Refuse a request with an HTTP status before routing it."""

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def _cookie_name(port):
    # cookies are not port-isolated: two app instances on 127.0.0.1 must
    # not overwrite each other's token
    return f"anyrag_token_{port}"


class Handler(BaseHTTPRequestHandler):
    _streaming = False             # True once an NDJSON response has begun

    def _send(self, code, body, ctype="application/json", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        if code >= 400 and self.command in ("POST", "PUT", "DELETE"):
            # the request body may be unread: never reuse this connection
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _content_length(self, cap):
        """Declared body length, validated BEFORE anything is read."""
        raw = self.headers.get("Content-Length")
        if raw is None:
            if self.headers.get("Transfer-Encoding"):
                raise RequestError(411, "Content-Length required")
            return 0
        try:
            length = int(raw.strip())
        except ValueError:
            raise RequestError(400, "invalid Content-Length")
        if length < 0:
            raise RequestError(400, "invalid Content-Length")
        if length > cap:
            raise RequestError(413, f"request body too large "
                                    f"({cap // (1024 * 1024)} MB max)")
        return length

    def _body(self):
        """Parsed JSON body: Content-Type must be application/json and the
        declared length at most MAX_JSON_BODY (checked before reading)."""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0]
        if ctype.strip().lower() != "application/json":
            raise RequestError(415, "Content-Type must be application/json")
        length = self._content_length(MAX_JSON_BODY)
        if not length:
            return {}
        try:
            obj = json.loads(self.rfile.read(length))
        except ValueError:
            raise RequestError(400, "invalid JSON body")
        if not isinstance(obj, dict):
            raise RequestError(400, "JSON body must be an object")
        return obj

    def _route(self):
        """(collection_id, tail_path, query) for /c/<id>/… paths."""
        url = urlparse(self.path)
        m = re.match(r"^/c/([\w-]+)(/.*)?$", url.path)
        if not m:
            return None, url.path, parse_qs(url.query)
        return m.group(1), m.group(2) or "/", parse_qs(url.query)

    # ------------------------------------------------------------ guarding
    def _allowed_hosts(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _check_access(self):
        """Host allow-list, Origin check, per-launch token. Returns True
        when the request may proceed; otherwise a response was sent."""
        hosts = self._allowed_hosts()
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in hosts:
            self._err(403, "forbidden host")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and \
                origin.strip().lower() not in {"http://" + h for h in hosts}:
            self._err(403, "forbidden origin")
            return False
        token = getattr(self.server, "token", None)
        if not token:
            return True
        name = _cookie_name(self.server.server_address[1])
        url = urlparse(self.path)
        given = parse_qs(url.query).get("token", [None])[0]
        if self.command == "GET" and url.path in ("/", "/index.html") \
                and given is not None:
            if not hmac.compare_digest(given.encode(), token.encode()):
                self._err(403, "invalid token")
                return False
            self._send(303, b"", "text/plain", headers=(
                ("Location", "/"),
                ("Set-Cookie", f"{name}={token}; HttpOnly; "
                               f"SameSite=Strict; Path=/"),
                ("Cache-Control", "no-store")))
            return False
        from http.cookies import SimpleCookie, CookieError
        try:
            jar = SimpleCookie(self.headers.get("Cookie") or "")
        except CookieError:
            jar = {}
        sent = jar[name].value if name in jar else ""
        if not hmac.compare_digest(sent.encode(), token.encode()):
            self._err(403, "missing or invalid session token — open the "
                           "app from its launcher")
            return False
        return True

    def _guarded(self, handler):
        """Run a route handler behind the access checks; turn early
        refusals and "not indexed" into proper responses instead of a
        dropped connection."""
        self._streaming = False        # handler objects serve keep-alives
        if not self._check_access():
            return
        cid = self._route()[0]
        try:
            handler()
        except RequestError as e:
            if self._streaming:
                self._stream_error(str(e))
            else:
                self._err(e.code, str(e))
        except NOT_INDEXED:
            msg = not_indexed_msg(cid)
            if self._streaming:
                self._stream_error(msg)
            else:
                self._err(409, msg)
        except BrokenPipeError:
            pass

    def _stream_error(self, msg):
        try:
            self._w({"type": "error", "stage": "error", "error": msg,
                     "status": msg})
        except (BrokenPipeError, ConnectionError):
            pass

    def do_GET(self):
        self._guarded(self._do_GET)

    def do_POST(self):
        self._guarded(self._do_POST)

    def do_DELETE(self):
        self._guarded(self._do_DELETE)

    # ------------------------------------------------------------------ GET
    def _do_GET(self):
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
                        turns, f"{col.name} conversation",
                        author=col.author).encode()
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
    def _do_DELETE(self):
        cid, path, _ = self._route()
        try:
            if cid is None:
                m = re.match(r"^/collections/([\w-]+)$", path)
                if not m:
                    return self._err(404, "not found")
                forget_runtime(m.group(1))
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
    def _do_POST(self):
        cid, path, qs = self._route()
        if cid is None:
            if path == "/collections":
                return self._create_collection(self._body())
            return self._err(404, "not found")
        # JSON routes: Content-Type + size are enforced before anything else
        req = None if path in _NON_JSON_POSTS else self._body()
        try:
            col = library.get(cid)
        except KeyError:
            return self._err(404, "unknown collection")
        if path == "/ask":
            self._do_ask(cid, req)
        elif path == "/ingest":
            self._do_ingest(col, qs)
        elif path == "/import":
            self._do_import(col)
        elif path == "/source-meta":
            self._do_source_meta(col, req)
        else:
            self._err(404, "not found")

    def _create_collection(self, req):
        try:
            if not str(req.get("name") or "").strip():
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

    def _do_source_meta(self, col, req):
        try:
            years = req.get("years")
            if years and not (isinstance(years, list) and len(years) == 2):
                raise ValueError("years must be [lo, hi]")
            col.set_source_meta(req["filename"], title=req.get("title"),
                                years=years)
            runtime(col.id, reload=True)   # titles are attached at load time
            self._json({"ok": True})
        except NOT_INDEXED:
            self._err(409, not_indexed_msg(col.id))
        except Exception as e:
            self._err(400, str(e))

    def _ndjson_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._streaming = True

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
            except NOT_INDEXED:          # no chunks produced
                self._w({"stage": "error", "error": not_indexed_msg(col.id),
                         "status": "no passages were produced — "
                                   + not_indexed_msg(col.id)})
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

    def _do_ask(self, cid, req):
        # resolve the runtime BEFORE the stream starts, so "not indexed" is
        # a real 409 (body is also a valid one-line NDJSON error event)
        try:
            rt = runtime(cid)
        except NOT_INDEXED:
            msg = not_indexed_msg(cid)
            return self._json({"type": "error", "error": msg}, 409)
        self._ndjson_start()
        w = self._w
        try:
            col = rt.col
            conv = req.get("conversation_id") or conversations.new_id()
            res = rt.ask_stream(req["question"],
                                mode=req.get("mode", "scholar"))
            w({"type": "meta", "mode": res["mode"], "sources": res["sources"],
               "conversation_id": conv, "dropped": res.get("dropped", 0)})
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
        filename = self.headers.get("X-Filename", "upload.pdf")
        if self.headers.get("Content-Length") is None:
            raise RequestError(411, "Content-Length required")
        length = self._content_length(MAX_UPLOAD)   # 413 before reading
        try:
            data = self.rfile.read(length)
            summary = import_work(col, filename, data)
            self._json(summary)
        except NOT_INDEXED:
            self._err(409, "no passages could be extracted from "
                      f"{filename!r} — " + not_indexed_msg(col.id))
        except Exception as e:
            self._err(400, str(e))

    def log_message(self, fmt, *args):
        pass


def make_server(port=config.PORT, token=None):
    """Threaded server on config.HOST:port (0 = any free port). With a
    `token`, every request must carry the per-launch session cookie, which
    GET /?token=<token> sets (see Handler._check_access)."""
    httpd = ThreadingHTTPServer((config.HOST, port), Handler)
    httpd.token = token or None
    return httpd


if __name__ == "__main__":
    n = len(library.list_all())
    print(f"AnyRAG on http://{config.HOST}:{config.PORT}  "
          f"({n} collection(s), {config.LLM_BACKEND}/{config.LLM_MODEL})")
    make_server().serve_forever()
