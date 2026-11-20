#!/usr/bin/env python3
"""AnyRAG web UI — stdlib only. Run: python serve.py → http://127.0.0.1:8000

All corpus routes are collection-scoped under /c/<id>/…:

  GET    /collections               [{id, name, sources, size_bytes, …}]
  POST   /collections               {name, link?, description?, author?,
                                     persona?} → create
  DELETE /collections/<id>          remove collection + derived data
  GET    /settings                  global settings: effective values,
  POST   /settings                  defaults, overrides, allow-list
                                     ({KEY: value|null}; null = remove)
  GET    /backend/status            LLM backend reachability / models
  PATCH  /c/<id>                    {name?, description?, author?, persona?}
  GET    /c/<id>/settings           per-collection settings (same shape)
  POST   /c/<id>/settings           {KEY: value|null}
  GET    /c/<id>/stats              capabilities + per-source table
  POST   /c/<id>/ask                NDJSON stream: meta, delta…, then
                                     verify (quotes) or citations
                                     (scholar/persona), done. A gated-out
                                     question's meta has "closest": the
                                     nearest passages (no model call)
  POST   /c/<id>/search             search only, no LLM: {question, k?} →
                                     {question, covered, results[]}
  POST   /c/<id>/ingest             start (or attach to) the collection's
                                     background ingest job; NDJSON view of
                                     its events (?force=1 ?summaries=0
                                     ?source=<name> ?from=<event index>)
  GET    /c/<id>/ingest/status      job state + per-stage done/total
  POST   /c/<id>/ingest/cancel      cooperative cancel
  DELETE /c/<id>/source/<name>      remove a MANAGED source + its artifacts
  POST   /c/<id>/import             file upload (X-Filename header)
  GET    /c/<id>/passage?id=…       windowed/full stitched passage (reader)
  GET    /c/<id>/read?id=…          full-section reader page
  GET    /c/<id>/timeline           dated sections (when dates exist)
  GET    /c/<id>/quote              random verbatim quote
  GET    /c/<id>/conversations      history; /conversation/<cid>[/export]
  GET    /c/<id>/source-meta        editable per-source metadata
  POST   /c/<id>/source-meta        {filename, title?, years?}
  GET    /logs                      {path, folder, tail} of the app log
  GET    /model/download            first-run model download status
  POST   /model/download            start/resume it; /model/download/cancel
  POST   /logs/open                 open the log folder (file manager)

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
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import logging

import applog
import config
import modeldl
import conversations
import errors
import library
import rag

INDEX_HTML_FILE = Path(__file__).parent / "static" / "index.html"
log = logging.getLogger("anyrag.serve")

MAX_RUNTIMES = 3
MAX_JSON_BODY = 10 * 1024 * 1024       # JSON POST bodies
MAX_UPLOAD = 200 * 1024 * 1024          # /import raw uploads
# POST routes whose body is NOT JSON (exempt from the Content-Type rule)
_NON_JSON_POSTS = ("/ingest", "/import", "/ingest/cancel")
BACKEND_TIMEOUT = 2.0                   # /backend/status probe (seconds)

# rag.load_sources raises errors.NotIndexed for a collection with no chunks
# (older rag.py raised SystemExit); both mean "not indexed" to the server.
NOT_INDEXED = (errors.NotIndexed, SystemExit)

_RUNTIMES = OrderedDict()      # collection id → rag.Runtime (LRU, max 3)
_RUNTIME_LOCK = threading.Lock()  # guards _RUNTIMES/_LOAD_LOCKS only (fast)
_LOAD_LOCKS = {}               # collection id → Lock held while loading it
_INGEST_LOCK = threading.Lock()   # one ingest at a time (embed/LLM heavy)
_JOBS = {}                     # collection id → latest IngestJob
_JOBS_LOCK = threading.Lock()


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


def forget_all_runtimes():
    """Drop every loaded runtime (after a global settings change: the next
    request rebuilds each index with the new values)."""
    with _RUNTIME_LOCK:
        _RUNTIMES.clear()


# ------------------------------------------------------------ ingest jobs --

class IngestJob:
    """One background ingest run of one collection (P3-4). Events from
    ingest.run_pipeline are appended to `events` (never dropped); any number
    of HTTP clients follow them (follow()), so closing a tab does not stop
    the run and a second tab can attach. The final event is always one of
    stage "complete", "error" or "cancelled"."""

    FINAL = ("complete", "error", "cancelled")

    def __init__(self, col, force=False, summaries=True, only=None):
        self.id = uuid.uuid4().hex[:12]
        self.col = col
        self.force, self.summaries, self.only = force, summaries, only
        self.events = []
        self.cond = threading.Condition()
        self.cancel_event = threading.Event()
        self.state = "running"
        self.stage = self.source = None
        self.done = self.total = None
        self.stages = {}
        self.clients = 0               # streams currently following
        self.result = self.error = None
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.finished = None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"ingest-{col.id}")

    @property
    def running(self):
        return self.state in ("running", "queued")

    def emit(self, ev):
        with self.cond:
            self.events.append(ev)
            st = ev.get("stage")
            self.stage = st
            self.source = ev.get("source") or None
            self.done, self.total = ev.get("done"), ev.get("total")
            if st == "embed" and "i" in ev:
                # stage-level count = sources: the batch events of source i
                # mean i-1 sources are finished, its final event i
                fin = "done" not in ev
                self.stages["embed"] = {"done": ev["i"] - (0 if fin else 1),
                                        "total": ev["of"]}
            elif st in ("chunk", "summarize", "dates", "quotes") \
                    and ev.get("total") is not None \
                    and ev.get("done") is not None:
                self.stages[st] = {"done": ev["done"], "total": ev["total"]}
            self.cond.notify_all()

    def _finish(self, state, ev):
        self.emit(ev)
        with self.cond:
            self.state = state
            self.finished = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            self.cond.notify_all()

    def _run(self):
        import ingest
        col = self.col
        if not _INGEST_LOCK.acquire(blocking=False):
            self.state = "queued"
            self.emit({"stage": "queued", "status": "waiting for another "
                       "collection's ingest to finish"})
            while not _INGEST_LOCK.acquire(timeout=0.25):
                if self.cancel_event.is_set():
                    return self._finish("cancelled", {
                        "stage": "cancelled",
                        "status": "ingest cancelled before it started"})
            self.state = "running"
        try:
            res = ingest.run_pipeline(
                col, force=self.force, summaries=self.summaries,
                progress=self.emit, workers=1,
                cancel=self.cancel_event.is_set, only=self.only)
            try:
                runtime(col.id, reload=True)
            except NOT_INDEXED:          # no chunks produced
                self.error = not_indexed_msg(col.id)
                return self._finish("error", {
                    "stage": "error", "error": self.error,
                    "status": "no passages were produced — " + self.error})
            self.result = res
            self._finish("complete", {"stage": "complete", **res})
        except ingest.Cancelled:
            try:                         # serve what is on disk now
                runtime(col.id, reload=True)
            except BaseException:
                forget_runtime(col.id)
            self._finish("cancelled", {
                "stage": "cancelled",
                "status": "ingest cancelled — finished sources are kept; "
                          "the rest is redone on the next ingest"})
        except Exception as e:
            self.error = str(e)
            self._finish("error", {"stage": "error", "status": str(e),
                                   "error": str(e)})
        finally:
            _INGEST_LOCK.release()

    def follow(self, start=0):
        """Yield events from index `start` on, live, until the job ends."""
        i = max(start, 0)
        with self.cond:
            self.clients += 1
        try:
            while True:
                with self.cond:
                    while i >= len(self.events) and self.running:
                        self.cond.wait(timeout=1.0)
                    batch = self.events[i:]
                    ended = not self.running
                for ev in batch:
                    yield ev
                i += len(batch)
                if ended and i >= len(self.events):
                    return
        finally:
            with self.cond:
                self.clients -= 1

    def status(self):
        with self.cond:
            return {"job": self.id, "state": self.state,
                    "running": self.running,
                    "cancel_requested": self.cancel_event.is_set(),
                    "stage": self.stage, "source": self.source,
                    "done": self.done, "total": self.total,
                    "stages": {k: dict(v) for k, v in self.stages.items()},
                    "force": self.force, "summaries": self.summaries,
                    "only": self.only, "started": self.started,
                    "finished": self.finished, "events": len(self.events),
                    "clients": self.clients,
                    "result": self.result, "error": self.error,
                    "last": self.events[-1] if self.events else None}


def start_ingest(col, force=False, summaries=True, only=None):
    """(job, attached): the collection's running job when there is one
    (attached=True — the new request's options are ignored), else a newly
    started job."""
    with _JOBS_LOCK:
        job = _JOBS.get(col.id)
        if job is not None and job.running:
            return job, True
        job = IngestJob(col, force=force, summaries=summaries, only=only)
        _JOBS[col.id] = job
        job.thread.start()
        return job, False


def running_job(cid):
    with _JOBS_LOCK:
        job = _JOBS.get(cid)
    return job if job is not None and job.running else None


def ingest_status(cid):
    with _JOBS_LOCK:
        job = _JOBS.get(cid)
    if job is None:
        return {"job": None, "state": "idle", "running": False}
    return job.status()


# --------------------------------------------------------- backend status --

def backend_status(backend=None, url=None, model=None, model_path=None,
                   timeout=BACKEND_TIMEOUT):
    """Can the configured (or given) LLM backend answer? Never blocks longer
    than ~`timeout` per probe. Returns {backend, url, model, reachable,
    model_present (bool|None = unknown), models (list|None), error}; for
    "embedded" also {model_path, file_exists, llama_cpp}."""
    backend = backend or config.get("LLM_BACKEND")
    url = (url or config.get("LLM_URL") or "").rstrip("/")
    model = model or config.get("LLM_MODEL")
    out = {"backend": backend, "url": url, "model": model,
           "reachable": False, "model_present": None, "models": None,
           "error": None}
    if backend == "embedded":
        path = Path(model_path or config.get("LLM_MODEL_PATH"))
        exists = path.is_file()
        lib_err = rag.llama_cpp_error()     # imports it: catches a missing .dylib
        out.update(url=None, model=path.name, model_path=str(path),
                   file_exists=exists, llama_cpp=lib_err is None,
                   model_present=exists, reachable=exists and lib_err is None,
                   downloadable=(not exists
                                 and modeldl.is_download_target(path)),
                   download=modeldl.status())
        if not exists:
            out["error"] = (f"model not downloaded yet: {path}"
                            if out["downloadable"] else
                            f"model file missing: {path}")
        elif lib_err:
            out["error"] = lib_err
        return out
    if backend not in ("ollama", "openai"):
        out["error"] = f"unknown backend {backend!r}"
        return out
    if not re.match(r"^https?://", url):
        out["error"] = f"invalid URL {url!r}"
        return out
    probe = url + ("/api/tags" if backend == "ollama" else "/models")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(probe, timeout=timeout) as r:
            data = json.loads(r.read(4 * 1024 * 1024) or b"{}")
        out["reachable"] = True
        if backend == "ollama":
            names = [m.get("name") or m.get("model")
                     for m in data.get("models") or [] if isinstance(m, dict)]
        else:
            names = [m.get("id") for m in data.get("data") or []
                     if isinstance(m, dict)]
        names = sorted(n for n in names if isinstance(n, str))
        out["models"] = names
        if backend == "ollama":
            out["model_present"] = (model in names or (
                ":" not in model and f"{model}:latest" in names))
        else:
            out["model_present"] = (model in names) if names else None
    except urllib.error.HTTPError as e:
        out["reachable"] = True          # a server answered, just not this
        out["error"] = f"{probe} answered HTTP {e.code}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", e)
        out["error"] = f"cannot reach {backend} at {url}: {reason}"
    if out["reachable"] and out["model_present"] is False:
        out["error"] = f"model {model!r} is not available on the server" + (
            f" — run: ollama pull {model}" if backend == "ollama" else "")
    return out


# --------------------------------------------------------------- settings --

def global_settings():
    keys = config.SETTINGS_SCHEMA
    overrides = config.user_overrides()
    eff = {k: config.get(k) for k in keys}
    defaults = {k: config.default_value(k) for k in keys}
    return {"backend": config.get("LLM_BACKEND"),
            "model": config.get("LLM_MODEL"), "url": config.get("LLM_URL"),
            "retrieval": config.get("RETRIEVAL"),
            "embed_model": config.get("EMBED_MODEL"),
            "file": str(config.SETTINGS_FILE),
            "overrides": overrides,
            "settings": config.describe_settings(eff, overrides, defaults)}


def collection_settings(col):
    overrides = col.overrides()
    keys = config.SETTINGS_SCHEMA
    defaults = {k: config.get(k) for k in keys}
    eff = dict(defaults, **overrides)
    return {"id": col.id, "overrides": overrides,
            "settings": config.describe_settings(eff, overrides, defaults,
                                                 scope="collection")}


def _settings_error(errors):
    return {"error": "invalid settings: " + "; ".join(
        f"{k}: {m}" for k, m in sorted(errors.items())), "errors": errors}


def collection_summary(col):
    return {"id": col.id, "name": col.name, "description": col.description,
            "author": col.author, "persona": col.meta.get("persona"),
            "linked": col.linked, "folder": str(col.sources_dir)}


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
    prefixes = ingest.chunk_id_prefixes(col.source_name(s)
                                        for s in col.source_files())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = rung = None
    touched = []
    cut = Path(name).stem[:ingest.ID_PREFIX_LEN]
    for s in col.source_files():
        sn = col.source_name(s)
        if sn != name and Path(sn).stem[:ingest.ID_PREFIX_LEN] != cut:
            continue                         # unrelated source: untouched
        sha = ingest.file_hash(s)
        r = ingest.chunk_source(col, s, force=(sn == name),
                                prev=manifest.get(sn), src_sha256=sha,
                                id_prefix=prefixes[sn])
        if r:
            ingest.record_chunked(manifest.setdefault(sn, {}), r, now,
                                  sha, params, prefixes[sn])
        touched.append(sn)
        if sn == name:
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
    # sources with no passages (failed, text-less, not yet chunked) are
    # listed too, so their error / no_text flags reach the UI
    for p in col.source_files():
        sources.setdefault(col.source_name(p), {"chunks": 0,
                                                 "essays": set()})
    works = [{"source": name, "title": col.work_title(name),
              "years": (col.source_meta.get(name, {}) or {}).get("years"),
              "chunks": s["chunks"], "essays": len(s["essays"]),
              "segmentation": manifest.get(name, {}).get("segmentation"),
              "embed_status": manifest.get(name, {}).get("embed_status"),
              "error": manifest.get(name, {}).get("error"),
              "no_text": bool(manifest.get(name, {}).get("no_text"))}
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
            "backend": config.get("LLM_BACKEND"),
            "model": config.get("LLM_MODEL")}


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
        if code >= 400 and self.command in ("POST", "PUT", "PATCH",
                                            "DELETE"):
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
        except Exception:
            log.exception("%s %s failed", self.command, self.path.split("?")[0])
            if not self._streaming:
                try:
                    self._err(500, "internal error — see the log "
                              "(Settings → Open log folder)")
                except Exception:
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

    def do_PATCH(self):
        self._guarded(self._do_PATCH)

    # ------------------------------------------------------------------ GET
    def _do_GET(self):
        cid, path, qs = self._route()
        try:
            if cid is None:
                return self._get_global(path, qs)
            if path == "/settings":          # no index needed
                return self._json(collection_settings(library.get(cid)))
            if path == "/ingest/status":
                library.get(cid)
                return self._json(ingest_status(cid))
            rt = runtime(cid)
        except KeyError:
            return self._err(404, "unknown collection")
        try:
            self._get_collection(rt, path, qs)
        except BrokenPipeError:
            pass
        except Exception as e:
            self._err(400, str(e))

    def _get_global(self, path, qs=None):
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML_FILE.read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/collections":
            out = []
            for col in library.list_all():
                files = col.source_files()
                out.append({"id": col.id, "name": col.name,
                            "author": col.author,
                            "description": col.description,
                            "linked": col.linked,
                            "folder": str(col.sources_dir),
                            "sources": len(files),
                            "indexed": col.chunks_dir.exists() and
                            any(col.chunks_dir.glob("*.jsonl")),
                            "ingesting": running_job(col.id) is not None,
                            **col.disk_usage(files)})
            self._json(out)
        elif path == "/settings":
            self._json(global_settings())
        elif path == "/backend/status":
            q = {k: (qs or {}).get(k, [None])[0]
                 for k in ("backend", "url", "model", "model_path")}
            self._json(backend_status(**q))
        elif path == "/model/download":
            self._json(modeldl.status())
        elif path == "/logs":
            self._json({"path": str(applog.log_file()),
                        "folder": str(applog.log_dir()),
                        "tail": applog.tail(200)})
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
                if running_job(m.group(1)):
                    return self._err(409, "an ingest is running for this "
                                          "collection — cancel it first")
                forget_runtime(m.group(1))
                library.delete(m.group(1))
                return self._json({"ok": True})
            col = library.get(cid)
            if path.startswith("/source/"):
                return self._delete_source(col, unquote(path[len("/source/"):]))
            if path.startswith("/conversation/"):
                conversations.delete(col, path.split("/")[2])
                return self._json({"ok": True})
            self._err(404, "not found")
        except (KeyError, ValueError) as e:
            self._err(400, str(e))

    def _delete_source(self, col, name):
        import ingest
        if col.linked:
            return self._err(409, f"This collection links your folder "
                             f"{col.sources_dir} — AnyRAG never deletes "
                             f"files there. Remove or move the file in that "
                             f"folder yourself, then Rescan.")
        if running_job(col.id):
            return self._err(409, "an ingest is running for this collection "
                                  "— wait for it or cancel it first")
        try:
            res = ingest.remove_source(col, name)
        except KeyError:
            return self._err(404, f"unknown source {name!r}")
        forget_runtime(col.id)
        try:
            runtime(col.id, reload=True)
            indexed = True
        except NOT_INDEXED:
            indexed = False
        self._json({"ok": True, **res, "indexed": indexed})

    # ----------------------------------------------------------------- PATCH
    def _do_PATCH(self):
        cid, path, _ = self._route()
        if cid is None or path != "/":
            return self._err(404, "not found")
        req = self._body()
        try:
            library.get(cid)
        except KeyError:
            return self._err(404, "unknown collection")
        errors = library.validate_update(req)
        if errors:
            return self._json({"error": "invalid fields: " + "; ".join(
                f"{k}: {m}" for k, m in sorted(errors.items())),
                "errors": errors}, 400)
        col = library.update(cid, **req)
        forget_runtime(cid)             # prompts read name/author at load
        self._json(collection_summary(col))

    # ------------------------------------------------------------------ POST
    def _do_POST(self):
        cid, path, qs = self._route()
        if cid is None:
            if path == "/collections":
                return self._create_collection(self._body())
            if path == "/settings":
                return self._post_global_settings(self._body())
            if path == "/model/download":
                self._body()
                return self._json(modeldl.start())
            if path == "/model/download/cancel":
                self._body()
                return self._json(modeldl.cancel())
            if path == "/logs/open":
                self._body()            # JSON POST rules apply (CSRF)
                return self._json({"folder": str(applog.open_folder())})
            return self._err(404, "not found")
        # JSON routes: Content-Type + size are enforced before anything else
        req = None if path in _NON_JSON_POSTS else self._body()
        try:
            col = library.get(cid)
        except KeyError:
            return self._err(404, "unknown collection")
        if path == "/ask":
            self._do_ask(cid, req)
        elif path == "/search":
            self._do_search(cid, req)
        elif path == "/ingest":
            self._do_ingest(col, qs)
        elif path == "/ingest/cancel":
            job = running_job(col.id)
            if job is not None:
                job.cancel_event.set()
            self._json({"ok": True, "cancelling": job is not None,
                        **ingest_status(col.id)})
        elif path == "/settings":
            errors = col.save_settings(req)
            if errors:
                return self._json(_settings_error(errors), 400)
            forget_runtime(col.id)
            self._json(collection_settings(library.get(col.id)))
        elif path == "/import":
            self._do_import(col)
        elif path == "/source-meta":
            self._do_source_meta(col, req)
        else:
            self._err(404, "not found")

    def _post_global_settings(self, req):
        errors = config.save_user_settings(req)
        if errors:
            return self._json(_settings_error(errors), 400)
        forget_all_runtimes()           # indexes rebuilt with new values
        rag.reset_backend()             # embedded model reloads if changed
        self._json(global_settings())

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

    def _ndjson_start(self, headers=()):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self._streaming = True

    def _w(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def _do_ingest(self, col, qs):
        """Start the collection's background ingest job, or attach to the
        one already running (then this request's options are ignored), and
        stream the job's events as NDJSON until it ends. An attaching
        client gets a REPLAY of the job's events from the start (or from
        ?from=<n>, an index into them — GET /ingest/status reports
        "events"), then the live tail, so it sees the same sequence as the
        first client; ?from is ignored when a new job starts. Closing the
        stream never stops the job.
        Response headers: X-Ingest-Job (job id), X-Ingest-Attached (1/0)."""
        force = qs.get("force", ["0"])[0] == "1"
        summaries = qs.get("summaries", ["1"])[0] != "0"
        only = qs.get("source", [None])[0]
        try:
            start = int(qs.get("from", ["0"])[0])
        except ValueError:
            raise RequestError(400, "from must be an integer")
        if only is not None and col.source_path(only) is None:
            raise RequestError(404, f"unknown source {only!r}")
        job, attached = start_ingest(col, force=force, summaries=summaries,
                                     only=only)
        self._ndjson_start(headers=(("X-Ingest-Job", job.id),
                                    ("X-Ingest-Attached",
                                     "1" if attached else "0")))
        for ev in job.follow(start if attached else 0):
            self._w(ev)

    def _do_search(self, cid, req):
        """Search-only mode (P2-9): ranked passages, no model call, nothing
        saved to the conversation."""
        q = req.get("question") if isinstance(req, dict) else None
        if not isinstance(q, str) or not q.strip():
            raise RequestError(400, "question must be a non-empty string")
        k = req.get("k")
        if k is not None and (isinstance(k, bool) or not isinstance(k, int)
                              or not 1 <= k <= 50):
            raise RequestError(400, "k must be an integer from 1 to 50")
        try:
            rt = runtime(cid)
        except NOT_INDEXED:
            return self._json({"type": "error",
                               "error": not_indexed_msg(cid)}, 409)
        self._json(rt.search(q.strip(), k=k))

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
            given = req.get("conversation_id")
            conv = given or conversations.new_id()
            # follow-up memory (P2-1): last turns of an existing conversation
            history = conversations.recent(
                col, given, int(col.setting("FOLLOWUP_TURNS", 3))) \
                if given else []
            res = rt.ask_stream(req["question"],
                                mode=req.get("mode", "scholar"),
                                history=history)
            # sources are in prompt order: answer marker [n] == sources[n-1]
            w({"type": "meta", "mode": res["mode"], "sources": res["sources"],
               "conversation_id": conv, "dropped": res.get("dropped", 0),
               "rewritten_question": res.get("rewritten_question"),
               "history_used": res.get("history_used", 0),
               "closest": res.get("closest") or []})
            verify = citations = None
            if res["stream"] is None:
                answer = res["answer"]
                w({"type": "delta", "text": answer})
            else:
                parts = []
                for t in res["stream"]:
                    parts.append(t)
                    w({"type": "delta", "text": t})
                answer = "".join(parts)
                if res["mode"] != "quotes":
                    # paragraphs + bare [n] markers (plan.md "Answer prose");
                    # the UI swaps the streamed text for this one
                    composed = rag.compose_answer(answer, res.get("excerpts"))
                    if composed != answer:
                        answer = composed
                        w({"type": "final", "text": answer})
                # quotes mode: mechanical fidelity check after generation
                if res["mode"] == "quotes" and res["sources"]:
                    chunks = [rt.by_id[s["id"]] for s in res["sources"]
                              if s["id"] in rt.by_id]
                    verify = rag.verify_quotes(
                        answer, chunks, col.setting("QUOTE_MATCH_RATIO"))
                    w({"type": "verify", "quotes": verify})
                elif res["mode"] != "quotes":
                    # scholar/persona: map [n] markers onto the sources
                    citations = rag.check_citations(answer,
                                                    len(res["sources"]))
                    w({"type": "citations", **citations})
            conversations.append_turn(
                col, conv, req["question"], res["mode"], answer,
                res["sources"], verify, citations=citations,
                rewritten_question=res.get("rewritten_question"))
            w({"type": "done"})
        except urllib.error.URLError:
            w({"type": "error", "error":
               f"Cannot reach {config.get('LLM_BACKEND')} backend at "
               f"{config.get('LLM_URL')} (model {config.get('LLM_MODEL')})."
               f" Check Settings and that the "
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
        if running_job(col.id):
            raise RequestError(409, "an ingest is running for this "
                                    "collection — wait for it to finish")
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
        pass                            # no per-request access log

    def log_error(self, fmt, *args):
        log.warning("http: " + fmt, *args)


def make_server(port=config.PORT, token=None):
    """Threaded server on config.HOST:port (0 = any free port). With a
    `token`, every request must carry the per-launch session cookie, which
    GET /?token=<token> sets (see Handler._check_access)."""
    httpd = ThreadingHTTPServer((config.HOST, port), Handler)
    httpd.handle_error = lambda request, addr: log.exception(
        "request from %s failed", addr)
    httpd.token = token or None
    return httpd


if __name__ == "__main__":
    applog.setup()
    n = len(library.list_all())
    print(f"AnyRAG on http://{config.HOST}:{config.PORT}  "
          f"({n} collection(s), {config.LLM_BACKEND}/{config.LLM_MODEL})")
    make_server().serve_forever()
