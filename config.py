"""AnyRAG global configuration: chunking/retrieval tuning + LLM backend.

Corpus-specific settings (name, description, author, persona, per-source
metadata) live on each COLLECTION — see library.py. Values here can be
overridden per-user via settings.json in the user data dir (desktop builds:
the Settings UI writes it); see _apply_settings() at the bottom.
"""
import json
import re
import threading
from pathlib import Path

import paths

ROOT = paths.bundle_dir()             # read-only in frozen builds
USER_ROOT = paths.user_data_dir()     # writable; ./devdata in dev
SETTINGS_FILE = USER_ROOT / "settings.json"

# Chunking (stage 1 — always runs first for every new source)
CHUNK_WORDS = 450         # ~ 600 tokens
CHUNK_OVERLAP_WORDS = 80

# Embedding (stage 2 — optional; runs only on already-chunked sources).
# Requires: pip install sentence-transformers numpy   (see requirements.txt)
# bge models need a query-side instruction prefix (applied to queries only).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_QUERY_PREFIX = ("Represent this sentence for searching relevant passages: "
                      if "bge" in EMBED_MODEL.lower() else "")

# Summaries (stage 1b — generated like chunks, BEFORE embedding)
SUMMARY_MAX_WORDS = 3000      # map-reduce window for long sections
SUMMARY_MIN_WORDS = 120       # skip trivial fragments

# Parallel ingestion
INGEST_WORKERS = 0            # chunk-stage processes; 0 = cpu_count()//2
SUMMARY_WORKERS = 4           # concurrent LLM calls in stage 1b
                              # (Ollama: set env OLLAMA_NUM_PARALLEL >= this;
                              #  llama.cpp server: -np; embedded backend → 1)
EMBED_BATCH = 64              # sentence-transformers encode batch size

# Hierarchical (summaries-first) retrieval
HIERARCHICAL = "auto"         # "auto" (on when summaries exist) | "on" | "off"
SUMMARY_TOP_M = 8             # sections selected in the explore stage
RESTRICTED_WEIGHT = 2.0       # boost for chunks inside selected sections

# Optional cross-encoder reranker over the final candidate pool
RERANK = "auto"               # "auto" (on if model importable) | "off"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
# RETRIEVAL: "auto" = hybrid when vectors + libs exist, else BM25.
#            Force with "bm25", "dense", or "hybrid".
RETRIEVAL = "auto"
TOP_K = 10
BM25_K1 = 1.5
BM25_B = 0.75
MIN_SCORE = 5.0           # top-hit BM25 score below this → "not covered"
MIN_COVERAGE = 0.5        # fraction of query terms present in a top hit
# Dense gate (raw cosine of the top chunk; bge-small, normalized vectors).
# Unrelated text already sits at ~0.30-0.64 cosine with bge, so the dense
# signal never passes on its own near those values:
#   accept  if  cos >= DENSE_SOLO_SIM                       (dense alone)
#           or  (BM25 signal passes  and  cos >= MIN_DENSE_SIM)
# BM25-only indexes (no vectors / no encoder) use the BM25 signal alone,
# unchanged. Set on 2026-11-19 from 234 labelled questions (plan.md "Gate
# thresholds on JMS"): the real JMS collection (4,257 chunks, 63 on / 48
# off-topic, hard negatives included), writings/IBON and the fixtures, with
# cross-collection negatives. On-topic top-1 cos >= 0.614; off-topic
# <= 0.661. On a large corpus BM25 passes many off-topic questions, so the
# floor that BM25 needs (MIN_DENSE_SIM) matters most: 0.57 let 11 of 48 JMS
# negatives through, 0.60 lets 3.
MIN_DENSE_SIM = 0.60      # cosine floor when BM25 agrees (BM25 FA veto)
DENSE_SOLO_SIM = 0.70     # cosine at which dense alone passes the gate
RRF_K = 60                # reciprocal-rank-fusion constant (hybrid)
HYBRID_CANDIDATES = 30    # candidates taken from each retriever before fusion

# Source diversity: one large document must not crowd the rest of a
# collection out of top-k. Retrieval pulls a deeper pool, then caps chunks
# per source, backfilling by rank if underfull.
MAX_PER_WORK = 6         # max chunks from one source in the final top-k
POOL_FACTOR = 6          # candidate pool = k * POOL_FACTOR before the cap

# Quotes mode
QUOTES_TOP_K = 10         # quote curation needs more raw material
QUOTES_TEMPERATURE = 0.1  # and zero creativity
QUOTE_MATCH_RATIO = 0.90  # fuzzy-verification floor (difflib ratio)

# Follow-ups (P2-1): recent turns of a conversation feed a one-call
# standalone-question rewrite (only when rag.is_followup() says the question
# depends on them) and the prompt, inside the LLM_CTX budget.
FOLLOWUP_TURNS = 3        # recent turns loaded per follow-up
HISTORY_ANSWER_WORDS = 150  # each past question/answer clipped to this
REWRITE_MAX_TOKENS = 64   # cap for the rewrite call (temperature 0)

# Every retrieval/gate/generation value above can be overridden PER
# COLLECTION in collections/<id>/settings.json (P2-8; read through
# Collection.setting(), falling back to these defaults).

# ---------------------------------------------------------------------------
# Generation backend — configurable.
#
# LLM_BACKEND options:
#   "ollama"   → Ollama's native API      (default; http://localhost:11434)
#   "openai"   → any OpenAI-compatible local server: llama.cpp server,
#                LM Studio, llamafile, vLLM, etc.  (set LLM_URL accordingly)
#   "embedded" → llama-cpp-python loading a bundled GGUF in-process
#                (desktop app default; set LLM_MODEL_PATH)
#
# Tiny-model options (great on low-RAM machines):
#   via Ollama:      qwen2.5:1.5b · gemma2:2b · phi3:mini · llama3.2:1b
#   via llama.cpp /  any GGUF of the above, e.g.
#   LM Studio:       Qwen2.5-1.5B-Instruct-GGUF, Phi-3-mini-4k-instruct-GGUF
# ---------------------------------------------------------------------------
LLM_BACKEND = "ollama"                    # "ollama" | "openai" | "embedded"
LLM_MODEL = "llama3.1:8b"
LLM_URL = "http://localhost:11434"        # ollama default; use e.g.
                                          # "http://localhost:8080/v1" for llama.cpp,
                                          # "http://localhost:1234/v1" for LM Studio
TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024                     # generation cap (runaway guard)

# Embedded backend (llama-cpp-python)
LLM_MODEL_PATH = ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_CTX = 8192                            # context window for embedded model
LLM_THREADS = 0                           # 0 = llama.cpp auto

# ONNX query encoder (desktop builds; falls back to sentence-transformers)
ONNX_ENCODER_DIR = ROOT / "models" / "encoder-onnx"

# Web UI
HOST = "127.0.0.1"
PORT = 8000


# Source scanning (P3-2): linked (and managed) source folders are walked
# RECURSIVELY; any path component matching one of these fnmatch patterns
# (case-insensitive) is skipped — the file itself, or a whole directory.
# Hidden names (".git", ".DS_Store", ".Trash", …) are covered by ".*".
SOURCE_IGNORE = (".*", "~$*", "~*.tmp", "*.tmp", "*.part", "*.crdownload",
                 "node_modules", "__pycache__", "venv", "Thumbs.db",
                 "desktop.ini", "$RECYCLE.BIN", "System Volume Information",
                 "@eaDir", "#recycle", "lost+found")
SOURCE_MAX_DEPTH = 12     # folder levels below the linked folder


# ---------------------------------------------------------------------------
# Settings (P3-1). ONE allow-list for both the user settings.json (global,
# USER_ROOT/settings.json) and per-collection settings.json files.
#
#   type     "enum" | "int" | "float" | "str" | "path"
#   scope    "global"     — only in the user settings.json
#            "collection" — also overridable per collection (read through
#                           Collection.setting(), falling back to the global
#                           value at call time)
#   min/max  inclusive range (int/float); maxlen for str/path
#   choices  allowed values (enum)
#   effect   "live" (next request), "reingest" (applies on the next ingest),
#            "restart" (read at startup only)
# Every module reads these values at CALL time through get() (or
# Collection.setting()), so a saved change applies without a restart.
# ---------------------------------------------------------------------------

def _s(type_, scope="collection", effect="live", help="", **kw):
    return {"type": type_, "scope": scope, "effect": effect, "help": help,
            **kw}


SETTINGS_SCHEMA = {
    # generation backend
    "LLM_BACKEND": _s("enum", "global", choices=["ollama", "openai",
                                                 "embedded"],
                      help="ollama (native API), openai (any OpenAI-"
                           "compatible local server) or embedded (GGUF)"),
    "LLM_MODEL": _s("str", "global", maxlen=200,
                    help="model name on the backend"),
    "LLM_URL": _s("str", "global", maxlen=500, url=True,
                  help="backend base URL (http://…)"),
    "LLM_MODEL_PATH": _s("path", "global", maxlen=1000,
                         help="GGUF file for the embedded backend"),
    "LLM_CTX": _s("int", "global", min=512, max=262144,
                  help="model context window (tokens)"),
    "LLM_MAX_TOKENS": _s("int", "global", min=16, max=32768,
                         help="generation cap (tokens)"),
    "LLM_THREADS": _s("int", "global", min=0, max=256,
                      help="embedded backend threads (0 = auto)"),
    "SUMMARY_WORKERS": _s("int", "global", min=1, max=32,
                          help="concurrent LLM calls while summarizing"),
    "EMBED_MODEL": _s("str", "global", "reingest", maxlen=200,
                      help="sentence-transformers embedding model"),
    "HOST": _s("enum", "global", "restart", choices=["127.0.0.1",
                                                     "localhost"]),
    "PORT": _s("int", "global", "restart", min=0, max=65535),
    # generation
    "TEMPERATURE": _s("float", min=0.0, max=2.0),
    # retrieval
    "RETRIEVAL": _s("enum", choices=["auto", "bm25", "dense", "hybrid"]),
    "TOP_K": _s("int", min=1, max=100, help="excerpts sent to the model"),
    "MIN_SCORE": _s("float", min=0.0, max=100.0,
                    help="BM25 relevance-gate floor"),
    "MIN_COVERAGE": _s("float", min=0.0, max=1.0,
                       help="fraction of query terms in a top hit"),
    "MIN_DENSE_SIM": _s("float", min=0.0, max=1.0),
    "DENSE_SOLO_SIM": _s("float", min=0.0, max=1.0),
    "MAX_PER_WORK": _s("int", min=1, max=100,
                       help="max excerpts from one source"),
    "POOL_FACTOR": _s("int", min=1, max=50),
    "HIERARCHICAL": _s("enum", choices=["auto", "on", "off"]),
    "RERANK": _s("enum", choices=["auto", "off"]),
    # quotes mode
    "QUOTES_TOP_K": _s("int", min=1, max=100),
    "QUOTES_TEMPERATURE": _s("float", min=0.0, max=2.0),
    "QUOTE_MATCH_RATIO": _s("float", min=0.5, max=1.0),
    # chunking
    "CHUNK_WORDS": _s("int", effect="reingest", min=50, max=5000),
    "CHUNK_OVERLAP_WORDS": _s("int", effect="reingest", min=0, max=2000),
    # follow-ups (P2-1)
    "FOLLOWUP_TURNS": _s("int", min=0, max=20),
    "HISTORY_ANSWER_WORDS": _s("int", min=0, max=2000),
    "REWRITE_MAX_TOKENS": _s("int", min=8, max=1024),
}

_PATH_KEYS = {k for k, v in SETTINGS_SCHEMA.items() if v["type"] == "path"}

# key -> (value applied from settings.json, value it replaced). Lets a
# removed override restore whatever the value was before it (tests
# monkeypatch module values; those are never clobbered).
_APPLIED = {}
SETTINGS_LOCK = threading.RLock()   # serializes settings read-merge-write


def get(key, default=None):
    """THE accessor for a global setting: the current module value, which
    reload_settings() keeps in sync with settings.json."""
    return globals().get(key, default)


def check_value(key, value):
    """Error message for one (key, value) against SETTINGS_SCHEMA, or None.
    `value` None is always valid (it removes an override)."""
    spec = SETTINGS_SCHEMA.get(key)
    if spec is None:
        return "unknown setting"
    if value is None:
        return None
    t = spec["type"]
    if t == "enum":
        if value not in spec["choices"]:
            return "must be one of " + ", ".join(map(str, spec["choices"]))
        return None
    if t == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return "must be an integer"
    elif t == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or value != value:
            return "must be a number"
    else:                                    # str / path
        if not isinstance(value, str) or not value.strip():
            return "must be a non-empty string"
        if len(value) > spec.get("maxlen", 1000):
            return f"must be at most {spec.get('maxlen', 1000)} characters"
        if spec.get("url") and not re.match(r"^https?://[^\s/]+", value):
            return "must be an http:// or https:// URL"
        return None
    lo, hi = spec.get("min"), spec.get("max")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return f"must be between {lo} and {hi}"
    return None


def _cross_errors(eff):
    """Constraints between settings, on the EFFECTIVE values."""
    errs = {}
    if eff.get("CHUNK_OVERLAP_WORDS", 0) >= eff.get("CHUNK_WORDS", 1):
        errs["CHUNK_OVERLAP_WORDS"] = "must be smaller than CHUNK_WORDS"
    if eff.get("LLM_MAX_TOKENS", 0) >= eff.get("LLM_CTX", 1):
        errs["LLM_MAX_TOKENS"] = "must be smaller than LLM_CTX"
    return errs


def validate_settings(changes, scope="global", base=None, fallback=None):
    """Validate a {key: value|None} change set. scope "global" accepts every
    key; scope "collection" only keys whose schema scope is "collection".
    `base` = the effective values the changes apply on top of and
    `fallback` the values a removed (None) override falls back to (both
    for the cross-field checks). Returns (clean, errors): `errors` maps a key to a
    message; nothing may be written when it is non-empty."""
    if not isinstance(changes, dict):
        return {}, {"_": "body must be a JSON object of {setting: value}"}
    clean, errors = {}, {}
    for k, v in changes.items():
        msg = check_value(k, v)
        if msg is None and scope == "collection" \
                and SETTINGS_SCHEMA[k]["scope"] != "collection":
            msg = "global only — set it in the app settings, not per collection"
        if msg:
            errors[k] = msg
            continue
        if v is not None and SETTINGS_SCHEMA[k]["type"] == "float":
            v = float(v)
        clean[k] = v
    if not errors:
        eff = dict(base or {})
        for k, v in clean.items():
            if v is None:
                if fallback and k in fallback:
                    eff[k] = fallback[k]
                else:
                    eff.pop(k, None)
            else:
                eff[k] = v
        errors.update({k: m for k, m in _cross_errors(eff).items()
                       if k in clean or k in eff})
    return clean, errors


def read_settings_file(path):
    """{key: value} from a settings.json ({} if absent or unreadable)."""
    try:
        data = json.loads(Path(path).read_text())
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def user_overrides():
    """Valid overrides in the user settings.json (invalid entries ignored)."""
    return {k: v for k, v in read_settings_file(SETTINGS_FILE).items()
            if k in SETTINGS_SCHEMA and v is not None
            and check_value(k, v) is None}


def default_value(key):
    """Value of `key` without the user settings.json override."""
    if key in _APPLIED and globals().get(key) == _APPLIED[key][0]:
        return _APPLIED[key][1]
    return globals().get(key)


def reload_settings():
    """(Re)apply the user settings.json to the module values. Overrides that
    were removed from the file restore the value they had replaced. Called
    at import and after every settings write, so changes apply without a
    restart."""
    global EMBED_QUERY_PREFIX
    overrides = user_overrides()
    for k in list(_APPLIED):
        if k not in overrides:
            applied, prev = _APPLIED.pop(k)
            if globals().get(k) == applied:      # nobody else changed it
                globals()[k] = prev
    for k, v in overrides.items():
        v = Path(v) if k in _PATH_KEYS else v
        cur = globals().get(k)
        if k in _APPLIED and cur == _APPLIED[k][0]:
            prev = _APPLIED[k][1]
        else:
            prev = cur
        _APPLIED[k] = (v, prev)
        globals()[k] = v
    EMBED_QUERY_PREFIX = ("Represent this sentence for searching relevant "
                          "passages: " if "bge" in str(EMBED_MODEL).lower()
                          else "")


def save_user_settings(changes):
    """Validate and merge a change set into the user settings.json (None
    removes a key), then reload. Returns {} on success, else {key: error}
    and nothing is written."""
    with SETTINGS_LOCK:
        return _save_user_settings(changes)


def _save_user_settings(changes):
    defaults = {k: default_value(k) for k in SETTINGS_SCHEMA}
    base = dict(defaults, **user_overrides())
    clean, errors = validate_settings(changes, "global", base, defaults)
    if errors:
        return errors
    data = read_settings_file(SETTINGS_FILE)
    for k, v in clean.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    import library                     # lazy: library imports config
    library.write_json_atomic(SETTINGS_FILE, data, indent=2)
    reload_settings()
    return {}


def describe_settings(effective, overrides, defaults, scope="global"):
    """GET /settings shape: {key: {value, default, overridden, type, scope,
    effect, help, choices?/min?/max?/maxlen?}} for the keys of `scope`."""
    out = {}
    for k, spec in SETTINGS_SCHEMA.items():
        if scope == "collection" and spec["scope"] != "collection":
            continue
        val = effective[k]
        d = defaults[k]
        out[k] = {"value": str(val) if isinstance(val, Path) else val,
                  "default": str(d) if isinstance(d, Path) else d,
                  "overridden": k in overrides, **spec}
    return out


_apply_settings = reload_settings      # older name (tests, callers)
reload_settings()
