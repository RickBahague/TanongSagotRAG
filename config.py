"""AnyRAG global configuration: chunking/retrieval tuning + LLM backend.

Corpus-specific settings (name, description, author, persona, per-source
metadata) live on each COLLECTION — see library.py. Values here can be
overridden per-user via settings.json in the user data dir (desktop builds:
the Settings UI writes it); see _apply_settings() at the bottom.
"""
import json
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
# unchanged. Measured on the fixture corpus, single-topic fixture
# collections and writings/IBON (plan.md "M2 eval"; tiny corpora — re-run
# on a real collection): on-topic top-1 cos >= 0.6118, off-topic <= 0.6383
# (cross-topic science questions), off-topic that BM25 lets through
# <= 0.5263, on-topic that BM25 rejects 0.8046.
MIN_DENSE_SIM = 0.57      # cosine floor when BM25 agrees (BM25 FA veto)
DENSE_SOLO_SIM = 0.72     # cosine at which dense alone passes the gate
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


def _apply_settings():
    """Override module constants from user settings.json."""
    allowed = {"LLM_BACKEND", "LLM_MODEL", "LLM_URL", "TEMPERATURE", "LLM_CTX",
               "LLM_THREADS", "RETRIEVAL", "TOP_K", "HOST", "PORT",
               "LLM_MODEL_PATH", "EMBED_MODEL", "CHUNK_WORDS",
               "CHUNK_OVERLAP_WORDS", "SUMMARY_WORKERS", "RERANK",
               "MIN_DENSE_SIM", "DENSE_SOLO_SIM"}
    try:
        overrides = json.loads(SETTINGS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    for k, v in overrides.items():
        if k in allowed:
            globals()[k] = Path(v) if k == "LLM_MODEL_PATH" else v


_apply_settings()
