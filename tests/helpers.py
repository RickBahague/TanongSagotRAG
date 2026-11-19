"""Constants and helpers shared by tests (import AFTER conftest has set
ANYRAG_HOME — pytest guarantees that for test modules)."""
import json
from pathlib import Path

import library

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
CORPUS = FIXTURES / "corpus"
CORPUS_FILES = ("astronomy.pdf", "beekeeping.md", "glaciers.docx",
                "lighthouses.html", "report.md", "report.txt", "tiny.txt",
                "volcanoes.txt")
# the "main" corpus: one file per topic, no deliberate defects
CLEAN_FILES = ("astronomy.pdf", "beekeeping.md", "glaciers.docx",
               "lighthouses.html", "volcanoes.txt")
UNREACHABLE_LLM = "http://127.0.0.1:9/v1"     # discard port: refused fast


def reingest(col, **kw):
    """Re-run the pipeline on a fresh Collection object; returns events."""
    import ingest
    events = []
    ingest.run_pipeline(library.get(col.id), summaries=False, workers=1,
                        progress=events.append, **kw)
    return events


# Known BM25-only false rejects on the fixtures (tests/test_gate.py): with
# a dense signal the gate accepts them. A test that compares gate verdicts
# with the eval labels skips these when the index has no dense part.
BM25_KNOWN_FALSE_REJECTS = {"What can ice cores tell us about past climate?"}


def has_dense(index):
    """True when retrieval has a dense signal (encoder + vectors), i.e. not
    the BM25-only install (requirements.txt without the optional deps)."""
    import rag
    return isinstance(index, rag.DenseIndex) or \
        getattr(index, "dense", None) is not None


def gate_expected(index, q):
    """The eval label as the gate should see it on this index."""
    if not has_dense(index) and q["q"] in BM25_KNOWN_FALSE_REJECTS:
        return False
    return bool(q["expect"])


def load_eval():
    return [json.loads(l) for l in open(FIXTURES / "eval.jsonl") if l.strip()]
