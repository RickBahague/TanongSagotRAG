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


def load_eval():
    return [json.loads(l) for l in open(FIXTURES / "eval.jsonl") if l.strip()]
