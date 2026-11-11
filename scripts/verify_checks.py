#!/usr/bin/env python3
"""Milestone checks for scripts/verify.sh (see implem-plan-enhancements.md).

  verify_checks.py <milestone> <junit.xml> <eval.json>

Prints one line per check ("ok …" / "FAIL …") and exits 1 if any failed.
Required tests are matched by base name (parametrized ids included: every
variant must pass). "Pass" means it ran and passed — an xfail, skip, failure
or a missing test all count as not passing.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINES = REPO / "tests" / "fixtures" / "baselines"

# xfail tests M0 must create (implem-plan-enhancements.md, M0 table + the
# chunk-tail defect found while building the fixtures)
M0_XFAIL = [
    "test_prune_removed_source", "test_same_stem_sources_both_indexed",
    "test_context_fits_budget", "test_max_tokens_sent_openai",
    "test_embedded_generation_serialized",
    "test_unindexed_collection_returns_409",
    "test_short_document_is_chunked", "test_chunk_tail_not_dropped",
    "test_duplicate_section_titles_kept",
    "test_rechunk_on_content_change_not_mtime", "test_runtime_cache_is_lru",
    "test_json_writes_atomic", "test_no_askj_leftovers",
    "test_foreign_host_rejected", "test_foreign_origin_rejected",
    "test_post_requires_json_content_type", "test_oversized_body_rejected",
    "test_embed_stage_uses_onnx_encoder",
]

REQUIRED_PASS = {
    "M1": [t for t in M0_XFAIL if t != "test_embed_stage_uses_onnx_encoder"]
          + ["test_stem_keyed_collection_migrates"],
    "M2": ["test_embed_stage_uses_onnx_encoder",
           "test_vectors_stale_on_pooling_change", "test_hybrid_index_loads",
           "test_dense_gate_rejects_off_topic"],
}

IBON = REPO / "writings" / "IBON"
IBON_CHUNKS = 14                      # count before M1 (plan-enhancements.md)


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            self.failed += 1
        return ok


def junit_outcomes(path):
    """{base_name: [outcome, …]} with outcome in pass/xfail/skip/fail."""
    out = {}
    for tc in ET.parse(path).iter("testcase"):
        base = re.sub(r"\[.*\]$", "", tc.get("name", ""))
        kids = {c.tag: c for c in tc}
        if "failure" in kids or "error" in kids:
            o = "fail"
        elif "skipped" in kids:
            s = kids["skipped"]
            o = "xfail" if "xfail" in (s.get("type", "") +
                                       s.get("message", "")) else "skip"
        else:
            o = "pass"
        out.setdefault(base, []).append(o)
    return out


def latest_baseline(before):
    order = ["M0", "M2", "M3", "M4"]
    for m in reversed(order[:order.index(before)] if before in order
                      else order):
        f = BASELINES / f"{m}.json"
        if f.exists():
            return m, json.loads(f.read_text())
    return None, None


def no_regression(c, cur, label):
    m, base = latest_baseline(label)
    if not c.check(base is not None, f"a baseline exists before {label}"):
        return
    for k in ("hit@10", "mrr"):
        c.check(cur[k] >= base[k] - 1e-9,
                f"{k} {cur[k]} >= {m} baseline {base[k]}")
    c.check(cur["gate_false_accepts"] <= base["gate_false_accepts"],
            f"gate false-accepts {cur['gate_false_accepts']} <= {m} "
            f"baseline {base['gate_false_accepts']}")


def ui_script_check(c):
    node = shutil.which("node")
    if not node:
        print("  note  node not found — static/index.html NOT machine-checked")
        return
    html = (REPO / "static" / "index.html").read_text()
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write("\n;\n".join(scripts))
    r = subprocess.run([node, "--check", f.name], capture_output=True,
                       text=True)
    c.check(r.returncode == 0, "static/index.html inline script parses"
            + ("" if r.returncode == 0 else ": " + r.stderr.strip()[:300]))


def ibon_check(c):
    """Linked read-only ingest of writings/IBON into a scratch home."""
    if not c.check(IBON.is_dir(), f"{IBON.relative_to(REPO)} present"):
        return
    code = f"""
import os, sys, tempfile
os.environ["ANYRAG_HOME"] = tempfile.mkdtemp(prefix="anyrag-ibon-")
sys.path.insert(0, {str(REPO)!r})
import library, ingest, rag
col = library.create("IBON check", link={str(IBON)!r})
ingest.run_pipeline(col, summaries=False, workers=1, progress=lambda e: None)
idx = rag.load_index(library.get(col.id))
print("RESULT", len(idx.chunks),
      rag.gate_ok(idx, "What does IBON say about EPIRA and power prices?"))
"""
    r = subprocess.run([str(REPO / ".venv/bin/python"), "-c", code],
                       capture_output=True, text=True, cwd=REPO)
    m = re.search(r"RESULT (\d+) (True|False)", r.stdout)
    if not c.check(bool(m), "IBON scratch ingest ran"
                   + ("" if m else ": " + r.stderr.strip()[-300:])):
        return
    c.check(int(m.group(1)) == IBON_CHUNKS,
            f"IBON chunk count {m.group(1)} == {IBON_CHUNKS}")
    c.check(m.group(2) == "True", "IBON sample question passes the gate")


def main():
    milestone, junit, eval_json = sys.argv[1], sys.argv[2], sys.argv[3]
    c = Checks()
    outcomes = junit_outcomes(junit)
    cur = json.loads(Path(eval_json).read_text())

    if milestone == "M0":
        for t in M0_XFAIL:
            got = outcomes.get(t)
            c.check(bool(got) and all(o == "xfail" for o in got),
                    f"{t} present and xfail ({got or 'missing'})")
        c.check(cur["questions_on_topic"] == 15
                and cur["questions_off_topic"] == 5,
                "fixture eval set: 15 on-topic + 5 off-topic")
        print(f"  info  fixture baseline: {cur['retrieval']} "
              f"hit@1={cur['hit@1']:.3f} hit@10={cur['hit@10']:.3f} "
              f"mrr={cur['mrr']} false_accepts={cur['gate_false_accepts']} "
              f"false_rejects={cur['gate_false_rejects']}")
    elif milestone in REQUIRED_PASS:
        for t in REQUIRED_PASS[milestone]:
            got = outcomes.get(t)
            c.check(bool(got) and all(o == "pass" for o in got),
                    f"{t} passes ({got or 'missing'})")
        if milestone == "M1":
            ibon_check(c)
            no_regression(c, cur, "M1")
        if milestone == "M2":
            c.check(cur["retrieval"] != "BM25Index",
                    f"fixture index is dense-backed ({cur['retrieval']})")
            keys = set(cur.get("vector_keys") or [])
            need = {"vectors", "model", "pooling", "max_len", "src_hash"}
            c.check(cur.get("vector_files", 0) > 0 and need <= keys,
                    f"vector files tagged {sorted(need)} (got {sorted(keys)})")
            no_regression(c, cur, "M2")
            c.check("M2 eval" in (REPO / "plan.md").read_text(),
                    'plan.md has an "M2 eval" section')
    elif milestone in ("M3", "M4"):
        no_regression(c, cur, milestone)
        ui_script_check(c)
    elif re.fullmatch(r"M5-\d+", milestone):
        print(f"  info  fixture eval: {json.dumps({k: cur[k] for k in ('hit@1', 'hit@10', 'mrr', 'gate_false_accepts', 'gate_false_rejects')})}")
        c.check(milestone in (REPO / "plan.md").read_text(),
                f"plan.md records {milestone} before/after numbers")
    else:
        c.check(False, f"unknown milestone {milestone!r}")

    if not c.failed and milestone in ("M0", "M2", "M3", "M4"):
        BASELINES.mkdir(parents=True, exist_ok=True)
        (BASELINES / f"{milestone}.json").write_text(
            json.dumps(cur, indent=2) + "\n")
        print(f"  info  baseline written: tests/fixtures/baselines/"
              f"{milestone}.json")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
