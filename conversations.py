"""Conversation persistence (design.md "Saving conversations" — Option A
storage + Option B export). Each conversation is a JSONL file of turns:

  {ts, question, mode, answer, sources, verify}

Stored per collection (data/conversations/), fully isolated between
collections. The schema deliberately carries everything Option C
(follow-up memory) will need later.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def new_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _path(col, cid):
    cid = re.sub(r"[^\w.-]", "", cid)[:40]
    if not cid:
        raise ValueError("bad conversation id")
    return col.conversations_dir / (cid + ".jsonl")


def append_turn(col, cid, question, mode, answer, sources, verify=None):
    col.conversations_dir.mkdir(parents=True, exist_ok=True)
    with open(_path(col, cid), "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question, "mode": mode, "answer": answer,
            "sources": sources or [], "verify": verify,
        }, ensure_ascii=False) + "\n")


def load(col, cid):
    p = _path(col, cid)
    if not p.exists():
        raise KeyError(cid)
    return [json.loads(l) for l in open(p) if l.strip()]


def delete(col, cid):
    _path(col, cid).unlink(missing_ok=True)


def list_all(col):
    """[{id, title, date, turns}] newest first."""
    if not col.conversations_dir.exists():
        return []
    out = []
    for f in sorted(col.conversations_dir.glob("*.jsonl"), reverse=True):
        try:
            lines = [l for l in open(f) if l.strip()]
            first = json.loads(lines[0])
        except Exception:
            continue
        out.append({"id": f.stem,
                    "title": first["question"][:80],
                    "date": first.get("ts", ""),
                    "turns": len(lines)})
    return out


def _md_sources(sources):
    lines = []
    for s in sources:
        pg = (f"p. {s['page_start']}" if s["page_start"] == s["page_end"]
              else f"pp. {s['page_start']}–{s['page_end']}")
        work = s.get("work") or Path(s.get("source", "")).stem
        where = s["essay"] if s["essay"] == work else f"{s['essay']} — {work}"
        lines.append(f"- {where}, {pg}")
    return "\n".join(lines)


def to_markdown(turns, title="AnyRAG conversation"):
    """Readable transcript: Q/A headings, blockquotes preserved, quote-check
    marks, source lists."""
    date = (turns[0].get("ts", "") if turns else "")[:10]
    parts = [f"# {title} — {date}", ""]
    for t in turns:
        parts.append(f"## {t['question']}")
        parts.append(f"*mode: {t['mode']}*")
        parts.append("")
        answer = t["answer"] or ""
        if t["mode"] == "persona":
            answer = ("*[AI reconstruction from published writings — "
                      "not the author himself.]*\n\n") + \
                     re.sub(r"^\*\[[^\]]*\]\*\s*", "", answer)
        parts.append(answer.strip())
        if t.get("verify"):
            parts.append("")
            parts.append("Quote check:")
            for q in t["verify"]:
                mark = "✓" if q["ok"] else "⚠ not found verbatim"
                parts.append(f"- {mark}: “{q['text'][:80]}”")
        if t.get("sources"):
            parts.append("")
            parts.append("Sources:")
            parts.append(_md_sources(t["sources"]))
        parts.append("")
    return "\n".join(parts).strip() + "\n"
