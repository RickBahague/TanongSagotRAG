#!/usr/bin/env python3
"""AnyRAG CLI.

  python ask.py -c my-docs "What does the report say about X?"
  python ask.py -c essays --mode quotes "What is said about the sea?"
  python ask.py -c essays --chat
  python ask.py -c essays --quote [--topic X]
  python ask.py --list                    # show collections
"""
import argparse
import sys
import urllib.error

import config
import library
import rag


def print_result(res, mode, rt):
    """Stream the answer to stdout as tokens arrive, then list sources.
    Returns (answer_text, verify_results, citations) for transcript
    saving (citations: rag.check_citations output, scholar/persona only)."""
    print()
    if res.get("rewritten_question"):
        print(f"(searched for: {res['rewritten_question']})\n")
    full = []
    verify = citations = None
    if res["stream"] is None:
        print(res["answer"])
        full = [res["answer"]]
    else:
        if res["mode"] == "persona":
            print(rag.persona_disclaimer(rt.col), end="")
        for t in res["stream"]:
            full.append(t)
            print(t, end="", flush=True)
        print()
    if res["mode"] == "quotes" and full and res["sources"]:
        chunks = [rt.by_id[s["id"]] for s in res["sources"]
                  if s["id"] in rt.by_id]
        verify = rag.verify_quotes("".join(full), chunks,
                                   rt.col.setting("QUOTE_MATCH_RATIO"))
        print("\nQuote check:")
        for r in verify:
            mark = "✓" if r["ok"] else "⚠ NOT FOUND VERBATIM"
            print(f"  {mark}  “{r['text'][:70]}…”" if len(r["text"]) > 70
                  else f"  {mark}  “{r['text']}”")
    numbered = res["mode"] != "quotes"
    if numbered and res["stream"] is not None:
        citations = rag.check_citations("".join(full), len(res["sources"]))
        if citations["out_of_range"]:
            bad = ", ".join(f"[{n}]" for n in citations["out_of_range"])
            print(f"\n⚠ Citation check: {bad} do not match any of the "
                  f"{len(res['sources'])} excerpts sent.")
    if res["sources"]:
        print("\nSources:")
        for i, s in enumerate(res["sources"], start=1):
            pages = (f"p. {s['page_start']}" if s["page_start"] == s["page_end"]
                     else f"pp. {s['page_start']}-{s['page_end']}")
            where = (s["essay"] if s["essay"] == s["work"]
                     else f"{s['essay']} — {s['work']}")
            unused = citations and i in citations["unused"]
            num = f"[{i}] " if numbered else "- "
            print(f"  {num}{where}, {pages}"
                  + ("  (not cited)" if unused else ""))
    print()
    answer = "".join(full)
    if numbered and res["stream"] is not None:
        # the saved transcript form (the stream above stays as generated)
        answer = rag.compose_answer(answer, res.get("excerpts"))
    return answer, verify, citations


def main():
    p = argparse.ArgumentParser(description="Ask questions about a collection.")
    p.add_argument("question", nargs="?", help="the question")
    p.add_argument("--collection", "-c", help="collection id")
    p.add_argument("--list", action="store_true", help="list collections")
    p.add_argument("--mode", choices=["scholar", "persona", "quotes"],
                   default="scholar")
    p.add_argument("-k", type=int, default=None, help="passages to retrieve")
    p.add_argument("--chat", action="store_true", help="interactive loop")
    p.add_argument("--save", metavar="FILE",
                   help="write the transcript to FILE on exit (.md or .json)")
    p.add_argument("--quote", action="store_true",
                   help="print a random verbatim quote and exit")
    p.add_argument("--topic", help="with --quote: quote about this topic")
    args = p.parse_args()

    if args.list:
        cols = library.list_all()
        if not cols:
            print("No collections yet. Create one:\n"
                  '  python -c "import library; '
                  "library.create('My docs', link='/path/to/folder')\"")
        for c in cols:
            print(f"  {c.id:24s} {c.name}  ({len(c.source_files())} sources)")
        return

    if not args.collection:
        p.error("--collection is required (see --list)")
    try:
        col = library.get(args.collection)
    except KeyError:
        ids = ", ".join(c.id for c in library.list_all()) or "(none yet)"
        sys.exit(f"Unknown collection {args.collection!r}. Available: {ids}")

    if args.quote:
        rt = rag.Runtime(col)
        q = rt.random_quote(topic=args.topic)
        if not q:
            sys.exit(f"Quote bank empty — run: python ingest.py -c {col.id}")
        print(f"\n“{q['text']}”\n")
        year = f" ({q['year']})" if q.get("year") else ""
        print(f"    — {q['essay']}{year}, {q['work']}, p. {q['page']}\n")
        return

    if not args.question and not args.chat:
        p.error("give a question or use --chat")

    rt = rag.Runtime(col)
    turns = []

    def run(q):
        try:
            # --chat: in-memory follow-up memory (same rules as /ask)
            history = turns[-int(col.setting("FOLLOWUP_TURNS", 3)):] \
                if turns and col.setting("FOLLOWUP_TURNS", 3) else []
            res = rt.ask_stream(q, mode=args.mode, k=args.k, history=history)
            answer, verify, citations = print_result(res, args.mode, rt)
            turns.append({"ts": "", "question": q, "mode": res["mode"],
                          "answer": answer, "sources": res["sources"],
                          "verify": verify, "citations": citations,
                          "rewritten_question":
                              res.get("rewritten_question")})
        except urllib.error.URLError:
            sys.exit(f"Cannot reach {config.LLM_BACKEND} backend at "
                     f"{config.LLM_URL} (model {config.LLM_MODEL}). Check "
                     f"config.py and that the server is running.")
        except RuntimeError as e:
            sys.exit(str(e))

    if args.chat:
        print(f"AnyRAG chat — {col.name} ({args.mode} mode). "
              f"Empty line to quit.")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run(q)
    else:
        run(args.question)

    if args.save and turns:
        import json as _json
        import conversations
        from datetime import datetime, timezone
        for t in turns:
            t["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if args.save.endswith(".json"):
            content = _json.dumps(turns, ensure_ascii=False, indent=1)
        else:
            content = conversations.to_markdown(turns, f"{col.name} conversation")
        with open(args.save, "w") as f:
            f.write(content)
        print(f"Transcript saved to {args.save}")


if __name__ == "__main__":
    import errors
    try:
        main()
    except errors.NotIndexed as e:     # collection not ingested yet
        sys.exit(str(e))
