"""M3-A: per-collection settings (P2-8), numbered citations (P2-2),
follow-up rewrite + history in the prompt (P2-1)."""
import math
import sys

import config
import conversations
import library
import rag
from helpers import FIXTURES, has_dense

BEES_Q = "How do worker bees build the wax comb?"
ICE_Q = "How does snow turn into glacial ice?"
QUEEN_Q = "What happens when a colony loses its queen?"


def _is_rewrite(body):
    return body["messages"][0]["content"] == rag.REWRITE_SYSTEM


def _answer_requests(fake):
    return [b for b in fake.requests if not _is_rewrite(b)]


def _rewrite_requests(fake):
    return [b for b in fake.requests if _is_rewrite(b)]


def _ask(server, col, question, **kw):
    st, evs = server.ndjson("POST", f"/c/{col.id}/ask",
                            {"question": question, **kw})
    assert st == 200, evs
    return evs


# ------------------------------------------------------------ P2-8 --------

def test_settings_defaults_unchanged(make_collection):
    col = make_collection()
    for key in ("TOP_K", "MIN_SCORE", "MIN_COVERAGE", "MIN_DENSE_SIM",
                "DENSE_SOLO_SIM", "MAX_PER_WORK", "POOL_FACTOR",
                "HIERARCHICAL", "RERANK", "TEMPERATURE", "QUOTES_TOP_K",
                "QUOTES_TEMPERATURE", "QUOTE_MATCH_RATIO"):
        assert col.setting(key) == getattr(config, key)
    index = rag.load_index(col)
    assert index.col is col
    for sub in ("bm25", "dense"):
        if getattr(index, sub, None) is not None:
            assert getattr(index, sub).col is col
    # a hand-built index (no collection) reads config as before
    assert rag._cfg(rag.BM25Index(rag.load_chunks(col)), "TOP_K") == config.TOP_K


def test_settings_per_collection(server, fake_llm, make_collection):
    strict = make_collection("Strict", settings={"MIN_COVERAGE": 1.01,
                                                 "DENSE_SOLO_SIM": 1.01})
    narrow = make_collection("Narrow", settings={"TOP_K": 2,
                                                 "TEMPERATURE": 0.05})
    wide = make_collection("Wide", settings={"TOP_K": 5, "TEMPERATURE": 0.9,
                                             "MAX_PER_WORK": 1})
    plain = make_collection("Plain", settings={"TOP_K": 5})

    # gate: same question, same corpus, one process — different verdicts
    assert not rag.gate_ok(rag.load_index(library.get(strict.id)), BEES_Q)
    assert rag.gate_ok(rag.load_index(library.get(narrow.id)), BEES_Q)

    evs = _ask(server, strict, BEES_Q)
    assert [e["type"] for e in evs] == ["meta", "delta", "done"]
    assert "do not appear to cover" in evs[1]["text"]
    assert fake_llm.requests == []          # gated: no model call

    meta = _ask(server, narrow, BEES_Q)[0]
    assert len(meta["sources"]) == 2
    assert fake_llm.last["temperature"] == 0.05

    meta = _ask(server, wide, BEES_Q)[0]
    assert fake_llm.last["temperature"] == 0.9
    wide_index = rag.load_index(library.get(wide.id))
    if has_dense(wide_index):
        assert len(meta["sources"]) == 5
        assert len({s["source"] for s in meta["sources"]}) == 5  # cap 1/source
    else:
        # BM25-only: only chunks with a query term rank; with one matching
        # source, diversify() backfills past the cap instead of leaving gaps
        hits = wide_index.search(BEES_Q, None)
        assert len(meta["sources"]) == min(5, len(hits)) >= 2

    meta = _ask(server, plain, BEES_Q)[0]
    plain_index = rag.load_index(library.get(plain.id))
    assert len(meta["sources"]) == (5 if has_dense(plain_index) else
                                    min(5, len(plain_index.search(BEES_Q, None))))
    assert fake_llm.last["temperature"] == config.TEMPERATURE
    assert len({s["source"] for s in meta["sources"]}) < 5    # default cap 6


def test_quote_match_ratio_per_collection(server, fake_llm, make_collection):
    sys.path.insert(0, str(FIXTURES))
    from build import QUOTE
    words = QUOTE.split()
    mid = len(words) // 2
    altered = " ".join(words[:mid] + ["purple"] + words[mid + 1:])
    fake_llm.reply = f"Framing.\n\n> {altered} [Hive Structure, p. 1]"
    lax = make_collection("Lax", settings={"QUOTE_MATCH_RATIO": 0.5})
    tight = make_collection("Tight", settings={"QUOTE_MATCH_RATIO": 0.999})
    got = {}
    for col in (lax, tight):
        evs = _ask(server, col, QUEEN_Q, mode="quotes")
        [verify] = [e for e in evs if e["type"] == "verify"]
        assert not [e for e in evs if e["type"] == "citations"]
        got[col.name] = [q["ok"] for q in verify["quotes"]]
        assert fake_llm.last["temperature"] == config.QUOTES_TEMPERATURE
    assert got == {"Lax": [True], "Tight": [False]}


# ------------------------------------------------------------ P2-2 --------

def test_check_citations():
    cc = rag.check_citations
    assert cc("Bees [1] build [3].", 3) == {
        "used": [1, 3], "unused": [2], "out_of_range": []}
    assert cc("A [1, 3] and [2][3].", 4) == {
        "used": [1, 2, 3], "unused": [4], "out_of_range": []}
    assert cc("Range [1-3] and [2–4].", 5) == {
        "used": [1, 2, 3, 4], "unused": [5], "out_of_range": []}
    assert cc("Bad [0] and [7] and [5-7].", 5) == {
        "used": [5], "unused": [1, 2, 3, 4], "out_of_range": [0, 6, 7]}
    # quotes-mode style citations are not numeric markers
    assert cc("> text [Hive Structure, p. 3] [Comets, pp. 2-4]", 2) == {
        "used": [], "unused": [1, 2], "out_of_range": []}
    assert cc("", 0) == {"used": [], "unused": [], "out_of_range": []}
    assert cc("None cited.", 2)["unused"] == [1, 2]


def test_numbered_context_and_prompts(make_collection):
    col = make_collection()
    hits = rag.BM25Index(rag.load_chunks(col)).search(BEES_Q, 3)
    ctx = rag.build_context(hits).split("\n\n---\n\n")
    assert [p.split(" ", 1)[0] for p in ctx] == ["[1]", "[2]", "[3]"]
    plain = rag.build_context(hits, numbered=False)
    assert not plain.startswith("[1] ")
    assert "[n]" not in rag.system_quotes(col)
    assert "[1]" in rag.system_scholar(col)
    assert "[Section Title, p. N]" not in rag.system_scholar(col)
    assert "[Section Title, p. N]" in rag.system_quotes(col)


def test_citations_event(server, fake_llm, make_collection):
    col = make_collection()
    by_id = {c["id"]: c for c in rag.load_chunks(col)}
    fake_llm.reply = "Wax comes from glands [1]. Cells are hexagonal [2][3] [99]."
    evs = _ask(server, col, BEES_Q)
    types = [e["type"] for e in evs]
    assert types[0] == "meta" and types[-2:] == ["citations", "done"]
    assert set(types[1:-2]) == {"delta"}
    meta = evs[0]
    assert meta["rewritten_question"] is None
    n = len(meta["sources"])
    assert n >= 3
    cit = evs[-2]
    assert cit == {"type": "citations", "used": [1, 2, 3],
                   "unused": list(range(4, n + 1)), "out_of_range": [99]}
    # meta.sources is in prompt-numbering order: [i] == sources[i-1]
    user = fake_llm.last["messages"][1]["content"]
    ctx = user.split("Excerpts from the documents:\n\n", 1)[1]
    ctx = ctx.rsplit("\n\nQuestion: ", 1)[0].split("\n\n---\n\n")
    assert len(ctx) == n
    for i, (part, src) in enumerate(zip(ctx, meta["sources"]), start=1):
        assert part.startswith(f"[{i}] ") and not part.startswith(f"[{i}] [")
        assert by_id[src["id"]]["text"] in part
    # stored with the turn
    [turn] = conversations.load(col, meta["conversation_id"])
    assert turn["citations"] == {k: cit[k] for k in
                                 ("used", "unused", "out_of_range")}
    assert turn["rewritten_question"] is None and turn["verify"] is None


def test_persona_mode_citations(server, fake_llm, make_collection):
    col = make_collection("P", author="Ada Example", persona={"enabled": True})
    assert "persona" in rag.available_modes(col)
    fake_llm.reply = "I wrote of wax [1]."
    evs = _ask(server, col, BEES_Q, mode="persona")
    assert evs[0]["mode"] == "persona"
    assert [e["type"] for e in evs][-2:] == ["citations", "done"]
    assert "[1]" in fake_llm.last["messages"][0]["content"]


# ------------------------------------------------------------ P2-1 --------

def test_is_followup():
    f = rag.is_followup
    for q in ("What about after 2020?", "Why did they do that?",
              "And the glaciers?", "Tell me more", "how so…",
              "Is it dangerous?", "Moraines?"):
        assert f(q), q
    for q in (BEES_Q, ICE_Q, "Why did Newton build a reflecting telescope?",
              "What landforms do retreating glaciers leave behind?"):
        assert not f(q), q


def test_first_turn_no_rewrite(server, fake_llm, make_collection):
    col = make_collection()
    fake_llm.reply = "Answer [1]."
    q = "How do worker bees build the wax comb and why is it hexagonal?"
    assert rag.is_followup(q)              # pronoun — but nothing to follow
    for kw in ({}, {"conversation_id": "brand-new-conv"}):
        before = len(fake_llm.requests)
        evs = _ask(server, col, q, **kw)
        assert len(fake_llm.requests) == before + 1
        assert _rewrite_requests(fake_llm) == []
        meta = evs[0]
        assert meta["rewritten_question"] is None
        assert meta["history_used"] == 0
        assert [e["type"] for e in evs][-2:] == ["citations", "done"]
        assert "Earlier in this conversation" not in fake_llm.prompt_text()


def test_followup_rewritten(server, fake_llm, make_collection):
    col = make_collection()
    fake_llm.reply = lambda body: (ICE_Q if _is_rewrite(body)
                                   else "It is answered [1].")
    conv = "chat-1"
    _ask(server, col, BEES_Q, conversation_id=conv)
    assert len(fake_llm.requests) == 1

    follow = "And what about those?"
    index = rag.load_index(library.get(col.id))
    assert not rag.gate_ok(index, follow)     # alone it would be gated out
    evs = _ask(server, col, follow, conversation_id=conv)
    assert len(fake_llm.requests) == 3        # one rewrite + one answer
    [rw] = _rewrite_requests(fake_llm)
    assert rw["temperature"] == 0
    assert rw["max_tokens"] == config.REWRITE_MAX_TOKENS
    assert BEES_Q in rw["messages"][1]["content"]
    assert follow in rw["messages"][1]["content"]
    meta = evs[0]
    assert meta["rewritten_question"] == ICE_Q
    assert meta["history_used"] == 1
    assert meta["sources"][0]["source"] == "glaciers.docx"  # retrieved on it
    ans = fake_llm.last["messages"][1]["content"]
    assert ans.startswith("Earlier questions in this conversation (context only; answer the new question below):\n- " + BEES_Q)
    # earlier ANSWERS never go into the prompt (a small model copied them)
    assert "A: " not in ans
    assert f"Question: {follow}\n(Standalone form: {ICE_Q})" in ans
    turns = conversations.load(col, conv)
    assert turns[1]["rewritten_question"] == ICE_Q
    assert turns[1]["citations"]["used"] == [1]

    # a standalone question in the same conversation: no rewrite, and no
    # history in its prompt (with it, a small model re-answered the old one)
    q3 = "Why did Newton build a reflecting telescope?"
    evs = _ask(server, col, q3, conversation_id=conv)
    assert len(fake_llm.requests) == 4
    assert len(_rewrite_requests(fake_llm)) == 1
    assert evs[0]["rewritten_question"] is None
    assert evs[0]["history_used"] == 0
    assert "Earlier questions" not in fake_llm.last["messages"][1]["content"]


def test_followup_uses_last_three_turns(server, fake_llm, make_collection):
    col = make_collection()
    fake_llm.reply = lambda body: (ICE_Q if _is_rewrite(body) else "ok [1]")
    conv = "chat-4"
    qs = [BEES_Q, ICE_Q, "Why did Newton build a reflecting telescope?",
          "What did Fresnel invent for lighthouses?"]
    for q in qs:
        _ask(server, col, q, conversation_id=conv)
    _ask(server, col, "And those?", conversation_id=conv)
    rw = _rewrite_requests(fake_llm)[-1]["messages"][1]["content"]
    assert BEES_Q not in rw and all(q in rw for q in qs[1:])


def test_rewrite_junk_or_failure_falls_back(server, fake_llm, make_collection,
                                            monkeypatch):
    col = make_collection()
    hist = [{"question": BEES_Q, "answer": "Wax glands [1]."}]
    # unreachable backend → None, no exception
    monkeypatch.setattr(config, "LLM_URL", "http://127.0.0.1:9/v1")
    assert rag.rewrite_question(col, "What about it?", hist) is None
    monkeypatch.setattr(config, "LLM_URL", fake_llm.url)
    for junk in ("", "   ", "...", "?!"):
        fake_llm.reply = junk
        assert rag.rewrite_question(col, "What about it?", hist) is None
    fake_llm.reply = "Standalone question: \"How is wax made?\"\nextra"
    assert rag.rewrite_question(col, "What about it?", hist) == \
        "How is wax made?"
    # end to end: junk rewrite → original question is used, answer streams
    fake_llm.reply = lambda body: "" if _is_rewrite(body) else "Fine [1]."
    _ask(server, col, BEES_Q, conversation_id="c-junk")
    del fake_llm.requests[:]
    follow = "And how do the worker bees build the wax comb then?"
    assert rag.is_followup(follow)
    evs = _ask(server, col, follow, conversation_id="c-junk")
    assert len(_rewrite_requests(fake_llm)) == 1
    assert evs[0]["rewritten_question"] is None
    assert [e["type"] for e in evs][-2:] == ["citations", "done"]


# ------------------------------------------------- history vs. budget ----

def _chunk(i, words):
    return {"id": f"doc.txt-s{i}-{i * 1000}", "source": "doc.txt",
            "essay": f"Section {i}", "work": "doc", "page_start": 1,
            "page_end": 1, "text": " ".join(f"w{i}x{j}" for j in range(words))}


def _need(system, user):
    return math.ceil(len(system.split()) * 1.35 + len(user.split()) * 1.35)


def test_history_trimmed_before_excerpts(monkeypatch):
    system = "sys prompt words here"
    hits = [(1.0 - i / 100, _chunk(i, 60)) for i in range(4)]
    history = [{"question": f"old question {i} ttt{i}",
                "answer": " ".join(f"a{i}y{j}" for j in range(80))}
               for i in range(3)]
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 100)

    # budget = every hit, no history: history is gone, no excerpt dropped
    _, bare, _ = rag.fit_context(system, "q?", hits)
    monkeypatch.setattr(config, "LLM_CTX", _need(system, bare) + 100 + 5)
    sent, user, dropped, used = rag.fit_prompt(system, "q?", hits, history)
    assert dropped == 0 and len(sent) == 4 and used == 0
    assert "Earlier questions" not in user

    # room for exactly the most recent turn: oldest turns go first
    monkeypatch.setattr(config, "LLM_CTX", 100000)
    one = rag.fit_prompt(system, "q?", hits, history[-1:])[1]
    assert "ttt2" in one
    monkeypatch.setattr(config, "LLM_CTX", _need(system, one) + 100 + 5)
    sent, user, dropped, used = rag.fit_prompt(system, "q?", hits, history)
    assert dropped == 0 and used == 1
    assert "ttt2" in user and "ttt1" not in user and "ttt0" not in user

    # plenty of room: all three turns' questions, oldest first; their
    # answers are never included
    monkeypatch.setattr(config, "LLM_CTX", 100000)
    sent, user, dropped, used = rag.fit_prompt(system, "q?", hits, history,
                                               answer_words=20)
    assert used == 3 and dropped == 0
    assert user.index("ttt0") < user.index("ttt1") < user.index("ttt2")
    assert "a0y" not in user

    # too small even without history: excerpts drop as before (M1)
    monkeypatch.setattr(config, "LLM_CTX", _need(system, bare) + 100 - 40)
    sent, user, dropped, used = rag.fit_prompt(system, "q?", hits, history)
    assert used == 0 and dropped >= 1
    assert "Earlier questions" not in user


def test_ask_stream_history_only_for_followups(make_collection, fake_llm,
                                               monkeypatch):
    # JMS bug (2026-11-19): with the previous Q&A in every prompt, the 1.5B
    # model answered "does election bring real social change?" with the
    # previous answer word for word. History now goes in only for follow-ups,
    # and only as earlier questions.
    col = make_collection()
    fake_llm.reply = lambda body: (ICE_Q if _is_rewrite(body) else "ok [1]")
    hist = [{"question": BEES_Q, "answer": "zqlong " * 400}]
    monkeypatch.setattr(config, "LLM_CTX", 100000)
    res = rag.ask_stream(col, ICE_Q, history=hist)          # standalone
    "".join(res["stream"])
    assert res["history_used"] == 0 and res["rewritten_question"] is None
    assert "zqlong" not in fake_llm.prompt_text()
    assert BEES_Q not in fake_llm.last["messages"][1]["content"]
    res = rag.ask_stream(col, "And what about those?", history=hist)
    "".join(res["stream"])
    assert res["rewritten_question"] == ICE_Q and res["history_used"] == 1
    user = fake_llm.last["messages"][1]["content"]
    assert BEES_Q in user and "zqlong" not in user


# ------------------------------------------------------------ ask.py -----

def test_cli_chat_followup_and_citation_warning(make_collection, fake_llm,
                                                monkeypatch, capsys):
    import builtins
    import ask
    col = make_collection()
    fake_llm.reply = lambda body: (ICE_Q if _is_rewrite(body)
                                   else "Answer [1] and [42].")
    lines = iter([BEES_Q, "And what about those?", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(lines))
    monkeypatch.setattr(sys, "argv", ["ask.py", "-c", col.id, "--chat"])
    ask.main()
    out = capsys.readouterr().out
    assert len(_rewrite_requests(fake_llm)) == 1     # only the follow-up
    assert f"(searched for: {ICE_Q})" in out
    assert out.count("Citation check: [42]") == 2
    assert "[1] " in out and "(not cited)" in out
    last = _answer_requests(fake_llm)[-1]["messages"][1]["content"]
    assert last.startswith("Earlier questions in this conversation (context only; answer the new question below):\n- " + BEES_Q)


def test_numbered_prompt_ends_with_cite_reminder(make_collection):
    # small models cite reliably only with the reminder as the last line
    # (plan.md, "Citation prompt"); quotes mode keeps its own format
    col = make_collection()
    hits = rag.BM25Index(rag.load_chunks(col)).search(BEES_Q, 3)
    sys_ = rag.system_scholar(col)
    _, user, _, _ = rag.fit_prompt(sys_, BEES_Q, hits)
    assert user.rstrip().endswith(rag.CITE_REMINDER)
    assert "[1] [" not in user                  # no nested header brackets
    _, quser, _, _ = rag.fit_prompt(rag.system_quotes(col), BEES_Q, hits,
                                    numbered=False)
    assert rag.CITE_REMINDER not in quser
    assert quser.rstrip().endswith(BEES_Q)
