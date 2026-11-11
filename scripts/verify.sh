#!/usr/bin/env bash
# Milestone verification for implem-plan-enhancements.md.
#
#   bash scripts/verify.sh M0|M1|M2|M3|M4|M5-<n>
#
# Runs the full pytest suite, the fixture eval (tests/eval_fixture.py), the
# milestone's checks (scripts/verify_checks.py), and confirms devdata/ and
# writings/ were not modified. The LAST line is always exactly
#   VERIFY <M>: PASS   or   VERIFY <M>: FAIL
# Read-only apart from tests/fixtures/baselines/<M>.json (written on PASS
# for M0/M2/M3/M4 — commit it with the milestone).
set -u
cd "$(dirname "$0")/.."
M="${1:-}"
PY=.venv/bin/python
TMP="$(mktemp -d "${TMPDIR:-/tmp}/anyrag-verify.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
FAILED=0
fail() { echo "  FAIL  $*"; FAILED=1; }

if [ -z "$M" ]; then
  echo "usage: bash scripts/verify.sh M0|M1|M2|M3|M4|M5-<n>"
  echo "VERIFY ?: FAIL"
  exit 2
fi
echo "== verify $M"

snapshot() {   # path, size, mtime of every file under devdata/ and writings/
  "$PY" - "$1" <<'EOF'
import os, sys
rows = []
for top in ("devdata", "writings"):
    for d, _, files in os.walk(top):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            rows.append(f"{p}\t{st.st_size}\t{st.st_mtime_ns}")
open(sys.argv[1], "w").write("\n".join(sorted(rows)))
EOF
}

[ -x "$PY" ] || { fail ".venv missing — run: bash setup.sh"; echo "VERIFY $M: FAIL"; exit 1; }
"$PY" -c "import pytest" 2>/dev/null \
  || fail "pytest not installed — run: $PY -m pip install -r requirements-dev.txt"

snapshot "$TMP/before.tsv"

echo "-- pytest"
if "$PY" -m pytest -q -rfE --junitxml="$TMP/junit.xml" > "$TMP/pytest.log" 2>&1; then
  echo "  ok    $(grep -E '^[0-9]+ (passed|failed)|passed|failed' "$TMP/pytest.log" | tail -1)"
else
  tail -25 "$TMP/pytest.log" | sed 's/^/        /'
  fail "pytest suite not green"
fi

echo "-- fixture eval"
if ! "$PY" tests/eval_fixture.py --out "$TMP/eval.json" > "$TMP/eval.log" 2>&1; then
  tail -15 "$TMP/eval.log" | sed 's/^/        /'
  fail "tests/eval_fixture.py failed"
fi

echo "-- milestone checks ($M)"
if [ -f "$TMP/junit.xml" ] && [ -f "$TMP/eval.json" ]; then
  "$PY" scripts/verify_checks.py "$M" "$TMP/junit.xml" "$TMP/eval.json" || FAILED=1
else
  fail "no pytest/eval results to check"
fi

echo "-- user data untouched"
snapshot "$TMP/after.tsv"
if cmp -s "$TMP/before.tsv" "$TMP/after.tsv"; then
  echo "  ok    devdata/ and writings/ unchanged ($(wc -l < "$TMP/after.tsv" | tr -d ' ') files)"
else
  diff "$TMP/before.tsv" "$TMP/after.tsv" | head -10 | sed 's/^/        /'
  fail "devdata/ or writings/ modified during verification"
fi
if [ -n "$(git status --porcelain -- devdata writings 2>/dev/null)" ]; then
  fail "git status shows changes under devdata/ or writings/"
else
  echo "  ok    git status clean under devdata/ and writings/"
fi

if [ "$FAILED" -eq 0 ]; then
  echo "VERIFY $M: PASS"
else
  echo "VERIFY $M: FAIL"
  exit 1
fi
