#!/usr/bin/env bash
# reproduce_and_verify.sh — one entry point to re-check everything in this bundle.
#
#   ./reproduce_and_verify.sh                 interactive menu
#   ./reproduce_and_verify.sh verify          check every path / def-line / number anchor / test name used in the bundle
#   ./reproduce_and_verify.sh audit-tests     run only the causality / leakage / invariance unit tests
#   ./reproduce_and_verify.sh pytest-q2       q2_p1 + app tests (the set the synthesis refers to)
#   ./reproduce_and_verify.sh pytest          the full test suite (slow: several minutes)
#   ./reproduce_and_verify.sh summary <dir>   show experiments/<dir>/summary.json (e.g. Q2-P1-E05_confirmation)
#   ./reproduce_and_verify.sh list            list experiment folders
#   ./reproduce_and_verify.sh open            open index.html in the default browser
#   ./reproduce_and_verify.sh streamlit [port]  launch the Streamlit explorer (../run_app.sh)
#   ./reproduce_and_verify.sh mlflow [port]   launch a local MLflow UI on experiments/mlflow.db
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${BUNDLE}/.." && pwd)"
EXP="${ROOT}/experiments"
PY="${EXP}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

need_pytest() {
  if ! "${PY}" -c "import pytest" >/dev/null 2>&1; then
    echo "pytest is not importable with ${PY}. Run: cd ${EXP} && uv sync --extra pose" >&2
    exit 1
  fi
}

cmd_verify() {
  BUNDLE="${BUNDLE}" ROOT="${ROOT}" "${PY}" - <<'PY'
import json, os, re, sys
from pathlib import Path

bundle, root = Path(os.environ["BUNDLE"]), Path(os.environ["ROOT"])
html = (bundle / "index.html").read_text(encoding="utf-8")
fails, warns, ok = [], [], 0

def block(name):
    m = re.search(r'<script id="%s" type="application/json">(.*?)</script>' % name, html, re.S)
    if not m:
        fails.append(f"index.html: missing <script id={name}>")
        return []
    return json.loads(m.group(1))

def lines(rel):
    p = root / rel
    return p.read_text(errors="replace").splitlines() if p.exists() else None

# 1. audit rows: file exists, symbol on the stated line, test on the stated line, report anchor present
for r in block("audit-data"):
    src = lines(r["src"])
    if src is None:
        fails.append(f"{r['id']}: missing source {r['src']}"); continue
    if r["src_line"] is None or r["src_line"] > len(src) or r["src_sym_rx"] is None or not re.search(r["src_sym_rx"], src[r["src_line"] - 1]):
        fails.append(f"{r['id']}: {r['src']}:{r['src_line']} does not match /{r['src_sym_rx']}/")
    else:
        ok += 1
    if r["test"]:
        t = lines(r["test"])
        if t is None:
            fails.append(f"{r['id']}: missing test file {r['test']}")
        elif not re.search(r["test_sym_rx"], t[r["test_line"] - 1]):
            fails.append(f"{r['id']}: {r['test']}:{r['test_line']} does not match /{r['test_sym_rx']}/")
        else:
            ok += 1
    rep = (root / r["report"])
    if not rep.exists():
        fails.append(f"{r['id']}: missing report {r['report']}")
    elif r["anchor"] not in rep.read_text(errors="replace"):
        fails.append(f"{r['id']}: anchor {r['anchor']!r} not found in {r['report']}")
    else:
        ok += 1

# 2. warning lines (W1..W6): the quoted code fragment must be on the stated line
for w in block("warn-data"):
    ln = lines(w["file"])
    if ln is None or not re.search(w["rx"], ln[w["line"] - 1]):
        fails.append(f"{w['id']}: {w['file']}:{w['line']} does not match /{w['rx']}/")
    else:
        ok += 1

# 3. literal number anchors
for f, lit in block("anchor-data"):
    p = root / f
    if not p.exists():
        fails.append(f"anchor file missing: {f}")
    elif lit not in p.read_text(errors="replace"):
        fails.append(f"anchor {lit!r} not in {f}")
    else:
        ok += 1

# 4. path-like tokens in markdown + html
roots = [root, root / "experiments", root / "experiments/src/edgelab", root / "experiments/tests", bundle, root / "docs"]
pat = re.compile(r"`((?:\.\./)?[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+\.(?:md|py|csv|json|html|sh|yaml|yml|png|tar|pt))`")
texts = {f.name: f.read_text(encoding="utf-8") for f in bundle.glob("*.md")}
texts["index.html"] = html
seen = set()
for name, txt in texts.items():
    for m in pat.finditer(txt):
        tok = m.group(1)
        if tok in seen or "<" in tok or tok.startswith("~"):
            continue
        seen.add(tok)
        cand = [(r / tok) for r in roots]
        cand += [(bundle / tok)]
        if any(c.exists() for c in cand):
            ok += 1
        elif tok.startswith("../"):
            if (bundle / tok).exists(): ok += 1
            else: fails.append(f"{name}: path not found: {tok}")
        else:
            warns.append(f"{name}: path not resolved (may be data outside the repo or generated): {tok}")

# 5. markdown relative links
for name, txt in texts.items():
    for m in re.finditer(r"\]\((?!http)([^)#]+)\)", txt):
        tgt = m.group(1)
        if not (bundle / tgt).exists():
            fails.append(f"{name}: broken relative link {tgt}")
        else:
            ok += 1

# 6. soft check: every decimal number written in the bundle should occur somewhere in the repo's own reports
corpus = []
skip_dirs = {".venv", "__pycache__", "artifacts", "data", ".git", "HUMAN_3DAY_REVIEW_BUNDLE", "node_modules", ".pytest_cache"}
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in skip_dirs]
    for f in fn:
        if f.endswith((".md", ".csv", ".json", ".py", ".txt", ".yaml", ".html")):
            p = Path(dp, f)
            try:
                if p.stat().st_size < 30_000_000:
                    corpus.append(p.read_text(errors="replace"))
            except OSError:
                pass
big = "\n".join(corpus).replace("\u2212", "-")
num = re.compile(r"(?<![\w.])[+\-\u2212]?\d[\d,]*\.\d+(?![\w])")
unmatched = {}
derived = set(json.loads(re.search(r'<script id="derived-data" type="application/json">(.*?)</script>', html, re.S).group(1)))
for name, txt in texts.items():
    scrub = re.sub(r"<script id=\"(audit-data|anchor-data|warn-data|derived-data)\".*?</script>", "", txt, flags=re.S) if name == "index.html" else txt
    for m in num.finditer(scrub):
        t = m.group(0).replace("\u2212", "-").lstrip("+").lstrip("-")
        if t in derived:
            continue
        if t not in big and t.replace(",", "") not in big:
            unmatched.setdefault(t, set()).add(name)

print(f"checks passed : {ok}")
print(f"FAILURES      : {len(fails)}")
for f in fails: print("  FAIL", f)
print(f"warnings      : {len(warns)}")
for w in warns[:40]: print("  warn", w)
print(f"numbers not found verbatim in repo files (review; allowed list = derived-data): {len(unmatched)}")
for k in sorted(unmatched)[:60]: print("  ?", k, sorted(unmatched[k]))
sys.exit(1 if fails else 0)
PY
}

cmd_audit_tests() {
  need_pytest
  cd "${EXP}"
  "${PY}" -m pytest -q \
    tests/test_q2_p1_e00.py tests/test_refresh_causality.py tests/test_models_stgnf_flow.py \
    tests/test_q2_p1_e06.py tests/test_q2_p1_e07.py tests/test_q2_p1_e08.py tests/test_q2_p1_e09.py tests/test_q2_p1_e10.py \
    tests/test_q2_p1_e01.py tests/test_q2_p1_e03.py tests/test_q2_p1_e05.py "$@"
}

cmd_pytest_q2() { need_pytest; cd "${EXP}"; "${PY}" -m pytest -q tests/test_q2_p1_*.py tests/test_app_catalog.py "$@"; }
cmd_pytest()    { need_pytest; cd "${EXP}"; "${PY}" -m pytest -q tests "$@"; }

cmd_list() { ls -1d "${EXP}"/EXP-* "${EXP}"/Q2-P1-* 2>/dev/null | sed "s#${EXP}/##"; }

cmd_summary() {
  local d="${1:-}"
  if [[ -z "${d}" ]]; then echo "usage: $0 summary <experiment-dir>   (try: $0 list)"; exit 2; fi
  local f="${EXP}/${d}/summary.json"
  if [[ ! -f "${f}" ]]; then
    echo "no summary.json in ${d}; files:"; ls -1 "${EXP}/${d}"; exit 1
  fi
  F="${f}" "${PY}" - <<'PY'
import json, os
d = json.load(open(os.environ["F"]))
def show(x, depth=0, key="", limit=14):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{key}{{{len(x)} keys}}")
        if depth < 2:
            for i, (k, v) in enumerate(x.items()):
                if i >= limit: print(f"{pad}  ... {len(x) - limit} more keys"); break
                show(v, depth + 1, f"{k}: ")
    elif isinstance(x, list):
        print(f"{pad}{key}[{len(x)} items]" + (f" first={json.dumps(x[0])[:160]}" if x else ""))
    else:
        print(f"{pad}{key}{json.dumps(x)[:160]}")
show(d, 0, os.environ["F"].split("/experiments/")[-1] + " -> ")
PY
}

cmd_open() {
  if command -v open >/dev/null 2>&1; then open "${BUNDLE}/index.html"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "${BUNDLE}/index.html"
  else echo "open ${BUNDLE}/index.html in your browser"; fi
}

cmd_streamlit() { cd "${ROOT}"; ./run_app.sh --port "${1:-8501}"; }

cmd_mlflow() {
  cd "${EXP}"
  echo "MLflow UI -> http://127.0.0.1:${1:-5000}  (backend sqlite:///mlflow.db; reference copy of runs: ${ROOT}/mlflow_runs.csv)"
  if command -v uv >/dev/null 2>&1; then uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port "${1:-5000}"
  else "${PY}" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port "${1:-5000}"; fi
}

menu() {
  while true; do
    cat <<EOF

 HUMAN_3DAY_REVIEW_BUNDLE — reproduce & verify
  1) verify bundle (paths, lines, number anchors, tests)
  2) run audit tests (causality / leakage)
  3) run q2_p1 + app tests (298 tests)
  4) run FULL pytest (slow)
  5) show an experiment summary.json
  6) list experiment folders
  7) open index.html
  8) launch Streamlit (http://127.0.0.1:8501)
  9) launch MLflow UI (http://127.0.0.1:5000)
  q) quit
EOF
    read -r -p " > " c
    case "${c}" in
      1) cmd_verify || true;;
      2) cmd_audit_tests || true;;
      3) cmd_pytest_q2 || true;;
      4) cmd_pytest || true;;
      5) cmd_list; read -r -p " folder name: " d; cmd_summary "${d}" || true;;
      6) cmd_list;;
      7) cmd_open;;
      8) cmd_streamlit;;
      9) cmd_mlflow;;
      q|Q) exit 0;;
      *) echo "?";;
    esac
  done
}

case "${1:-menu}" in
  verify) cmd_verify;;
  audit-tests) shift; cmd_audit_tests "$@";;
  pytest-q2) shift; cmd_pytest_q2 "$@";;
  pytest) shift; cmd_pytest "$@";;
  summary) shift; cmd_summary "${1:-}";;
  list) cmd_list;;
  open) cmd_open;;
  streamlit) shift; cmd_streamlit "${1:-8501}";;
  mlflow) shift; cmd_mlflow "${1:-5000}";;
  menu|"") menu;;
  *) sed -n 2,16p "${BASH_SOURCE[0]}"; exit 2;;
esac
