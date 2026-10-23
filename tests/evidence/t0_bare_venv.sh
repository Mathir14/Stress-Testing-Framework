#!/usr/bin/env bash
# Rebuild the T0 (bare-interpreter) test environment from scratch and run the
# T0 tier against it.
#
# Why this script exists (reviewer attempt-3, minor m3): the headline T0 evidence
# was originally reported as "ran under /tmp/opencode/t0venv", a path that no
# longer existed by the time it was audited, so the claim was unreproducible as
# written.  A claim about a gate that cannot be re-run is not evidence.  This
# script makes the claim falsifiable: it prints the interpreter version, installs
# *only* pytest from requirements-dev.txt, proves the scientific stack is
# genuinely absent, and then runs the four T0 files.
#
# Usage:  bash tests/evidence/t0_bare_venv.sh [scratch-dir]
#
# architecture.md §10.1 makes the tiering binding: T0 is
# test_architecture.py, test_model_io.py, test_config.py and test_cli.py, and
# they must pass on an interpreter with nothing but the standard library and
# pytest.  §4.5.6 (lazy estimator gate) and ADR-011.3 (library_versions via
# importlib.metadata) exist precisely so this run is possible -- if either
# regressed to a module-level `import sklearn`, this script would fail, which is
# the point.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# <evidence>/ -> tests/ -> repo root
REPO="$(cd "${HERE}/../.." && pwd)"
SCRATCH="${1:-/tmp/opencode/t0-bare}"
PYTEST_PIN="$(grep -E '^pytest==' "${REPO}/requirements-dev.txt" | head -1)"

echo "repo:        ${REPO}"
echo "scratch venv: ${SCRATCH}"
echo "pin:         ${PYTEST_PIN}"

rm -rf "${SCRATCH}"
python3 -m venv "${SCRATCH}"

# --no-index is deliberately NOT used: pytest must come from the index like any
# other pin. Only the dev requirement is installed -- not requirements.txt.
# Strip the "==" pin and let pip resolve, so the venv is genuinely minimal.
"${SCRATCH}/bin/python" -m pip install --quiet --upgrade pip
"${SCRATCH}/bin/python" -m pip install --quiet "${PYTEST_PIN}"

echo
echo "--- interpreter ---"
"${SCRATCH}/bin/python" -V
"${SCRATCH}/bin/python" -m pytest --version

echo
echo "--- the scientific stack MUST be absent, or T0 is not T0 ---"
for mod in numpy pandas sklearn streamlit plotly; do
  if "${SCRATCH}/bin/python" -c "import ${mod}" 2>/dev/null; then
    echo "FAIL: ${mod} is importable in the bare venv"
    exit 1
  fi
  echo "ok: ${mod} unimportable"
done

echo
echo "--- T0 tier (architecture.md §10.1) ---"
cd "${REPO}"
"${SCRATCH}/bin/python" -m pytest \
  tests/test_architecture.py \
  tests/test_model_io.py \
  tests/test_config.py \
  tests/test_cli.py \
  -q

echo
echo "--- the CLI must also work on the bare interpreter (§4.9) ---"
# -S hides every site-packages directory, which is the strictest form of the
# T0 promise: --help / version / info may not need anything installed at all.
for arg in --help --version info; do
  "${SCRATCH}/bin/python" -S main.py ${arg} >/dev/null
  echo "ok: python -S main.py ${arg} -> exit 0"
done
if "${SCRATCH}/bin/python" -S main.py --non-existent-option >/dev/null; then
  echo "ok: an unrecognised argument is reported and ignored -> exit 0"
else
  echo "FAIL: unrecognised argument did not exit 0"
  exit 1
fi

echo
echo "T0 GATE VERIFIED"