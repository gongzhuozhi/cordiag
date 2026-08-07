#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONNOUSERSITE=1
export CORDIAG_PROJECT_ROOT="$REPO_ROOT"
export CORDIAG_GOLDEN_CSV="$REPO_ROOT/output/tg_bcm-only_v12.csv"
export CORDIAG_CALIBRATION_DIR="$REPO_ROOT/output/calibration"
if command -v cygpath >/dev/null 2>&1; then
  export PYTHONPATH="$(cygpath -w "$REPO_ROOT")${PYTHONPATH:+;$PYTHONPATH}"
else
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
PYTHON="${PYTHON:-python}"
MODE="${1:-quick}"
echo "== [cordiag] reproduction - mode: $MODE =="
mkdir -p code/_env
"$PYTHON" -c "
import sys
for mod in ('numpy','pandas','scipy','sklearn','statsmodels'):
    try:
        m=__import__(mod)
        print(f'{mod} {m.__version__}')
    except ImportError:
        print(f'{mod} MISSING')
print(f'python {sys.version.split()[0]}')
" | tee code/_env/versions.txt
"$PYTHON" -m pip freeze > code/_env/pip_freeze.txt
( cd cordiag && "$PYTHON" -m pytest tests/ -v --tb=short )
if [ "$MODE" = "full" ] || [ "$MODE" = "--full" ]; then
    ( cd cordiag && "$PYTHON" tests/regression_tg_golden.py --workers 16 )
    ( cd cordiag && "$PYTHON" tests/regression_zpg_golden.py )
else
    ( cd cordiag && "$PYTHON" tests/regression_tg_golden.py --subset 1 )
fi
echo "== [cordiag] DONE ($MODE) =="
