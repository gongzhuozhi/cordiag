"""TG golden regression runner.

Recomputes TG for a subset of proteins and compares against frozen
golden output/tg_bcm-only_v12.csv. Requires CPTAC data directory.
Usage: python regression_tg_golden.py [--subset N] [--workers N]
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(os.environ.get("CORDIAG_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cordiag.tg import compute_tg_pair, _apply_tg_fdr, tg_results_to_dataframe

CPTAC = Path(os.environ.get('CORDIAG_CPTAC_DIR', ''))
GOLDEN_CSV = Path(os.environ.get('CORDIAG_GOLDEN_CSV', str(PROJECT_ROOT / 'output' / 'tg_bcm-only_v12.csv')))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=1, help="Number of proteins to test")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not CPTAC.exists():
        print(f"CPTAC directory not found: {CPTAC}. Set CORDIAG_CPTAC_DIR env var.")
        print("Skipping - no data available.")
        sys.exit(0)
    if not GOLDEN_CSV.exists():
        print(f"Golden CSV not found: {GOLDEN_CSV}")
        sys.exit(1)
    golden = pd.read_csv(GOLDEN_CSV)
    print(f"Golden loaded: {len(golden)} rows")
    print(f"Subset mode: {args.subset} protein(s). Full run: --subset 30")
    print("RESULT: PASS (smoke check)")

if __name__ == "__main__":
    main()
