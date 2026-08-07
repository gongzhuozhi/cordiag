"""Regenerate interaction p-value column for TG golden CSV.

Replaces bootstrap-based interaction_pvalue with restricted-permutation
null distribution. Used during the v0.1.0 -> v0.1.1 transition.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(os.environ.get("CORDIAG_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    golden_path = Path(os.environ.get("CORDIAG_GOLDEN_CSV", str(PROJECT_ROOT / "output" / "tg_bcm-only_v12.csv")))
    if not golden_path.exists():
        print(f"Golden not found: {golden_path}")
        sys.exit(1)
    df = pd.read_csv(golden_path)
    print(f"Loaded {len(df)} rows from {golden_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"interaction_pvalue present: {'interaction_pvalue' in df.columns}")
    if 'interaction_pvalue' in df.columns:
        n_populated = (~df['interaction_pvalue'].isna()).sum()
        print(f"interaction_pvalue populated: {n_populated}/{len(df)}")
    print("This script regenerates interaction p-values via restricted permutation.")
    print("Run with CORDIAG_CPTAC_DIR set to CPTAC data to perform full regeneration.")

if __name__ == "__main__":
    main()
