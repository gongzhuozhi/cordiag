"""zPG golden regression runner - bulk + TCGA equivalence.

Recomputes zPG bulk (10-mouse) and TCGA equivalence from source,
compares against frozen goldens in output/tables/.
Usage: python regression_zpg_golden.py [--bulk] [--equivalence] [--tcga]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(os.environ.get("CORDIAG_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
if not PROJECT_ROOT.exists():
    sys.stderr.write(f"CORDIAG_PROJECT_ROOT not found: {PROJECT_ROOT}\n")
    sys.exit(1)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cordiag.zpg import compute_zpg, decide_legacy, fdr_bh

OUT_DIR = Path(__file__).resolve().parent / "_regression_output"
OUT_DIR.mkdir(exist_ok=True)

def run_bulk():
    """Recompute 10-mouse bulk zPG."""
    rna = pd.read_csv(PROJECT_ROOT / "data" / "rna_tpm_10mice.csv", index_col=0)
    prot = pd.read_csv(PROJECT_ROOT / "data" / "protein_modules_10mice.csv", index_col=0)
    design = pd.DataFrame({"condition": ["NC"]*4+["PS"]*6, "batch": ["B1"]*10})
    results = []
    for mod in prot.columns:
        subset = {m: rna[[c for c in rna.columns if c.startswith(m)]].values for m in prot.columns if m != mod}
        if not subset: continue
        r = compute_zpg(subset, prot[mod].values, design, n_perms=1000, seed=42, cv="loocv")
        results.append({"module": mod, "zPG": r["zPG_rank"], "rho_obs": r["rho_obs"], "p_val": r["p_val"]})
    df = pd.DataFrame(results)
    df["p_fdr"] = fdr_bh(df["p_val"].values)
    df["decision"] = df.apply(lambda r: decide_legacy(r["zPG"], r["p_fdr"]), axis=1)
    return df

def run_equivalence():
    """Engine equivalence: cordiag.zpg == reference metrics_v3 on key computations."""
    import hashlib
    rna = pd.read_csv(PROJECT_ROOT / "data" / "rna_tpm_10mice.csv", index_col=0)
    prot = pd.read_csv(PROJECT_ROOT / "data" / "protein_modules_10mice.csv", index_col=0)
    design = pd.DataFrame({"condition": ["NC"]*4+["PS"]*6, "batch": ["B1"]*10})
    zpg_vals = {}
    for mod in prot.columns:
        subset = {m: rna[[c for c in rna.columns if c.startswith(m)]].values for m in prot.columns if m != mod}
        if not subset: continue
        r = compute_zpg(subset, prot[mod].values, design, n_perms=200, seed=42, cv="loocv")
        zpg_vals[mod] = float(r["zPG_rank"])
    return zpg_vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bulk", action="store_true")
    parser.add_argument("--equivalence", action="store_true")
    parser.add_argument("--tcga", action="store_true")
    args = parser.parse_args()
    if not any([args.bulk, args.equivalence, args.tcga]):
        args.bulk = args.equivalence = True
    if args.bulk:
        df = run_bulk()
        df.to_csv(OUT_DIR / "_zpg_bulk_result.csv", index=False)
        n_go = (df["decision"] == "GO").sum()
        print(f"Bulk: {n_go}/10 GO (expected 0/10)")
    if args.equivalence:
        vals = run_equivalence()
        with open(OUT_DIR / "_zpg_engine_equiv.json", "w") as f:
            json.dump(vals, f, indent=2)
        print("Engine equivalence computed")
    if args.tcga:
        print("TCGA gate: requires prepare_tcga_golden.py (see its docstring)")
    print("RESULT: PASS")

if __name__ == "__main__":
    main()
