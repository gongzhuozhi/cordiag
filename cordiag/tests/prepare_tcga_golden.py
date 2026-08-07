"""TCGA-TGCT golden rebuild - documented-recipe generator.

Rebuilds tcga_validation.csv from processed TCGA-TGCT inputs
using the data_driven_modules construction (eigh 1st-PC loading sort
+ quantile split) and compute_zpg with seed=42.
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

PROJECT_ROOT = Path(os.environ.get("CORDIAG_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cordiag.zpg import compute_zpg, data_driven_modules, decide_legacy
from statsmodels.stats.multitest import multipletests

OUT_DIR = Path(__file__).resolve().parent / "_regression_output"
ZPG_GO_TOL = 0.1
TCGA = PROJECT_ROOT / "data" / "external" / "TCGA_TGCT" / "processed"
GOLDEN_PATH = PROJECT_ROOT / "output" / "tables" / "tcga_validation.csv"
GOLDEN_ZPG = {"M4": 2.9869213288520022, "M7": 3.026976400801053}

def build_inputs():
    prot_full = pd.read_csv(TCGA / "protein_paired_common_genes.csv", index_col=0).T
    rna_full = pd.read_csv(TCGA / "rna_paired_common_genes.csv", index_col=0).T
    design_df = pd.read_csv(TCGA / "design_matrix.csv")
    design_df["sample_id"] = design_df["sample_id"].str.strip()
    dmap = {str(r["sample_id"]): (str(r["condition"]).strip(), str(r["batch"]).strip()) for _, r in design_df.iterrows()}
    known = [s for s in prot_full.index if dmap.get(s, ("", ""))[0] != ""]
    modules = data_driven_modules(prot_full.loc[known], n_modules=8)
    keep = [s for s in rna_full.index if dmap.get(s, ("", ""))[0].lower() in ("stagei", "stageiii")]
    design = pd.DataFrame({"condition": [dmap[s][0] for s in keep], "batch": [dmap[s][1] for s in keep]})
    rna_mod = {}
    for m, gs in modules.items():
        g = [x for x in gs if x in rna_full.columns]
        if len(g) >= 2: rna_mod[m] = np.log1p(rna_full.loc[keep, g]).mean(axis=1).values
    prot_mod = {m: prot_full.loc[keep, [x for x in gs if x in prot_full.columns]].mean(axis=1).values for m, gs in modules.items()}
    return rna_mod, prot_mod, design, modules, len(keep)

def main():
    n_perms = int(os.environ.get("CORDIAG_ZPG_TCGA_PERMS", "1000"))
    all_mods = os.environ.get("CORDIAG_ZPG_TCGA_ALL") == "1"
    print(f"=== TCGA-TGCT golden rebuild (n_perms={n_perms}) ===", flush=True)
    rna_mod, prot_mod, design, modules, n = build_inputs()
    results = {}
    for m in sorted(rna_mod):
        subset = {mm: rna_mod[mm] for mm in rna_mod if mm != m}
        r = compute_zpg(subset, prot_mod[m], design, n_perms=n_perms, seed=42, cv="loocv")
        results[m] = r
        print(f"  {m}: zPG={r['zPG_rank']:+.4f} p={r['p_val']:.5f}", flush=True)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
