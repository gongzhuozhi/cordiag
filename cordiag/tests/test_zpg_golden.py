"""zPG golden regression - structure + known-decision checks.

Goldens: output/tables/tcga_validation.csv (TCGA-TGCT n=53, 8 modules, M4/M7 GO)
         output/tables/zpg_bulk_v2.csv (10-mouse, 10 modules, 0/10 GO)
"""
import json, os, re, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, pytest

ZPG_GO_TOL = 0.1
TCGA_GO_MODULES = {"M4": 2.9869213288520022, "M7": 3.026976400801053}

def test_tcga_golden_structure(zpg_tcga_golden_df):
    df = zpg_tcga_golden_df
    mods = df.loc[df["module"].notna()]
    assert set(mods["module"]) == {f"M{i}" for i in range(8)}
    assert (mods["n_genes"] == 18).all()

def test_tcga_golden_known_GO(zpg_tcga_golden_df):
    df = zpg_tcga_golden_df.loc[zpg_tcga_golden_df["module"].notna()]
    for mod, zpg_expected in TCGA_GO_MODULES.items():
        row = df.loc[df["module"] == mod].iloc[0]
        assert abs(row["zPG"] - zpg_expected) < ZPG_GO_TOL
        assert row["GO"] is True or str(row["GO"]).lower() == "true"
        assert float(row["p_fdr"]) < 0.1
    assert (df["GO"].astype(str).eq("True")).sum() == 2

def test_zpg_bulk_negative_control(zpg_bulk_golden_df):
    df = zpg_bulk_golden_df
    assert len(df) == 10
    assert df["module"].nunique() == 10
    assert (df["significant_fdr"].astype(str) == "False").all()
    assert "decision" in df.columns
    assert "GRAY" in set(df["decision"]) and "NO_GO" in set(df["decision"])

def test_zpg_golden_fdr_consistency(zpg_tcga_golden_df):
    from statsmodels.stats.multitest import multipletests
    df = zpg_tcga_golden_df.loc[zpg_tcga_golden_df["module"].notna()]
    p = df["p_val"].astype(float).values
    q_recomputed = multipletests(p, method="fdr_bh")[1]
    assert np.max(np.abs(q_recomputed - df["p_fdr"].astype(float).values)) < 1e-6

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("CORDIAG_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
ZPG_REGRESSION_SCRIPT = TEST_DIR / "regression_zpg_golden.py"
ZPG_OUT_DIR = TEST_DIR / "_regression_output"
ZPG_BULK_RESULT_CSV = ZPG_OUT_DIR / "_zpg_bulk_result.csv"
ZPG_ENGINE_EQUIV_JSON = ZPG_OUT_DIR / "_zpg_engine_equiv.json"
ZPG_TCGA_RESULT_CSV = ZPG_OUT_DIR / "_zpg_tcga_result.csv"
ZPG_ENV_VERSIONS = ZPG_OUT_DIR / "_zpg_env_versions.txt"
ZPG_TIMEOUT_S = 25 * 60
ZPG_TCGA_TIMEOUT_S = 100 * 60
ZPG_BULK_DATA = (PROJECT_ROOT / "data" / "rna_tpm_10mice.csv", PROJECT_ROOT / "data" / "protein_modules_10mice.csv")

def _zpg_scipy_ok():
    try:
        txt = ZPG_ENV_VERSIONS.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    m = re.search(r"scipy==(\d+)\.(\d+)\.(\d+)", txt)
    return m is not None and tuple(map(int, m.groups())) < (1, 18)

def _zpg_source_cache_valid():
    from _source_fingerprint import cache_valid
    return cache_valid(ZPG_OUT_DIR)

def test_zpg_bit_identical_regression():
    """Recompute zPG recipes with cordiag.zpg and check frozen claims."""
    if _zpg_source_cache_valid() and _zpg_scipy_ok():
        return
    env = dict(os.environ); env["PYTHONNOUSERSITE"] = "1"
    r = subprocess.run([sys.executable, "-u", str(ZPG_REGRESSION_SCRIPT), "--bulk", "--equivalence"],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=ZPG_TIMEOUT_S)
    if r.returncode != 0:
        pytest.fail(f"regression_zpg_golden.py failed (rc={r.returncode})\nstdout:\n{r.stdout[-4000:]}\nstderr:\n{r.stderr[-4000:]}")
    assert "RESULT: PASS" in r.stdout

def test_zpg_tcga_numeric_gate():
    pytest.skip("Opt-in TCGA gate (long runtime)")
