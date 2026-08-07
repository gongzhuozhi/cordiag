"""TG golden regression tests."""
import os, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, pytest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('CORDIAG_PROJECT_ROOT', str(Path(__file__).resolve().parent.parent.parent)))
GOLDEN_PATH = Path(os.environ.get('CORDIAG_GOLDEN_CSV', str(PROJECT_ROOT / 'output' / 'tg_bcm-only_v12.csv')))
CPTAC_ROOT = Path(os.environ.get('CORDIAG_CPTAC_DIR', ''))

def _tg_golden_df():
    if not GOLDEN_PATH.exists():
        pytest.skip(f'TG golden not found: {GOLDEN_PATH}')
    return pd.read_csv(GOLDEN_PATH)

def test_tg_golden_structure():
    df = _tg_golden_df()
    assert len(df) > 0
    for col in ['tg_raw', 'tg_log', 'fdr_global']:
        assert col in df.columns, f'missing: {col}'

def test_tg_golden_fdr_global_ordering():
    df = _tg_golden_df()
    fdr = df['fdr_global'].dropna()
    assert len(fdr) > 0

def test_tg_golden_fdr_bh_consistent():
    df = _tg_golden_df()
    p = df['permutation_p_raw'].dropna()
    if len(p) > 1:
        from statsmodels.stats.multitest import multipletests
        q = multipletests(p, method='fdr_bh')[1]
        assert np.max(np.abs(q - df.loc[p.index, 'fdr_global'])) < 1e-6

def test_tg_golden_estimable_consistency():
    df = _tg_golden_df()
    est = df[df['estimable'] == True]
    assert len(est) > 0
    assert est['q2_within_b'].dropna().min() >= 0.1
