"""TG calibration golden regression tests."""
import os, warnings
from pathlib import Path
import numpy as np, pandas as pd, pytest

PROJECT_ROOT = Path(os.environ.get('CORDIAG_PROJECT_ROOT', str(Path(__file__).resolve().parent.parent.parent)))
CALIBRATION_DIR = Path(os.environ.get('CORDIAG_CALIBRATION_DIR', str(PROJECT_ROOT / 'output' / 'calibration')))

@pytest.fixture(scope='session')
def calib_dir():
    if not CALIBRATION_DIR.exists():
        pytest.skip(f'Calibration directory not found: {CALIBRATION_DIR}')
    return CALIBRATION_DIR

def _load_csv(path, label):
    if not path.exists():
        pytest.skip(f'{label} not found: {path}')
    return pd.read_csv(path)

def test_calibration_theta_golden(calib_dir):
    df = _load_csv(calib_dir / 'tg_simulation_v10_thresholds.csv', 'thresholds')
    assert len(df) >= 1
    assert 'theta' in df.columns
    thetas = df['theta'].dropna().values
    assert np.all((thetas >= 0.9) & (thetas <= 1.3)), f'theta range: {thetas}'

def test_calibration_typeI_golden(calib_dir):
    df = _load_csv(calib_dir / 'tg_simulation_v10_eval.csv', 'eval')
    null = df[df['scenario_group'].isin(['null_identical', 'transportable'])]
    if len(null) > 0:
        type1 = null['type_i_p95'].dropna()
        assert np.all(type1 <= 0.10), f'type I exceeds 10%: {type1.values}'

def test_calibration_power_golden(calib_dir):
    df = _load_csv(calib_dir / 'tg_simulation_v10_power.csv', 'power')
    strong = df[df['delta_rho_mean'] > 0.15]
    if len(strong) > 0:
        power = strong['power_theta'].dropna()
        assert np.any(power >= 0.2), f'power too low: {power.values}'

def test_calibration_module_smoke():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from cordiag.calibration import generate_scenario, simulate_one_rep
            data = generate_scenario('NULL_IDENTICAL', n_per_condition=30, seed=42)
            assert 'rna_a' in data
    except ImportError:
        pytest.skip('cordiag.calibration not importable')
