# -*- coding: utf-8 -*-
"""
cordiag.calibration - TG simulation scenarios (8 ground-truth mechanisms).

Threshold calibration suite for the Transportability Gap (TG) diagnostic.
Derived from tgdecomp simulation.py; numerical behavior is byte-identical
to the reference implementation.

8 Scenarios: NULL_IDENTICAL, NULL_STRATUM_SHIFT, TRANSPORTABLE,
NON_TRANSPORTABLE_WEAK, NON_TRANSPORTABLE_STRONG, SAMPLE_ASYMMETRY,
HIGH_DIM, REALISTIC.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import contextlib, functools, warnings, os

from cordiag.tg import (_compute_stratum_means_loocv, _m1_loocv, _m1_train_test,
                     _compute_q2_within_matched, _compute_q2_crossed_matched, _spearmanr)

@contextlib.contextmanager
def _suppress_internal_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning); yield

def _quiet_internal(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _suppress_internal_warnings(): return func(*args, **kwargs)
    return wrapper

def _pearsonr(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    num, den = np.dot(xm, ym), np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
    return float(np.clip(num / den, -1.0, 1.0)) if den >= 1e-15 else 0.0

def _random_orthogonal(v, rng):
    p = len(v); u = rng.normal(0, 1, p).astype(np.float64)
    u -= np.dot(u, v) * v / max(np.dot(v, v), 1e-15)
    nu = np.linalg.norm(u)
    if nu > 1e-15: u /= nu
    return u

def _ensure_dir(path): os.makedirs(path, exist_ok=True)

SCENARIO_NAMES = ['NULL_IDENTICAL','NULL_STRATUM_SHIFT','TRANSPORTABLE',
    'NON_TRANSPORTABLE_WEAK','NON_TRANSPORTABLE_STRONG','SAMPLE_ASYMMETRY','HIGH_DIM','REALISTIC']

@dataclass
class TGSimConfig:
    scenario: str = 'NULL_IDENTICAL'; n_a: int = 30; n_b: int = 30
    p_features: int = 20; n_batches: int = 2
    effect_size: float = 0.5; delta_rho_target: float = 0.0; noise_scale: float = 0.5
    batch_imbalance: float = 0.0; seed: int = 42
    realized_rho_a: Optional[float] = None; realized_rho_b: Optional[float] = None
    realized_delta_rho: Optional[float] = None
    n_a_actual: Optional[int] = None; n_b_actual: Optional[int] = None

@dataclass
class TGSimResult:
    config: TGSimConfig; scenario: str = ''
    delta_rho_true: float = 0.0; rho_a_true: float = 0.0; rho_b_true: float = 0.0
    coupling_identical: bool = True
    q2_within_b: float = float('nan'); q2_a_to_b: float = float('nan')
    q2_crossed: float = float('nan'); tg_raw: float = float('nan')
    tg_design: float = float('nan'); tg_rna: float = float('nan')
    tg_relative: float = float('nan'); tg_design_fraction: float = float('nan')
    mse_stratum_b: float = float('nan')
    n_source: int = 0; n_target: int = 0; size_ratio: float = 1.0
    estimable: bool = True; ridge_alpha: float = 1.0; ridge_alpha_within_b: float = 1.0
    scenario_group: str = ''

_RIDGE_ALPHAS = np.logspace(-3, 3, 21)

def _generate_autoregressive_cov(p, rho=0.3):
    rows, cols = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    return np.float64(rho ** np.abs(rows - cols))

def _make_design_matrix(n_a, n_b, n_batches, batch_imbalance, rng):
    n_total = n_a + n_b
    conditions = np.array(['a'] * n_a + ['b'] * n_b)
    batches = np.empty(n_total, dtype=object)
    if batch_imbalance == 0.0:
        for cond_start, cond_end in [(0, n_a), (n_a, n_total)]:
            c_n = cond_end - cond_start; per_batch = max(1, c_n // n_batches)
            remaining = c_n
            for bi in range(n_batches):
                n_this = min(per_batch, remaining) if bi < n_batches - 1 else remaining
                start = cond_start + bi * per_batch; end = min(start + n_this, cond_end)
                batches[start:end] = f'B{bi}'; remaining -= n_this
    else:
        for ci, (cond_label, cond_start, cond_end) in enumerate([('a', 0, n_a), ('b', n_a, n_total)]):
            c_n = cond_end - cond_start; shift = batch_imbalance if cond_label == 'b' else 0.0
            batch_probs = np.ones(n_batches, dtype=np.float64)
            batch_probs += shift * np.arange(n_batches); batch_probs /= batch_probs.sum()
            assigned = rng.choice([f'B{b}' for b in range(n_batches)], size=c_n, p=batch_probs)
            batches[cond_start:cond_end] = assigned
    design = pd.DataFrame({'condition': conditions, 'batch': batches})
    idx_a = np.array([c == 'a' for c in conditions])
    idx_b = np.array([c == 'b' for c in conditions])
    return design, idx_a, idx_b

def _build_stratum_encoding(design):
    return pd.get_dummies(design[['condition', 'batch']], drop_first=True).values.astype(np.float64)

def _build_stratum_labels(design):
    return (design['condition'].astype(str) + '_' + design['batch'].astype(str)).values

def generate_scenario(scenario_name, n_per_condition=30, p_features=20, n_batches=2, seed=42):
    rng = np.random.default_rng(seed); sname = scenario_name.upper().replace('-', '_')
    if isinstance(n_per_condition, (tuple, list)): n_a, n_b = int(n_per_condition[0]), int(n_per_condition[1])
    else: n_a = n_b = int(n_per_condition)
    if sname == 'SAMPLE_ASYMMETRY': n_a, n_b = 50, 10
    config = TGSimConfig(scenario=sname, n_a=n_a, n_b=n_b, p_features=p_features, n_batches=n_batches, seed=seed)
    batch_imbalance, delta_rho_target, coupling_strength, noise_scale = 0.0, 0.0, 1.0, 0.3
    if sname == 'REALISTIC': p_features = max(p_features, 30); coupling_strength = 0.4; noise_scale = 0.6
    if sname == 'NON_TRANSPORTABLE_WEAK': delta_rho_target = 0.1
    if sname == 'NON_TRANSPORTABLE_STRONG': delta_rho_target = 0.3
    if sname == 'NULL_STRATUM_SHIFT': batch_imbalance = 1.5
    if sname == 'HIGH_DIM': p_features = max(p_features, 50); n_a = n_b = min(n_a, 20)
    config.p_features, config.n_a, config.n_b = p_features, n_a, n_b
    config.batch_imbalance, config.delta_rho_target, config.noise_scale = batch_imbalance, delta_rho_target, noise_scale
    design, idx_a, idx_b = _make_design_matrix(n_a, n_b, n_batches, batch_imbalance, rng)
    n_total = n_a + n_b
    Z = _build_stratum_encoding(design); beta_stratum = rng.normal(0, 0.5, Z.shape[1]).astype(np.float64)
    stratum_effect = Z @ beta_stratum
    cov_rna = _generate_autoregressive_cov(p_features, rho=0.3)
    L = np.linalg.cholesky(cov_rna)
    rna_raw = (L @ rng.normal(0, 1, (p_features, n_total))).T.astype(np.float64)
    norm_base = coupling_strength / np.sqrt(p_features)
    beta_shared = rng.normal(0, norm_base, p_features).astype(np.float64) if sname != 'HIGH_DIM' else np.zeros(p_features, dtype=np.float64)
    if sname == 'HIGH_DIM': beta_shared[:3] = rng.normal(0, coupling_strength, 3)
    coupling_identical = True
    if sname in ('NON_TRANSPORTABLE_WEAK', 'NON_TRANSPORTABLE_STRONG'):
        coupling_identical = False
        norm_beta = max(np.linalg.norm(beta_shared), 1e-15)
        v = beta_shared / norm_beta; u = _random_orthogonal(beta_shared, rng)
        theta = np.pi * delta_rho_target; beta_b = norm_beta * (np.cos(theta) * v + np.sin(theta) * u)
    else: beta_b = beta_shared.copy()
    beta_a = beta_shared.copy()
    rna_effect_a = rna_raw[idx_a] @ beta_a; rna_effect_b = rna_raw[idx_b] @ beta_b
    noise_a = rng.normal(0, noise_scale, n_a).astype(np.float64)
    noise_b = rng.normal(0, noise_scale, n_b).astype(np.float64)
    protein_a = stratum_effect[idx_a] + rna_effect_a + noise_a
    protein_b = stratum_effect[idx_b] + rna_effect_b + noise_b
    def _semi_partial_r(rna_eff, protein, Z_cond):
        try:
            beta_rna = np.linalg.lstsq(Z_cond, rna_eff, rcond=None)[0]
            beta_prot = np.linalg.lstsq(Z_cond, protein, rcond=None)[0]
            rna_resid = rna_eff - Z_cond @ beta_rna; prot_resid = protein - Z_cond @ beta_prot
            return _pearsonr(rna_resid, prot_resid)
        except np.linalg.LinAlgError: return _pearsonr(rna_eff, protein)
    Za, Zb = Z[idx_a], Z[idx_b]
    rho_a, rho_b = abs(_semi_partial_r(rna_effect_a, protein_a, Za)), abs(_semi_partial_r(rna_effect_b, protein_b, Zb))
    config.realized_rho_a, config.realized_rho_b = rho_a, rho_b
    config.realized_delta_rho = abs(rho_a - rho_b)
    design_a, design_b = design.iloc[idx_a].reset_index(drop=True), design.iloc[idx_b].reset_index(drop=True)
    ground_truth = {'scenario': sname, 'coupling_identical': coupling_identical,
        'delta_rho_target': delta_rho_target, 'delta_rho_realized': float(abs(rho_a - rho_b)),
        'rho_a': float(rho_a), 'rho_b': float(rho_b), 'beta_a': beta_a, 'beta_b': beta_b,
        'n_a': n_a, 'n_b': n_b, 'p_features': p_features, 'n_batches': n_batches,
        'batch_imbalance': batch_imbalance, 'noise_scale': noise_scale,
        'coupling_strength': coupling_strength, 'seed': seed}
    return {'rna_a': rna_raw[idx_a], 'rna_b': rna_raw[idx_b], 'protein_a': protein_a,
        'protein_b': protein_b, 'design_a': design_a, 'design_b': design_b,
        'design_combined': design, 'Z_combined': Z, 'idx_a': idx_a, 'idx_b': idx_b,
        'config': config, 'ground_truth': ground_truth}

def _compute_q2_components(rna_a, rna_b, protein_a, protein_b, design_a, design_b, design_combined, n_subsamples=100):
    n_a, n_b = len(protein_a), len(protein_b)
    result = TGSimResult(config=None); result.n_source, result.n_target = n_a, n_b
    result.size_ratio = max(n_a, n_b) / max(min(n_a, n_b), 1)
    strata_a = (design_a['condition'].astype(str) + '_' + design_a['batch'].astype(str)).values
    strata_b = (design_b['condition'].astype(str) + '_' + design_b['batch'].astype(str)).values
    strata_pooled = (design_combined['condition'].astype(str) + '_' + design_combined['batch'].astype(str)).values
    _, mse_stratum_b = _compute_stratum_means_loocv(protein_b, strata_b)
    result.mse_stratum_b = float(mse_stratum_b)
    if mse_stratum_b < 1e-10:
        result.q2_within_b = result.q2_a_to_b = result.q2_crossed = 0.0
        result.tg_raw = result.tg_design = result.tg_rna = 0.0; result.tg_design_fraction = 0.5
        return result
    train_size = min(n_a, max(n_b // 2, n_b - 10)); sim_seed = 42
    if train_size >= 8 and n_b - train_size >= 3:
        q2_w, _, _, _, within_alpha = _compute_q2_within_matched(protein_b, rna_b, strata_b, train_size, _RIDGE_ALPHAS.tolist(), n_subsamples=n_subsamples, seed=sim_seed)
    else:
        _, mse_w, _, within_alpha = _m1_loocv(protein_b, rna_b, strata_b, _RIDGE_ALPHAS.tolist())
        q2_w = float(1.0 - mse_w / mse_stratum_b)
    result.q2_within_b = float(q2_w) if not np.isnan(q2_w) else 0.0
    result.ridge_alpha_within_b = float(within_alpha)
    preds_ab = _m1_train_test(protein_a, rna_a, strata_a, protein_b, rna_b, strata_b, _RIDGE_ALPHAS.tolist())
    valid_ab = ~np.isnan(preds_ab)
    if valid_ab.sum() >= 1:
        result.q2_a_to_b = float(1.0 - float(np.mean((preds_ab[valid_ab] - protein_b[valid_ab])**2)) / mse_stratum_b)
    else: result.q2_a_to_b = float('nan')
    rna_pooled = np.vstack([rna_a, rna_b]); protein_pooled = np.concatenate([protein_a, protein_b])
    b_indices = np.arange(n_a, n_a + n_b)
    q2_cr, mse_crossed, _ = _compute_q2_crossed_matched(protein_pooled, rna_pooled, strata_pooled, b_indices, train_size, mse_stratum_b, _RIDGE_ALPHAS.tolist(), n_subsamples=n_subsamples, seed=sim_seed + 1)
    result.q2_crossed = float(q2_cr) if not np.isnan(q2_cr) else 0.0
    result.tg_raw = result.q2_within_b - result.q2_a_to_b
    result.tg_design = result.q2_within_b - result.q2_crossed
    result.tg_rna = result.q2_crossed - result.q2_a_to_b
    result.tg_relative = result.tg_raw / max(abs(result.q2_within_b), 0.01)
    result.tg_design_fraction = result.tg_design / max(abs(result.tg_raw), 1e-10)
    return result

@_quiet_internal
def simulate_one_rep(scenario_name, n_per_condition=30, p_features=20, n_batches=2, seed=42, n_subsamples=100):
    data = generate_scenario(scenario_name, n_per_condition, p_features, n_batches, seed)
    result = _compute_q2_components(data['rna_a'], data['rna_b'], data['protein_a'], data['protein_b'], data['design_a'], data['design_b'], data['design_combined'], n_subsamples=n_subsamples)
    gt = data['ground_truth']; result.config = data['config']
    result.scenario = gt['scenario']; result.coupling_identical = gt['coupling_identical']
    result.delta_rho_true = gt['delta_rho_realized']; result.rho_a_true = gt['rho_a']; result.rho_b_true = gt['rho_b']
    sname = gt['scenario']
    if sname == 'NULL_IDENTICAL': result.scenario_group = 'null_identical'
    elif sname == 'NULL_STRATUM_SHIFT': result.scenario_group = 'null_stratum_shift'
    elif sname == 'TRANSPORTABLE': result.scenario_group = 'transportable'
    elif sname in ('NON_TRANSPORTABLE_WEAK', 'NON_TRANSPORTABLE_STRONG'): result.scenario_group = 'non_transportable'
    elif sname == 'SAMPLE_ASYMMETRY': result.scenario_group = 'asymmetry'
    elif sname == 'HIGH_DIM': result.scenario_group = 'high_dim'
    elif sname == 'REALISTIC': result.scenario_group = 'realistic'
    return result

def summarize_reps(results):
    rows = []
    for i, r in enumerate(results):
        cfg = r.config
        rows.append({'rep': i, 'scenario': r.scenario, 'scenario_group': r.scenario_group,
            'n_a': r.n_source, 'n_b': r.n_target, 'n_total': r.n_source + r.n_target,
            'size_ratio': r.size_ratio, 'p_features': cfg.p_features if cfg else 20,
            'n_batches': cfg.n_batches if cfg else 2, 'tg_raw': r.tg_raw,
            'tg_design': r.tg_design, 'tg_rna': r.tg_rna, 'tg_relative': r.tg_relative,
            'tg_design_fraction': r.tg_design_fraction, 'q2_within_b': r.q2_within_b,
            'q2_a_to_b': r.q2_a_to_b, 'q2_crossed': r.q2_crossed,
            'mse_stratum_b': r.mse_stratum_b, 'delta_rho_true': r.delta_rho_true,
            'rho_a_true': r.rho_a_true, 'rho_b_true': r.rho_b_true,
            'coupling_identical': r.coupling_identical, 'ridge_alpha': r.ridge_alpha,
            'ridge_alpha_within_b': r.ridge_alpha_within_b, 'estimable': r.estimable,
            'seed': cfg.seed if cfg else 0})
    return pd.DataFrame(rows)

def evaluate_simulation(summary_df):
    group_cols = ['scenario', 'scenario_group', 'n_total']
    for c in ['n_a', 'n_b', 'p_features', 'n_batches']:
        if c in summary_df.columns: group_cols.append(c)
    results = []
    for group_vals, grp in summary_df.groupby(group_cols, sort=False):
        if not isinstance(group_vals, tuple): group_vals = (group_vals,)
        row = dict(zip(group_cols, group_vals))
        tg_raw = grp['tg_raw'].dropna().values
        row['n_reps'] = len(tg_raw)
        if len(tg_raw) > 0:
            row['tg_raw_mean'] = float(np.mean(tg_raw)); row['tg_raw_std'] = float(np.std(tg_raw))
            row['tg_raw_p50'] = float(np.percentile(tg_raw, 50)); row['tg_raw_p90'] = float(np.percentile(tg_raw, 90))
            row['tg_raw_p95'] = float(np.percentile(tg_raw, 95)); row['tg_raw_p99'] = float(np.percentile(tg_raw, 99))
            row['tg_raw_max'] = float(np.max(tg_raw)); row['tg_design_mean'] = float(np.mean(grp['tg_design'].dropna()))
            row['tg_rna_mean'] = float(np.mean(grp['tg_rna'].dropna()))
            row['delta_rho_mean'] = float(np.mean(grp['delta_rho_true'].dropna()))
        if row.get('scenario_group') in ('non_transportable',):
            n_total_reps = len(tg_raw); n_pos = int(np.sum(tg_raw > 0))
            row['detection_rate'] = n_pos / max(n_total_reps, 1)
            row['practical_detection_005'] = int(np.sum(tg_raw > 0.05)) / max(n_total_reps, 1)
            row['practical_detection_010'] = int(np.sum(tg_raw > 0.10)) / max(n_total_reps, 1)
        if row.get('scenario_group') in ('null_identical', 'null_stratum_shift', 'transportable'):
            n_total_reps = len(tg_raw)
            row['type_i_005'] = int(np.sum(tg_raw > 0.05)) / max(n_total_reps, 1)
            row['type_i_010'] = int(np.sum(tg_raw > 0.10)) / max(n_total_reps, 1)
            row['type_i_p95'] = int(np.sum(tg_raw > row.get('tg_raw_p95', 0.05))) / max(n_total_reps, 1)
        results.append(row)
    return pd.DataFrame(results)

def calibrate_tg_thresholds(summary_df, target_alpha=0.05):
    null_df = summary_df[summary_df['scenario'] == 'NULL_IDENTICAL'].copy()
    if len(null_df) == 0: null_df = summary_df[summary_df['scenario_group'].isin(['null_identical', 'transportable'])].copy()
    thresholds = []
    for n_total, grp in null_df.groupby('n_total', sort=True):
        tg_vals = grp['tg_raw'].dropna().values
        if len(tg_vals) < 3: continue
        theta_raw = float(np.percentile(tg_vals, 100 * (1 - target_alpha))); theta = max(0.05, theta_raw)
        thresholds.append({'n_total': n_total, 'n_a': int(grp['n_a'].iloc[0]) if 'n_a' in grp.columns else n_total // 2,
            'n_b': int(grp['n_b'].iloc[0]) if 'n_b' in grp.columns else n_total // 2,
            'theta_raw': theta_raw, 'theta': theta, 'theta_99': float(np.percentile(tg_vals, 99)),
            'tg_raw_mean_null': float(np.mean(tg_vals)), 'tg_raw_std_null': float(np.std(tg_vals)), 'n_reps': len(tg_vals)})
    return pd.DataFrame(thresholds)

def compute_power_curves(summary_df, threshold_df):
    theta_map = dict(zip(threshold_df['n_total'], threshold_df['theta'])); results = []
    for (n_total, scenario), grp in summary_df.groupby(['n_total', 'scenario'], sort=False):
        theta = theta_map.get(n_total, 0.05); tg_vals = grp['tg_raw'].dropna().values
        delta_rhos = grp['delta_rho_true'].dropna().values
        if len(tg_vals) == 0: continue
        n_total_reps = len(tg_vals)
        results.append({'n_total': n_total, 'scenario': scenario,
            'delta_rho_mean': float(np.mean(delta_rhos)), 'delta_rho_std': float(np.std(delta_rhos)),
            'power_theta': int(np.sum(tg_vals > theta)) / max(n_total_reps, 1),
            'power_theta_2x': int(np.sum(tg_vals > 2 * theta)) / max(n_total_reps, 1),
            'power_005': int(np.sum(tg_vals > 0.05)) / max(n_total_reps, 1),
            'tg_raw_mean': float(np.mean(tg_vals)), 'n_reps': n_total_reps})
    return pd.DataFrame(results)

DEFAULT_N_SAMPLES = (8, 11, 15, 20, 30, 50, 100)
CALIBRATION_SCENARIOS = ('NULL_IDENTICAL', 'NULL_STRATUM_SHIFT', 'TRANSPORTABLE',
    'NON_TRANSPORTABLE_WEAK', 'NON_TRANSPORTABLE_STRONG', 'SAMPLE_ASYMMETRY', 'HIGH_DIM', 'REALISTIC')

@_quiet_internal
def run_calibration(config=None):
    if config is None: config = {}
    n_samples_grid = config.get('n_samples_grid', DEFAULT_N_SAMPLES)
    n_reps = config.get('n_reps', 100); scenarios = config.get('scenarios', CALIBRATION_SCENARIOS)
    p_features = config.get('p_features', 20); n_batches = config.get('n_batches', 2)
    n_subsamples = config.get('n_subsamples', 100); seed = config.get('seed', 42)
    output_dir = config.get('output_dir', os.path.join(os.getcwd(), 'output'))
    _ensure_dir(output_dir)
    all_results = []; total_combos = len(scenarios) * len(n_samples_grid); combo_idx = 0
    for scenario in scenarios:
        for n_idx, n in enumerate(n_samples_grid):
            combo_idx += 1
            n_per_cond = (50, 10) if scenario == 'SAMPLE_ASYMMETRY' else ((min(20, n), min(20, n)) if scenario == 'HIGH_DIM' else (n, n))
            p_actual = 50 if scenario == 'HIGH_DIM' else p_features
            rep_results = []
            for rep in range(n_reps):
                rep_seed = seed + combo_idx * 1000 + rep * 7
                res = simulate_one_rep(scenario_name=scenario, n_per_condition=n_per_cond, p_features=p_actual, n_batches=n_batches, seed=rep_seed, n_subsamples=n_subsamples)
                rep_results.append(res)
            df_rep = summarize_reps(rep_results); df_rep['n_total'] = df_rep['n_a'] + df_rep['n_b']
            all_results.append(df_rep)
    summary_df = pd.concat(all_results, ignore_index=True)
    eval_df = evaluate_simulation(summary_df)
    threshold_df = calibrate_tg_thresholds(summary_df)
    power_df = compute_power_curves(summary_df, threshold_df)
    summary_df.to_csv(os.path.join(output_dir, 'tg_simulation_summary.csv'), index=False)
    eval_df.to_csv(os.path.join(output_dir, 'tg_simulation_eval.csv'), index=False)
    threshold_df.to_csv(os.path.join(output_dir, 'tg_simulation_thresholds.csv'), index=False)
    power_df.to_csv(os.path.join(output_dir, 'tg_simulation_power.csv'), index=False)
    return {'summary_df': summary_df, 'eval_df': eval_df, 'threshold_df': threshold_df, 'power_df': power_df}
