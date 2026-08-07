"""
cordiag.m1 — shared core: M1 model family (stratum-conditioned Ridge)

M1 = stratum-conditioned Ridge model family, shared modeling primitives
for zPG (Layer 1) and TG (Layer 3) diagnostics:

  1. Train-side stratum de-meaning (subtract stratum mean from y and X)
  2. StandardScaler (X_ctr, y_ctr) → RidgeCV(alphas, cv=min(5, n_tr))
     when n_tr >= 20, else Ridge(alpha=1.0)
  3. Test sample centered by stratum mean → predict → inverse-transform → add back
  4. Test sample stratum not in training set → two explicit modes
     (see m1_loocv unseen_stratum parameter)
  5. When groups provided: all same-group samples excluded (patient-level LOOCV)

Origin
------
- TG side: tgdecomp v0.2.0 core.py
- zPG side: code/simulation/metrics_v3.py

bit-identical invariants (package specification, any refactor must preserve)
--------------------------------------------------------------------
1. md5-derived seed chain (pair_seed, +10000 offset, rep offset) and default_rng
   call order — derive_seed / subsample_seed encapsulate; callers (tg.py) pass params
2. matched-subsample threshold chain: train_size>=8 / n_target-train_size>=3 /
   len(mse_vals)<3 fallback — located in tg.py, not reimplemented in m1.py
3. RidgeCV cv=min(5, n_tr) with n_tr>=20 boundary; fallback Ridge(alpha=1.0)
4. StandardScaler application order: first stratum de-mean, then fit scaler
5. mse_stratum_b < 1e-10 / q2>=0 estimability gate — enforced by callers

Behavioral differences (explicit caller choice, no silent unification)
----------------------------------------------------------------------
unseen/singleton stratum handling (m1_loocv unseen_stratum parameter, required):
  - 'fallback' (TG semantics): test stratum not in train → global train mean fallback
  - 'skip'    (zPG semantics): same case → prediction NaN (skip, excluded from MSE/edf)
m1.py never auto-infers — zpg.py / tg.py explicitly specify at each call.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

__all__ = [
    'derive_seed',
    'subsample_seed',
    'm0_stratum_means_loocv',
    'm1_loocv',
    'm1_train_test',
    'ridge_edf',
    'within_stratum_permute',
    'empirical_p',
    'zscore_stat',
]

SEED_OFFSET_CROSS = 10000


def derive_seed(key_str: str) -> int:
    """md5-derived per-pair RNG seed (TG family semantics). Returns int in [0, 2**31)."""
    return int(hashlib.md5(key_str.encode()).hexdigest(), 16) % 2**31


def subsample_seed(base_seed: Optional[int], rep: int, offset: int = 0) -> int:
    """Per-rep seed on the matched-subsample chain (TG family semantics)."""
    base = base_seed if base_seed is not None else 42
    return base + rep + offset


def _spearmanr(x, y) -> Tuple[float, float]:
    """Spearman correlation using scipy.stats.spearmanr."""
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    from scipy.stats import spearmanr
    rho, pval = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(pval)


def m0_stratum_means_loocv(
    P: np.ndarray, strata: np.ndarray, groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Leave-one-out stratum means (M0 baseline, sole Q2 denominator source)."""
    strata = np.asarray(strata)
    n = len(P)
    means = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        s = strata[i]
        same = np.where(strata == s)[0]
        if groups is not None:
            same_group = np.where(groups == groups[i])[0]
            others = same[~np.isin(same, same_group)]
        else:
            others = same[same != i]
        if len(others) > 0:
            means[i] = np.mean(P[others]).astype(np.float64)
        else:
            if groups is not None:
                others = np.array([j for j in range(n) if groups[j] != groups[i]], dtype=int)
            else:
                others = np.array([j for j in range(n) if j != i], dtype=int)
            means[i] = np.mean(P[others]).astype(np.float64)
    mse = float(np.nanmean((means - P) ** 2))
    return means, mse


def m1_loocv(
    P, X, strata, cv_alphas, eval_indices=None, fixed_alpha=None, groups=None,
    *, unseen_stratum,
) -> Tuple[np.ndarray, float, float, float]:
    """LOOCV with stratum-conditioned Ridge (M1 model)."""
    if unseen_stratum not in ('fallback', 'skip'):
        raise ValueError(
            "unseen_stratum must be 'fallback' (TG) or 'skip' (zPG), "
            f"got {unseen_stratum!r}"
        )
    strata = np.asarray(strata)
    n = len(P)
    if eval_indices is None:
        eval_indices = np.arange(n)
    n_eval = len(eval_indices)
    preds = np.full(n_eval, np.nan, dtype=np.float64)
    edf_list, alpha_list = [], []
    for fold_idx in range(n_eval):
        i = int(eval_indices[fold_idx])
        if groups is not None:
            tr = [j for j in range(n) if groups[j] != groups[i]]
        else:
            tr = [j for j in range(n) if j != i]
        if len(tr) < 3:
            continue
        X_tr, P_tr = X[tr].astype(np.float64), P[tr].astype(np.float64)
        strata_tr = strata[tr]
        s_test = strata[i]
        y_ctr, X_ctr = P_tr.copy(), X_tr.copy()
        stratum_means_y, stratum_means_X = {}, {}
        for s in np.unique(strata_tr):
            idx = np.where(strata_tr == s)[0]
            if len(idx) > 0:
                my = np.mean(P_tr[idx]).astype(np.float64)
                mX = np.mean(X_tr[idx], axis=0).astype(np.float64)
                stratum_means_y[s], stratum_means_X[s] = my, mX
                y_ctr[idx] -= my
                X_ctr[idx] -= mX
        if s_test in stratum_means_y:
            test_mean_y = stratum_means_y[s_test]
            test_mean_X = stratum_means_X[s_test]
        elif unseen_stratum == 'fallback':
            test_mean_y = np.mean(P_tr).astype(np.float64)
            test_mean_X = np.mean(X_tr, axis=0).astype(np.float64)
        else:
            preds[fold_idx] = np.nan
            continue
        y_test_ctr = P[i].astype(np.float64) - test_mean_y
        X_test_ctr = X[i].astype(np.float64) - test_mean_X
        try:
            sX = StandardScaler().fit(X_ctr)
            sy = StandardScaler().fit(y_ctr.reshape(-1, 1))
            n_tr = len(tr)
            if fixed_alpha is not None:
                alpha = float(fixed_alpha)
                m = Ridge(alpha=alpha).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
            elif n_tr >= 20:
                try:
                    m = RidgeCV(alphas=cv_alphas, cv=min(5, n_tr)).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
                    alpha = m.alpha_
                except Exception:
                    m = Ridge(alpha=1.0).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
                    alpha = 1.0
            else:
                alpha = 1.0
                m = Ridge(alpha=alpha).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
            X_scaled = sX.transform(X_ctr).astype(np.float64)
            edf_list.append(ridge_edf(X_scaled, alpha))
            alpha_list.append(alpha)
            pred_ctr = sy.inverse_transform(m.predict(sX.transform(X_test_ctr.reshape(1, -1))).reshape(-1, 1)).ravel()[0]
            preds[fold_idx] = pred_ctr + test_mean_y
        except Exception:
            preds[fold_idx] = np.nan
    valid = ~np.isnan(preds)
    mse = float(np.nanmean((preds[valid] - P[eval_indices[valid]]) ** 2)) if valid.sum() >= 1 else float('nan')
    avg_edf = float(np.mean(edf_list)) if edf_list else 0.0
    avg_alpha = float(np.mean(alpha_list)) if alpha_list else 1.0
    return preds, mse, avg_edf, avg_alpha


def m1_train_test(P_train, X_train, strata_train, P_test, X_test, strata_test, cv_alphas):
    """Train M1 on source, predict on target (no retraining)."""
    strata_train = np.asarray(strata_train)
    strata_test = np.asarray(strata_test)
    n_train, n_test = len(P_train), len(P_test)
    y_ctr = P_train.copy().astype(np.float64)
    X_ctr = X_train.copy().astype(np.float64)
    stratum_means_y, stratum_means_X = {}, {}
    for s in np.unique(strata_train):
        idx = np.where(strata_train == s)[0]
        if len(idx) > 0:
            my = float(np.mean(P_train[idx]))
            mX = np.mean(X_train[idx], axis=0).astype(np.float64)
            stratum_means_y[s], stratum_means_X[s] = my, mX
            y_ctr[idx] -= my
            X_ctr[idx] -= mX
    try:
        sX = StandardScaler().fit(X_ctr)
        sy = StandardScaler().fit(y_ctr.reshape(-1, 1))
        if n_train >= 20:
            try:
                m = RidgeCV(alphas=cv_alphas, cv=min(5, n_train)).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
            except Exception:
                m = Ridge(alpha=1.0).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
        else:
            m = Ridge(alpha=1.0).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1, 1)).ravel())
    except Exception:
        return np.full(n_test, np.nan, dtype=np.float64)
    predictions = np.full(n_test, np.nan, dtype=np.float64)
    for j in range(n_test):
        s = strata_test[j] if j < len(strata_test) else ''
        if s in stratum_means_y:
            test_mean_y, test_mean_X = stratum_means_y[s], stratum_means_X[s]
        else:
            test_mean_y = float(np.mean(P_train))
            test_mean_X = np.mean(X_train, axis=0).astype(np.float64)
        y_te_ctr = P_test[j].astype(np.float64) - test_mean_y
        X_te_ctr = X_test[j].astype(np.float64) - test_mean_X
        try:
            pred_ctr = sy.inverse_transform(m.predict(sX.transform(X_te_ctr.reshape(1, -1))).reshape(-1, 1)).ravel()[0]
            predictions[j] = pred_ctr + test_mean_y
        except Exception:
            predictions[j] = np.nan
    return predictions


def ridge_edf(X_scaled: np.ndarray, alpha: float) -> float:
    """Ridge effective degrees of freedom: edf = sum_i sigma_i^2 / (sigma_i^2 + alpha)."""
    n, p = X_scaled.shape
    k = min(n, p)
    if k == 0:
        return 0.0
    try:
        U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        return float(np.sum(s[:k] ** 2 / (s[:k] ** 2 + alpha)))
    except np.linalg.LinAlgError:
        return float(np.trace(X_scaled @ X_scaled.T) / (np.trace(X_scaled @ X_scaled.T) + alpha * n))


def within_stratum_permute(P, strata, rng):
    """Within-stratum permutation primitive (zPG semantics)."""
    strata = np.asarray(strata)
    P_perm = P.copy()
    for s in _strata_unique_order(strata):
        s_idx = np.where(strata == s)[0]
        if len(s_idx) >= 2:
            P_perm[s_idx] = P[s_idx[rng.permutation(len(s_idx))]]
    return P_perm


def empirical_p(null_vals, obs, two_sided=True, denominator=None):
    """Empirical p-value (permutation null vs observed). Returns float in (0, 1]."""
    null_arr = np.asarray(null_vals, dtype=np.float64)
    null_arr = null_arr[~np.isnan(null_arr)]
    cnt = int(np.sum(np.abs(null_arr) >= np.abs(obs))) if two_sided else int(np.sum(null_arr >= obs))
    denom = (len(null_arr) + 1) if denominator is None else (denominator + 1)
    return float((cnt + 1) / denom)


def zscore_stat(obs, null_vals):
    """Z-score statistic: (obs - mean(null)) / std(null). Returns float or NaN."""
    null_arr = np.asarray(null_vals, dtype=np.float64)
    null_arr = null_arr[~np.isnan(null_arr)]
    if len(null_arr) < 1:
        return float('nan')
    mu = float(np.mean(null_arr))
    sd = float(np.std(null_arr))
    if np.isnan(sd) or sd <= 1e-10:
        return float('nan')
    return float((obs - mu) / sd)


def _strata_unique_order(strata):
    """Unique stratum labels in first-appearance order (pandas Series.unique semantics)."""
    _, first_idx = np.unique(strata, return_index=True)
    return strata[np.sort(first_idx)]
