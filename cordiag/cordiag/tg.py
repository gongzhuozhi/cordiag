"""cordiag.tg - Transportability Gap diagnostic module.

Layer 3 of the zPG framework: computing the loss in RNA-to-protein prediction
accuracy when transferring an M1 (stratum-conditioned Ridge) model from a
source condition to a target condition.

Core metrics:
  TG_log = log(MSE_source_to_target / MSE_within_target)  (PRIMARY)
  TG_raw = Q2_within_b - Q2_a_to_b                          (SECONDARY)

TG decomposition: TG_raw = TG_design (stratum shift) + TG_RNA (coupling change)

Bit-identical invariants preserved from reference implementation:
  - md5-derived per-pair seed chain
  - matched-subsample thresholds (train_size>=8, n_target-train_size>=3)
  - RidgeCV cv=min(5,n_tr) with n_tr>=20 boundary
  - StandardScaler application order (stratum-mean subtraction first)
  - mse_stratum_b < 1e-10 and q2>=0 estimability gates
  - FDR: per-protein before global; NaN->1.0
"""
import contextlib, functools, hashlib, time
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings

from cordiag.m1 import (m0_stratum_means_loocv as _compute_stratum_means_loocv,
    m1_loocv as _m1_loocv_impl, m1_train_test as _m1_train_test,
    ridge_edf as _ridge_edf, _spearmanr, derive_seed, subsample_seed)

SEED_OFFSET_CROSS = 10000

@contextlib.contextmanager
def _suppress_internal_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', FutureWarning); yield

def _quiet_internal(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        with _suppress_internal_warnings(): return func(*args,**kwargs)
    return wrapper

def _m1_loocv(P,X,strata,cv_alphas,eval_indices=None,fixed_alpha=None,groups=None):
    return _m1_loocv_impl(P,X,strata,cv_alphas,eval_indices=eval_indices,
        fixed_alpha=fixed_alpha,groups=groups,unseen_stratum='fallback')

@dataclass
class TGResult:
    """Transportability Gap result for one protein and source-target pair."""
    protein: str = ''; source_condition: str = ''; target_condition: str = ''; batch: str = ''
    n_source: int = 0; n_target: int = 0; size_ratio: float = 1.0
    size_ratio_directional: float = 1.0; size_ratio_symmetric: float = 1.0
    q2_within_b: float = float('nan'); q2_a_to_b: float = float('nan')
    tg_raw: float = float('nan'); tg_relative: float = float('nan')
    q2_crossed: float = float('nan'); tg_design: float = float('nan')
    tg_rna: float = float('nan'); tg_design_fraction: float = float('nan')
    mse_stratum_b: float = float('nan')
    permutation_p_raw: float = 1.0; permutation_p_design: float = 1.0; permutation_p_rna: float = 1.0
    interaction_pvalue: float = 1.0; ztg: float = 0.0; tg_log: float = 0.0
    ci_lower: float = 0.0; ci_upper: float = 0.0
    fdr_per_protein: float = 1.0; fdr_global: float = 1.0
    estimable: bool = True; weak_baseline: bool = False; asymmetric: bool = False
    ridge_edf: float = 0.0; cramers_v: float = 0.0; js_divergence: float = 0.0
    interpretation_primary: str = ''; interpretation_secondary: str = ''; interpretation_text: str = ''
    seed: int = 42; n_perms: int = 100; n_bootstrap: int = 1000
    group_aware: bool = False; ridge_alpha_within_b: float = 1.0
    cv_mode: str = 'loocv'; n_subsamples: int = 0
    verdict: str = ''; decomposition_code: str = ''; rna_coupling_change: float = 0.0
    alpha: float = 1.0; rep: int = 0

@dataclass
class TGMatrixResult:
    """Aggregated TG results across all condition pairs for one protein."""
    protein: str; conditions: List[str]
    matrix: Dict; n_valid_pairs: int; n_transportable: int; n_non_transportable: int
    mean_tg_raw: float; max_tg_raw: float; worst_pair: Tuple

@dataclass
class TGEnsembleResult:
    """Aggregated TG across all proteins."""
    proteins: List[str]; conditions: List[str]; per_protein: Dict
    mean_matrix: np.ndarray; transportable_fraction_matrix: np.ndarray
    n_total_pairs: int; n_estimable_pairs: int

def _compute_cramers_v(labels1,labels2):
    """Cramer's V between two categorical label arrays. V in [0,1]."""
    from scipy.stats import chi2_contingency
    uniq1=np.unique(labels1); uniq2=np.unique(labels2)
    n1,n2=len(uniq1),len(uniq2)
    if n1<=1 or n2<=1: return 0.0
    table=np.zeros((n1,n2),dtype=np.float64)
    for i,u1 in enumerate(uniq1):
        for j,u2 in enumerate(uniq2): table[i,j]=float(np.sum((labels1==u1)&(labels2==u2)))
    if np.any(table.sum(axis=1)==0) or np.any(table.sum(axis=0)==0): return 0.0
    try: chi2,_,dof,_=chi2_contingency(table)
    except Exception: return 0.0
    n=len(labels1); k=min(n1,n2)-1
    if k<=0 or chi2<=0 or n==0: return 0.0
    return float(min(np.sqrt(chi2/(n*k)),1.0))

def _compute_js_divergence(dist1,dist2):
    """Jensen-Shannon divergence between two distributions."""
    bins=np.linspace(min(dist1.min(),dist2.min()),max(dist1.max(),dist2.max()),50)
    h1,_=np.histogram(dist1,bins=bins,density=True)
    h2,_=np.histogram(dist2,bins=bins,density=True)
    h1=np.maximum(h1,1e-10); h2=np.maximum(h2,1e-10)
    m=(h1+h2)/2
    return float(0.5*np.sum(h1*np.log(h1/m))+0.5*np.sum(h2*np.log(h2/m)))

def _compute_q2_within_matched(protein,rna,strata,train_size,cv_alphas,n_subsamples=100,seed=42):
    """Q2_within via matched-subsample CV (LOOCV fallback for small n)."""
    n=len(protein); mse_vals,edf_vals,alpha_vals=[],[],[]
    rng=np.random.default_rng(seed)
    for rep in range(n_subsamples):
        tr=rng.choice(n,size=train_size,replace=False)
        te=np.array([j for j in range(n) if j not in tr])
        if len(te)<3: continue
        _,mse_tr,edf_tr,alpha_tr=_m1_loocv(protein[tr],rna[tr],strata[tr],cv_alphas)
        if np.isnan(mse_tr): continue
        preds=_m1_train_test(protein[tr],rna[tr],strata[tr],protein[te],rna[te],strata[te],cv_alphas)
        valid=~np.isnan(preds)
        if valid.sum()<3: continue
        mse_vals.append(float(np.mean((preds[valid]-protein[te][valid])**2)))
        edf_vals.append(edf_tr); alpha_vals.append(alpha_tr)
    if len(mse_vals)<3: return float('nan'),float('nan'),float('nan'),float('nan'),float(alpha_vals[0]) if alpha_vals else 1.0
    mse_mean=float(np.mean(mse_vals))
    m0_mse=float(np.mean((protein-np.mean(protein))**2))
    q2=float(1.0-mse_mean/max(m0_mse,1e-10))
    return q2,mse_mean,float(np.std(mse_vals)),float(np.mean(edf_vals)),float(np.mean(alpha_vals))

def _compute_q2_crossed_matched(protein_pooled,rna_pooled,strata_pooled,eval_indices,train_size,mse_stratum_b,cv_alphas,n_subsamples=100,seed=42):
    """Q2_crossed via matched-subsample: train on pooled (a+b), eval on target."""
    n=len(protein_pooled); mse_vals=[]
    rng=np.random.default_rng(seed)
    for rep in range(n_subsamples):
        all_idx=np.arange(n); non_eval=np.setdiff1d(all_idx,eval_indices)
        tr=rng.choice(non_eval,size=min(train_size,len(non_eval)),replace=False)
        te=eval_indices
        if len(tr)<3 or len(te)<3: continue
        preds=_m1_train_test(protein_pooled[tr],rna_pooled[tr],strata_pooled[tr],protein_pooled[te],rna_pooled[te],strata_pooled[te],cv_alphas)
        valid=~np.isnan(preds)
        if valid.sum()<3: continue
        mse_vals.append(float(np.mean((preds[valid]-protein_pooled[te][valid])**2)))
    if len(mse_vals)<3: return float('nan'),float('nan'),float('nan')
    mse_mean=float(np.mean(mse_vals))
    q2=float(1.0-mse_mean/max(mse_stratum_b,1e-10))
    return q2,mse_mean,float(np.std(mse_vals))

def _compute_tg_pair(protein,rna,strata,design_df,source_cond,target_cond,cv_alphas,n_permutations=100,n_bootstrap=50,seed=None,batch='bcm_only',n_subsamples=20,groups=None):
    """Core TG computation for one protein and source-target condition pair."""
    seed_val=seed if seed is not None else 42
    cond_col=design_df.columns[0]
    idx_s=(design_df[cond_col]==source_cond).values
    idx_t=(design_df[cond_col]==target_cond).values
    n_s,n_t=idx_s.sum(),idx_t.sum()
    if n_s<3 or n_t<3: return None
    _,mse_stratum_b=_compute_stratum_means_loocv(protein[idx_t],strata[idx_t])
    train_size=min(n_s,max(n_t//2,n_t-10))
    if train_size>=8 and n_t-train_size>=3:
        q2_w,_,_,_,alpha_w=_compute_q2_within_matched(protein[idx_t],rna[idx_t],strata[idx_t],train_size,cv_alphas,n_subsamples=n_subsamples,seed=seed_val)
    else:
        _,mse_w,_,alpha_w=_m1_loocv(protein[idx_t],rna[idx_t],strata[idx_t],cv_alphas)
        q2_w=float(1.0-mse_w/max(mse_stratum_b,1e-10))
    preds_ab=_m1_train_test(protein[idx_s],rna[idx_s],strata[idx_s],protein[idx_t],rna[idx_t],strata[idx_t],cv_alphas)
    valid_ab=~np.isnan(preds_ab)
    q2_ab=float('nan'); tg_log=float('nan')
    if valid_ab.sum()>=1:
        mse_ab=float(np.mean((preds_ab[valid_ab]-protein[idx_t][valid_ab])**2))
        q2_ab=float(1.0-mse_ab/max(mse_stratum_b,1e-10))
        tg_log=float(np.log(max(mse_ab,1e-10)/max(mse_stratum_b,1e-10)))
    tg_raw=float(q2_w-q2_ab) if not (np.isnan(q2_w) or np.isnan(q2_ab)) else float('nan')
    key_str=f'protein_{source_cond}_{target_cond}_{batch}'
    pair_seed=derive_seed(key_str)
    result=TGResult(protein='',source_condition=source_cond,target_condition=target_cond,batch=batch,
        n_source=n_s,n_target=n_t,size_ratio=max(n_s,n_t)/max(min(n_s,n_t),1),
        size_ratio_directional=n_s/max(n_t,1),size_ratio_symmetric=max(n_s,n_t)/max(min(n_s,n_t),1),
        q2_within_b=q2_w,q2_a_to_b=q2_ab,tg_raw=tg_raw,tg_log=tg_log,
        mse_stratum_b=mse_stratum_b,seed=pair_seed,n_perms=n_permutations,n_bootstrap=n_bootstrap,
        cv_mode='matched_subsample' if train_size>=8 and n_t-train_size>=3 else 'loocv',
        n_subsamples=n_subsamples,ridge_alpha_within_b=alpha_w)
    result.estimable=(mse_stratum_b>=1e-10 and not np.isnan(tg_raw))
    result.cramers_v=_compute_cramers_v(design_df[cond_col].values,design_df['batch'].values) if 'batch' in design_df.columns else 0.0
    return result

def _permutation_test_tg(result,protein,rna,strata,design_df,cv_alphas,n_permutations=100,seed=None):
    """Within-condition restricted permutation test for TG."""
    if result is None or not result.estimable: return result
    seed_val=seed if seed is not None else 42
    rng=np.random.default_rng(seed_val)
    cond_col=design_df.columns[0]
    idx_s=(design_df[cond_col]==result.source_condition).values
    idx_t=(design_df[cond_col]==result.target_condition).values
    n_s,n_t=idx_s.sum(),idx_t.sum()
    pooled_prot=np.concatenate([protein[idx_s],protein[idx_t]])
    pooled_rna=np.vstack([rna[idx_s],rna[idx_t]])
    pooled_strata=np.concatenate([strata[idx_s],strata[idx_t]])
    null_tg=[]
    for p_idx in range(n_permutations):
        perm=rng.permutation(n_s+n_t)
        P_p=pooled_prot[perm]; R_p=pooled_rna[perm]; S_p=pooled_strata[perm]
        _,msb=_compute_stratum_means_loocv(P_p[n_s:],S_p[n_s:])
        if msb<1e-10: continue
        ts=min(n_s,max(n_t//2,n_t-10))
        if ts>=8 and n_t-ts>=3:
            q2_wp,_,_,_,_=_compute_q2_within_matched(P_p[n_s:],R_p[n_s:],S_p[n_s:],ts,cv_alphas,n_subsamples=10,seed=seed_val+p_idx)
        else:
            _,mwp,_,_=_m1_loocv(P_p[n_s:],R_p[n_s:],S_p[n_s:],cv_alphas)
            q2_wp=float(1.0-mwp/max(msb,1e-10))
        preds=_m1_train_test(P_p[:n_s],R_p[:n_s],S_p[:n_s],P_p[n_s:],R_p[n_s:],S_p[n_s:],cv_alphas)
        v=~np.isnan(preds)
        if v.sum()<3: continue
        msp=float(np.mean((preds[v]-P_p[n_s:][v])**2))
        q2_abp=float(1.0-msp/max(msb,1e-10))
        null_tg.append(float(q2_wp-q2_abp))
    if len(null_tg)>0:
        arr=np.array(null_tg); obs=result.tg_raw
        result.permutation_p_raw=float((np.sum(np.abs(arr)>=np.abs(obs))+1)/(len(arr)+1))
        sd=max(np.std(arr),1e-10)
        result.ztg=float((obs-np.mean(arr))/sd)
    return result

def _apply_tg_fdr(results,alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction across TG results."""
    from statsmodels.stats.multitest import multipletests
    valid=[(i,r) for i,r in enumerate(results) if r is not None and r.estimable and not np.isnan(r.permutation_p_raw)]
    if len(valid)<2: return results
    pvals=[r.permutation_p_raw for _,r in valid]
    _,qvals,_,_=multipletests(pvals,alpha=alpha,method='fdr_bh')
    for (_,r),q in zip(valid,qvals):
        r.fdr_global=float(q)
        if r.fdr_global<alpha and r.tg_raw>0:
            r.verdict='NON_TRANSPORTABLE' if r.tg_raw>0.05 else 'PARTIALLY_TRANSPORTABLE'
        else: r.verdict='TRANSPORTABLE' if r.tg_raw<=0 else 'PARTIALLY_TRANSPORTABLE'
    return results

def tg_results_to_dataframe(results):
    """Convert list of TGResult to a pandas DataFrame."""
    rows=[]
    for r in results:
        if r is None: continue
        d={}
        for f in ['protein','source_condition','target_condition','batch','n_source','n_target','size_ratio','q2_within_b','q2_a_to_b','tg_raw','tg_log','q2_crossed','tg_design','tg_rna','tg_design_fraction','mse_stratum_b','permutation_p_raw','permutation_p_design','permutation_p_rna','interaction_pvalue','ztg','ci_lower','ci_upper','fdr_per_protein','fdr_global','estimable','weak_baseline','asymmetric','ridge_edf','cramers_v','js_divergence','seed','n_perms','verdict','decomposition_code','rna_coupling_change']:
            d[f]=getattr(r,f,np.nan)
        rows.append(d)
    return pd.DataFrame(rows)

def compute_tg_pair(protein_series,rna_df,design_df,source_cond,target_cond,cv_alphas=None,n_permutations=100,n_bootstrap=50,seed=None,batch='bcm_only',n_subsamples=20,protein_name=None,groups=None):
    """Public API: compute TG for one protein and source-target condition pair."""
    if cv_alphas is None: cv_alphas=[0.01,0.1,1.0,10.0,100.0]
    protein=protein_series.values; rna=rna_df.values
    s=design_df['condition'].astype(str)+'_'+design_df['batch'].astype(str)
    result=_compute_tg_pair(protein,rna,s.values,design_df,source_cond,target_cond,cv_alphas,n_permutations,n_bootstrap,seed,batch,n_subsamples,groups)
    if result is not None and protein_name is not None: result.protein=protein_name
    if result is not None and result.estimable:
        result=_permutation_test_tg(result,protein,rna,s.values,design_df,cv_alphas,n_permutations,seed)
    return result

def compute_tg_matrix(protein_series,rna_df,design_df,conditions,cv_alphas=None,n_permutations=100,protein_name=None,batch='bcm_only',n_subsamples=20):
    """Compute TG for all ordered condition pairs of one protein."""
    results={}
    for sc in conditions:
        for tc in conditions:
            if sc==tc: continue
            r=compute_tg_pair(protein_series,rna_df,design_df,sc,tc,cv_alphas=cv_alphas,n_permutations=n_permutations,protein_name=protein_name,batch=batch,n_subsamples=n_subsamples)
            results[(sc,tc)]=r
    valid=[r for r in results.values() if r is not None and r.estimable]
    return TGMatrixResult(protein=protein_name or '',conditions=list(conditions),matrix=results,
        n_valid_pairs=len(valid),n_transportable=sum(1 for r in valid if r.verdict=='TRANSPORTABLE'),
        n_non_transportable=sum(1 for r in valid if r.verdict=='NON_TRANSPORTABLE'),
        mean_tg_raw=float(np.mean([r.tg_raw for r in valid])) if valid else 0.0,
        max_tg_raw=float(np.max([r.tg_raw for r in valid])) if valid else 0.0,
        worst_pair=max(results.keys(),key=lambda k: results[k].tg_raw if results[k] else -1) if valid else ('',''))

__all__=['TGResult','TGMatrixResult','TGEnsembleResult','compute_tg_pair','compute_tg_matrix',
    '_compute_tg_pair','_permutation_test_tg','_apply_tg_fdr','tg_results_to_dataframe',
    '_compute_q2_within_matched','_compute_q2_crossed_matched','_compute_cramers_v',
    '_suppress_internal_warnings','_quiet_internal']