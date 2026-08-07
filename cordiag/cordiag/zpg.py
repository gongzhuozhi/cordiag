"""cordiag.zpg - z-score of Paired Gain (zPG) core module.

zPG metric: under condition x batch stratification, measures individual-level
RNA-to-protein predictive gain beyond stratum labels (cross-omics bridgeability).

Single source of truth: code/simulation/metrics_v3.py (BridgeOmics v3).
Decision logic: GO / INCONCLUSIVE / NO_GO three-tier framework.

Core invariants:
  1. H0 = no individual-level pairing information beyond stratum structure
  2. M1 stratum-conditioned Ridge model (unseen_stratum='skip')
  3. FDR via BH procedure (pure numpy fdr_bh)
  4. Seeds: global seed=42, reproducible across 16 parallel workers
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

_RNG = np.random.default_rng(42)
from .m1 import m1_loocv as _m1_loocv
_CV_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

def set_seed(seed=42):
    global _RNG; _RNG = np.random.default_rng(seed)

def _coerce_design(design, n, require_batch=True):
    if isinstance(design, pd.DataFrame):
        needed = ['condition','batch'] if require_batch else ['condition']
        missing = [c for c in needed if c not in design.columns]
        if missing: raise ValueError(f"design missing: {missing}")
        if len(design) != n: raise ValueError(f"rows {len(design)} vs samples {n}")
        return design
    if isinstance(design, (tuple,list)):
        if len(design) != 2: raise ValueError("design tuple must be (condition, batch)")
        cond = np.asarray(design[0]); batch = np.asarray(design[1])
        if len(cond) != n or len(batch) != n: raise ValueError(f"length mismatch")
        return pd.DataFrame({'condition':cond, 'batch':batch})
    raise TypeError("design must be DataFrame or (condition,batch) tuple")

def _loo_stratum_ridge(X, y, strata, mice=None):
    n = len(y) if mice is None else len(mice)
    preds, _mse, _edf, _alpha = _m1_loocv(
        P=np.asarray(y[:n],dtype=np.float64), X=np.asarray(X[:n],dtype=np.float64),
        strata=np.asarray(strata[:n]), cv_alphas=_CV_ALPHAS, fixed_alpha=None,
        groups=None, unseen_stratum='skip')
    truths = np.asarray(y[:n],dtype=np.float64).copy()
    return preds, truths

def _loo_fast(X, y, strata, mice):
    n = len(mice); preds = np.full(n, np.nan); truths = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        if len(tr) < 3: truths[i]=y[i]; continue
        strata_tr = strata.iloc[tr]; y_ctr, X_ctr = y[tr].copy(), X[tr].copy()
        for s in strata_tr.unique():
            idx = np.where(strata_tr == s)[0]
            if len(idx) > 0:
                y_ctr[idx] -= np.mean(y[tr][idx]); X_ctr[idx] -= np.mean(X[tr][idx], axis=0)
        s_test = strata.iloc[i]; s_tr_idx = np.where(strata_tr == s_test)[0]
        if len(s_tr_idx) == 0: truths[i]=y[i]; continue
        y_test_ctr = y[i] - np.mean(y[tr][s_tr_idx])
        X_test_ctr = X[i] - np.mean(X[tr][s_tr_idx], axis=0)
        try:
            sX = StandardScaler().fit(X_ctr); sy = StandardScaler().fit(y_ctr.reshape(-1,1))
            m = Ridge(alpha=1.0).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1,1)).ravel())
            preds[i] = sy.inverse_transform(m.predict(sX.transform(X_test_ctr.reshape(1,-1))).reshape(-1,1)).ravel()[0]
            preds[i] += np.mean(y[tr][s_tr_idx]); truths[i] = y[i]
        except Exception: truths[i] = y[i]
    return preds, truths

def _kfold_stratum_ridge(X, y, strata, n_folds=5, seed=42):
    rng = np.random.default_rng(seed); n = len(y)
    preds = np.full(n, np.nan); truths = np.zeros(n)
    indices = np.arange(n); rng.shuffle(indices)
    fold_assignments = np.zeros(n, dtype=int); fold_size = n // n_folds
    for f in range(n_folds):
        start = f * fold_size; end = start + fold_size if f < n_folds - 1 else n
        fold_assignments[indices[start:end]] = f
    for f in range(n_folds):
        test_idx = np.where(fold_assignments == f)[0]; train_idx = np.where(fold_assignments != f)[0]
        for i in test_idx:
            tr = [j for j in train_idx]
            if len(tr) < 3: truths[i]=y[i]; continue
            strata_tr = strata.iloc[tr]; y_ctr, X_ctr = y[tr].copy(), X[tr].copy()
            for s in strata_tr.unique():
                idx = np.where(strata_tr == s)[0]
                if len(idx)>0: y_ctr[idx]-=np.mean(y[tr][idx]); X_ctr[idx]-=np.mean(X[tr][idx],axis=0)
            s_test = strata.iloc[i]; s_tr_idx = np.where(strata_tr == s_test)[0]
            if len(s_tr_idx)==0: truths[i]=y[i]; continue
            y_test_ctr = y[i]-np.mean(y[tr][s_tr_idx]); X_test_ctr = X[i]-np.mean(X[tr][s_tr_idx],axis=0)
            try:
                sX = StandardScaler().fit(X_ctr); sy = StandardScaler().fit(y_ctr.reshape(-1,1))
                m = Ridge(alpha=1.0).fit(sX.transform(X_ctr), sy.transform(y_ctr.reshape(-1,1)).ravel())
                preds[i] = sy.inverse_transform(m.predict(sX.transform(X_test_ctr.reshape(1,-1))).reshape(-1,1)).ravel()[0]
                preds[i] += np.mean(y[tr][s_tr_idx]); truths[i]=y[i]
            except Exception: truths[i]=y[i]
    return preds, truths

def compute_rank_zPG(R_modules, P, design, n_perms, seed=None):
    design = _coerce_design(design, len(P))
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    n = len(P); strata = design['condition']+'_'+design['batch']
    unique_strata = strata.unique()
    X_modules = np.column_stack([R_modules[m] for m in sorted(R_modules.keys())])
    true_preds, true_truths = _loo_stratum_ridge(X_modules, P, strata, np.arange(n))
    valid = ~np.isnan(true_preds)
    stratum_preds = np.array([
        np.mean(P[[j for j in range(n) if j!=i and strata.iloc[j]==strata.iloc[i]]])
        if np.any([j!=i and strata.iloc[j]==strata.iloc[i] for j in range(n)])
        else np.mean(P[[j for j in range(n) if j!=i]]) for i in range(n)])
    mse_model = np.mean((true_preds[valid]-true_truths[valid])**2)
    mse_stratum = np.mean((stratum_preds[valid]-true_truths[valid])**2)
    Q2_obs = np.nan if mse_stratum<1e-10 else 1-mse_model/mse_stratum
    if valid.sum()<3: rho_obs=np.nan
    else: rho_obs,_ = spearmanr(true_preds[valid], true_truths[valid])
    perm_rhos, perm_Q2s = [], []
    for _ in range(n_perms):
        P_perm = P.copy()
        for s in unique_strata:
            s_idx = np.where(strata==s)[0]
            if len(s_idx)>=2: P_perm[s_idx]=P[s_idx[rng.permutation(len(s_idx))]]
        preds, truths = _loo_stratum_ridge(X_modules, P_perm, strata, np.arange(n))
        v = ~np.isnan(preds)
        if v.sum()>=3:
            perm_rhos.append(spearmanr(preds[v],truths[v])[0])
            mse_p = np.mean((preds[v]-truths[v])**2)
            perm_Q2s.append(np.nan if mse_stratum<1e-10 else 1-mse_p/mse_stratum)
    perm_rhos=np.array(perm_rhos); perm_Q2s=np.array(perm_Q2s)
    perm_std=np.std(perm_rhos)
    zPG_rank=np.nan if perm_std<1e-10 else (rho_obs-np.mean(perm_rhos))/perm_std
    if np.isnan(rho_obs) or len(perm_rhos)==0: p_val=np.nan
    else: p_val=(np.sum(perm_rhos>=rho_obs)+1)/(n_perms+1)
    q2_std=np.std(perm_Q2s)
    zPG_Q2=np.nan if q2_std<1e-10 or np.isnan(q2_std) else (Q2_obs-np.mean(perm_Q2s))/q2_std
    min_p=1.0
    for s in unique_strata:
        n_s=(strata==s).sum()
        if n_s>=2: min_p=min(min_p, 1.0/(math.factorial(n_s) if n_s<=10 else 1000))
    min_p=max(min_p, 1.0/(n_perms+1))
    return {'zPG_rank':zPG_rank,'zPG_Q2':zPG_Q2,'rho_obs':rho_obs,'Q2_obs':Q2_obs,
            'p_val':p_val,'perm_rho_mean':np.mean(perm_rhos),'perm_rho_std':np.std(perm_rhos),
            'min_achievable_p':min_p,'n_valid':valid.sum()}

def compute_rank_zPG_partial(R_modules, P, design, n_perms, seed=None, n_pcs=3):
    design=_coerce_design(design,len(P))
    rng=np.random.default_rng(seed) if seed is not None else _RNG
    n=len(P); strata=design['condition']+'_'+design['batch']
    unique_strata=strata.unique()
    X=np.column_stack([R_modules[m] for m in sorted(R_modules.keys())])
    X_scaled=StandardScaler().fit_transform(X)
    pca=PCA(n_components=min(n_pcs,X.shape[1],n-1)); X_pca=pca.fit_transform(X_scaled)
    cond_dummies=pd.get_dummies(design[['condition','batch']],drop_first=True).values
    X_resid=np.zeros_like(X_pca)
    for j in range(X_pca.shape[1]):
        beta=np.linalg.lstsq(cond_dummies,X_pca[:,j],rcond=None)[0]
        X_resid[:,j]=X_pca[:,j]-cond_dummies@beta
    beta_y=np.linalg.lstsq(cond_dummies,P,rcond=None)[0]; P_resid=P-cond_dummies@beta_y
    rna_combined=X_resid.mean(axis=1); rho_obs,_=spearmanr(rna_combined,P_resid)
    perm_rhos=[]
    for _ in range(n_perms):
        P_perm=P.copy()
        for s in unique_strata:
            s_idx=np.where(strata==s)[0]
            if len(s_idx)>=2: P_perm[s_idx]=P[s_idx[rng.permutation(len(s_idx))]]
        beta_yp=np.linalg.lstsq(cond_dummies,P_perm,rcond=None)[0]
        Pp_resid=P_perm-cond_dummies@beta_yp; rho_p,_=spearmanr(rna_combined,Pp_resid)
        if not np.isnan(rho_p): perm_rhos.append(rho_p)
    perm_rhos=np.array(perm_rhos); perm_std=np.std(perm_rhos)
    zPG_partial=np.nan if perm_std<1e-10 else (rho_obs-np.mean(perm_rhos))/perm_std
    if np.isnan(rho_obs) or len(perm_rhos)==0: p_val=np.nan
    else: p_val=(np.sum(perm_rhos>=rho_obs)+1)/(n_perms+1)
    min_p=1.0
    for s in unique_strata:
        n_s=(strata==s).sum()
        if n_s>=2: min_p=min(min_p,1.0/(math.factorial(n_s) if n_s<=10 else 1000))
    min_p=max(min_p,1.0/(n_perms+1))
    return {'zPG_partial':zPG_partial,'rho_obs':rho_obs,'p_val':p_val,
            'perm_rho_mean':np.mean(perm_rhos),'perm_rho_std':np.std(perm_rhos),
            'min_achievable_p':min_p,'n_pcs':n_pcs,'pca_variance_explained':pca.explained_variance_ratio_.sum()}

def _zpg_with_cv(R_modules,P,design,n_folds,n_perms,seed=42,actual_perms=None):
    design=_coerce_design(design,len(P)); rng=np.random.default_rng(seed)
    n=len(P); strata=design['condition'].astype(str)+'_'+design['batch'].astype(str)
    unique_strata=strata.unique()
    X_modules=np.column_stack([R_modules[m] for m in sorted(R_modules.keys())])
    if n_folds==n: true_preds,true_truths=_loo_fast(X_modules,P,strata,np.arange(n))
    else: true_preds,true_truths=_kfold_stratum_ridge(X_modules,P,strata,n_folds=n_folds,seed=seed)
    valid=~np.isnan(true_preds)
    if valid.sum()<3: return {'zPG_rank':np.nan,'rho_obs':np.nan,'p_val':np.nan,'n_valid':valid.sum(),'n_folds':n_folds}
    actual_perms=min(n_perms,20 if n_folds==n else 30)
    stratum_preds=np.array([
        np.mean(P[[j for j in range(n) if j!=i and strata.iloc[j]==strata.iloc[i]]])
        if np.any([j!=i and strata.iloc[j]==strata.iloc[i] for j in range(n)])
        else np.mean(P[[j for j in range(n) if j!=i]]) for i in range(n)])
    rho_obs,_=spearmanr(true_preds[valid],true_truths[valid])
    mse_model=np.mean((true_preds[valid]-true_truths[valid])**2)
    mse_stratum=np.mean((stratum_preds[valid]-true_truths[valid])**2)
    Q2_obs=np.nan if mse_stratum<1e-10 else 1-mse_model/mse_stratum
    perm_rhos,perm_Q2s=[],[]
    for _ in range(actual_perms):
        P_perm=P.copy()
        for s in unique_strata:
            s_idx=np.where(strata==s)[0]
            if len(s_idx)>=2: P_perm[s_idx]=P[s_idx[rng.permutation(len(s_idx))]]
        if n_folds==n: preds,truths=_loo_stratum_ridge(X_modules,P_perm,strata,np.arange(n))
        else: preds,truths=_kfold_stratum_ridge(X_modules,P_perm,strata,n_folds=n_folds,seed=rng.integers(0,2**31))
        v=~np.isnan(preds)
        if v.sum()>=3:
            perm_rhos.append(spearmanr(preds[v],truths[v])[0])
            mse_p=np.mean((preds[v]-truths[v])**2)
            perm_Q2s.append(np.nan if mse_stratum<1e-10 else 1-mse_p/mse_stratum)
    perm_rhos=np.array(perm_rhos); p_std=np.std(perm_rhos)
    zPG_rank=(rho_obs-np.mean(perm_rhos))/p_std if p_std>1e-10 else np.nan
    if np.isnan(rho_obs) or len(perm_rhos)==0: p_val=np.nan
    else: p_val=(np.sum(perm_rhos>=rho_obs)+1)/(actual_perms+1)
    q2_std=np.std(perm_Q2s)
    zPG_Q2=(Q2_obs-np.mean(perm_Q2s))/q2_std if q2_std>1e-10 and not np.isnan(q2_std) else np.nan
    return {'zPG_rank':zPG_rank,'zPG_Q2':zPG_Q2,'rho_obs':rho_obs,'Q2_obs':Q2_obs,
            'p_val':p_val,'n_valid':valid.sum(),'n_folds':n_folds,
            'perm_rho_mean':float(np.mean(perm_rhos)),'perm_rho_std':float(np.std(perm_rhos))}

def select_cv(n):
    if n<30: return ('loocv',None,1)
    if n<100: return ('kfold',10,3)
    return ('kfold',5,10)

def compute_zpg(R_modules,P,design,n_perms,seed=None,cv=None,repeats=None):
    design=_coerce_design(design,len(P)); n=len(P)
    if cv is None: method,n_folds,n_rep=select_cv(n)
    elif cv=='loocv': method,n_folds,n_rep='loocv',None,1
    else: method,n_folds,n_rep='kfold',int(cv),1
    if repeats is not None: n_rep=int(repeats)
    if method=='loocv':
        res=dict(compute_rank_zPG(R_modules,P,design,n_perms,seed=seed))
        res['cv']='loocv'; res['n_folds']=n; res['n_repeats']=1; return res
    if seed is None: base_seed=int(_RNG.integers(0,2**31))
    else: base_seed=int(seed)
    results=[_zpg_with_cv(R_modules,P,design,n_folds,n_perms=n_perms,seed=base_seed+i) for i in range(n_rep)]
    if n_rep>1:
        res=dict(results[0]); z_list=[r['zPG_rank'] for r in results]
        res['zPG_rank']=float(np.mean(z_list)); res['zPG_rank_repeats']=z_list
    else: res=dict(results[0])
    res['cv']=f'{n_folds}fold'; res['n_folds']=n_folds; res['n_repeats']=n_rep; return res

def fdr_bh(p_values):
    p=np.asarray(p_values,dtype=float); n=p.size
    if n==0: return p.copy()
    nan_mask=np.isnan(p)
    if nan_mask.any():
        out=np.full(n,np.nan); out[~nan_mask]=fdr_bh(p[~nan_mask]); return out
    order=np.argsort(p,kind='mergesort')
    q=p[order]*n/np.arange(1,n+1); q=np.minimum.accumulate(q[::-1])[::-1]; q=np.minimum(q,1.0)
    result=np.empty(n); result[order]=q; return result

def decide(zpg,p_fdr,n_per_condition,zpg_go=1.0,fdr_go=0.1,n_min=12):
    if zpg>zpg_go and p_fdr<fdr_go: return 'GO'
    if zpg>0 and n_per_condition<n_min: return 'INCONCLUSIVE'
    return 'NO_GO'

def decide_legacy(zpg,p_fdr,zpg_go=1.0,fdr_go=0.1):
    if zpg>zpg_go and p_fdr<fdr_go: return 'GO'
    elif zpg>0: return 'GRAY'
    else: return 'NO_GO'

def compute_ECI(R_modules,design,cond_pairs,n_bootstrap=200,seed=None):
    n=len(next(iter(R_modules.values()))) if R_modules else 0
    design=_coerce_design(design,n,require_batch=False)
    rng=np.random.default_rng(seed) if seed is not None else _RNG
    results={}
    for ca,cb in cond_pairs:
        idx_a=design['condition']==ca; idx_b=design['condition']==cb
        if idx_a.sum()<2 or idx_b.sum()<2: results[f'{ca}_vs_{cb}']={'ECI':np.nan,'ECI_ci':(np.nan,np.nan)}; continue
        cos_sims=[]
        for mod_name in sorted(R_modules.keys()):
            vals=R_modules[mod_name]; boot_cos=[]
            for _ in range(min(n_bootstrap,100)):
                ba=rng.choice(vals[idx_a],size=idx_a.sum(),replace=True)
                bb=rng.choice(vals[idx_b],size=idx_b.sum(),replace=True)
                mu_a=ba.mean(); mu_b=bb.mean()
                cos_sim=1.0 if mu_a*mu_b>0 else (-1.0 if mu_a*mu_b<0 else 0.0)
                boot_cos.append(cos_sim)
            cos_sims.append(np.mean(boot_cos))
        cos_sims_arr=np.array(cos_sims); eci=1.0-np.mean(cos_sims_arr)
        eci_std=np.std(cos_sims_arr)/max(np.sqrt(len(cos_sims_arr)),1)
        eci_ci=(max(0,eci-1.96*eci_std),min(1,eci+1.96*eci_std))
        results[f'{ca}_vs_{cb}']={'ECI':eci,'ECI_ci':eci_ci,'n_modules':len(cos_sims),'interpretation':_interpret_ECI(eci)}
    return results

def _interpret_ECI(eci):
    if eci<0.3: return "Highly consistent"
    elif eci<0.5: return "Moderately consistent"
    elif eci<0.7: return "Notably inconsistent"
    else: return "Severely inconsistent"

def joint_decision(zPG,ECI,zPG_thresh=0,ECI_thresh=0.5):
    if zPG>zPG_thresh and ECI<ECI_thresh: return "GO: signal + direction stable"
    elif zPG>zPG_thresh and ECI>=ECI_thresh: return "CAUTION: signal but direction unstable"
    elif zPG<=zPG_thresh and ECI>=ECI_thresh: return "STOP: batch-dominated"
    else: return "NO_SIGNAL: weak individual coupling"

def compute_module_q2_simple(R_modules,P,design):
    n=len(P)
    if n<3: return np.nan,np.nan,0
    strata=design['condition'].astype(str)+'_'+design['batch'].astype(str)
    mod_names=sorted(R_modules.keys()); X=np.column_stack([R_modules[m] for m in mod_names])
    preds=np.full(n,np.nan)
    for i in range(n):
        train_idx=[j for j in range(n) if j!=i]; X_train,X_test=X[train_idx],X[i:i+1]; y_train,y_test=P[train_idx],P[i]
        try: model=RidgeCV(alphas=[0.01,0.1,1.0,10.0,100.0]); model.fit(X_train,y_train); preds[i]=model.predict(X_test)[0]
        except Exception: continue
    valid=~np.isnan(preds)
    if valid.sum()<3: return np.nan,np.nan,0
    stratum_preds=np.full(n,np.nan)
    for i in range(n):
        same_stratum=[j for j in range(n) if j!=i and strata.iloc[j]==strata.iloc[i]]
        stratum_preds[i]=np.mean(P[same_stratum]) if len(same_stratum)>=1 else np.mean(P[[j for j in range(n) if j!=i]])
    mse_model=np.mean((preds[valid]-P[valid])**2); mse_stratum=np.mean((stratum_preds[valid]-P[valid])**2)
    Q2=1-mse_model/(mse_stratum+1e-10); rho,_=spearmanr(preds[valid],P[valid])
    return Q2,rho,valid.sum()

def module_scores(rna_df,modules_ref,log1p=True):
    out={}
    for mod_name in sorted(modules_ref.keys()):
        genes=[g for g in modules_ref[mod_name] if g in rna_df.columns]
        if len(genes)>=2:
            block=np.log1p(rna_df[genes]) if log1p else rna_df[genes]; out[mod_name]=block.mean(axis=1)
    return pd.DataFrame(out)

def data_driven_modules(prot_df,n_modules=8,genes_per_module=None):
    from numpy.linalg import eigh
    prot_corr=prot_df.corr().values; n_genes=prot_corr.shape[0]
    _,evecs=eigh(prot_corr); pc1_loadings=np.abs(evecs[:,-1]); order=np.argsort(pc1_loadings)
    clusters=np.zeros(n_genes,dtype=int)
    if genes_per_module is None:
        for ci in range(n_modules):
            start,end=int(ci*n_genes/n_modules),int((ci+1)*n_genes/n_modules)
            for idx in order[start:end]: clusters[idx]=ci
    else:
        gpm=int(genes_per_module)
        for ci in range(n_modules):
            start,end=ci*gpm,min((ci+1)*gpm,n_genes)
            for idx in order[start:end]: clusters[idx]=ci
    prot_genes=list(prot_df.columns); modules={}
    for ci in range(n_modules):
        mod_genes=[prot_genes[j] for j in range(len(prot_genes)) if clusters[j]==ci]
        if len(mod_genes)>=3: modules[f'M{ci}']=mod_genes
    return modules

def simulate_paired_data(n,effect_size_d,n_modules=8,seed=42):
    rng_local=np.random.default_rng(seed); true_rho=effect_size_d/np.sqrt(4+effect_size_d**2)
    n_half1=n//2; n_half2=n-n_half1; conditions=['A']*n_half1+['B']*n_half2
    design=pd.DataFrame({'condition':conditions,'batch':['B1']*n})
    L=rng_local.normal(0,1,(n,2)); loadings=rng_local.uniform(0.3,1.0,(n_modules,2))
    rna_scores,prot_scores={},{}
    n_bridgeable=max(1,n_modules//2)
    for m in range(n_modules):
        latent=L@loadings[m]+rng_local.normal(0,0.1,n)
        rna_scores[f'M{m}']=latent+rng_local.normal(0,0.2,n)
        if m<n_bridgeable: prot_scores[f'M{m}']=true_rho*latent+(1-abs(true_rho))*rng_local.normal(0,1,n)+rng_local.normal(0,0.25,n)
        else: prot_scores[f'M{m}']=rng_local.normal(0,1,n)
    return rna_scores,prot_scores,design

__all__=['set_seed','compute_zpg','compute_rank_zPG','compute_rank_zPG_partial','compute_ECI',
         '_interpret_ECI','joint_decision','decide','decide_legacy','fdr_bh',
         'compute_module_q2_simple','module_scores','data_driven_modules','simulate_paired_data','select_cv']