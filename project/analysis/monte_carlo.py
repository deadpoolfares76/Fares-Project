"""Monte Carlo GBM simulation — correlated & independent paths"""
import numpy as np
import pandas as pd

def _sq(s):
    if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
    return s.squeeze() if hasattr(s,"squeeze") else s

def simulate_gbm(rp, n_sims=1000, horizon_years=5.0, tdays=252, seed=42):
    """GBM: S(t)=S(0)*exp[(mu-sigma^2/2)*t + sigma*W(t)]"""
    r=_sq(rp).dropna().values.astype(float)
    if len(r)<20: raise ValueError("Need >= 20 observations")
    mu=float(r.mean()); sg=float(r.std(ddof=1))
    n_steps=int(horizon_years*tdays)
    np.random.seed(seed)
    shocks=np.random.normal(0,1,(n_steps,n_sims))
    daily_lr=(mu-0.5*sg**2)+sg*shocks
    paths=100*np.exp(np.cumsum(daily_lr,axis=0))
    full=np.vstack([np.full((1,n_sims),100.0),paths])
    term=full[-1,:]
    pct={k:np.percentile(full,k,axis=1) for k in [5,25,50,75,95]}
    return {
        "paths":full,"time_axis":np.arange(n_steps+1),"percentiles":pct,
        "terminal":term,
        "metrics":{
            "ann_ret_pct":mu*tdays*100,"ann_vol_pct":sg*np.sqrt(tdays)*100,
            "prob_loss_pct":float(np.mean(term<100))*100,
            "prob_double_pct":float(np.mean(term>200))*100,
            "var5":float(np.percentile(term,5)),
            "expected":float(np.mean(term)),"median":float(np.median(term)),
            "n_sims":n_sims,"horizon_years":horizon_years,
        }
    }

def simulate_correlated(rp_esg, rp_cls, n_sims=1000, horizon_years=5.0, tdays=252, seed=42):
    """Two correlated GBM paths via Cholesky decomposition"""
    try:
        from analysis.risk_metrics import _align
    except ImportError:
        from risk_metrics import _align
    re,rc=_align(rp_esg,rp_cls)
    data=np.column_stack([re.values.astype(float),rc.values.astype(float)])
    if len(data)<20: raise ValueError("Need >= 20 aligned obs")
    mu=data.mean(axis=0); cov=np.cov(data,rowvar=False)
    corr=float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))
    n_steps=int(horizon_years*tdays)
    np.random.seed(seed)
    try: L=np.linalg.cholesky(cov)
    except np.linalg.LinAlgError: L=np.linalg.cholesky(cov+np.eye(2)*1e-10)
    raw=np.random.normal(0,1,(n_steps,2,n_sims))
    corr_shocks=np.einsum("ij,tjk->tik",L,raw)
    out={}
    for i,(name,mu_i,var_i) in enumerate([("esg",mu[0],cov[0,0]),("classic",mu[1],cov[1,1])]):
        sg=np.sqrt(var_i)
        lr=(mu_i-0.5*sg**2)+corr_shocks[:,i,:]
        paths=100*np.exp(np.cumsum(lr,axis=0))
        full=np.vstack([np.full((1,n_sims),100.0),paths])
        term=full[-1,:]
        out[name]={"paths":full,"percentiles":{k:np.percentile(full,k,axis=1) for k in [5,25,50,75,95]},
                   "terminal":term,
                   "metrics":{"ann_ret_pct":mu_i*tdays*100,"ann_vol_pct":sg*np.sqrt(tdays)*100,
                               "prob_loss_pct":float(np.mean(term<100))*100,
                               "prob_double_pct":float(np.mean(term>200))*100,
                               "var5":float(np.percentile(term,5)),
                               "expected":float(np.mean(term)),"median":float(np.median(term))}}
    out["time_axis"]=np.arange(n_steps+1)
    out["correlation"]=corr; out["n_sims"]=n_sims; out["horizon_years"]=horizon_years
    return out
