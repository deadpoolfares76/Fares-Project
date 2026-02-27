"""Institutional risk metrics: Sharpe, Sortino, Calmar, Omega, IR, Tracking Error, VaR, CVaR..."""
import numpy as np
import pandas as pd
from scipy import stats

def _sq(s):
    if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
    return s.squeeze() if hasattr(s,"squeeze") else s

def _align(*series):
    df = pd.concat([_sq(s).rename(i) for i,s in enumerate(series)], axis=1).dropna()
    return tuple(df[i] for i in range(len(series)))

def ann_ret(r, p=252):
    try: return float((1+float(_sq(r).dropna().mean()))**p-1)
    except: return np.nan

def ann_vol(r, p=252):
    try: return float(_sq(r).dropna().std()*np.sqrt(p))
    except: return np.nan

def compute_total_return(r):
    r=_sq(r).dropna()
    return float((1+r).prod()-1)*100 if len(r)>0 else np.nan

def compute_beta(rp, rm):
    rp,rm=_align(rp,rm)
    if len(rp)<10: return np.nan
    cov=np.cov(rp.values.astype(float),rm.values.astype(float))
    var=float(np.var(rm.values.astype(float),ddof=1))
    return float(cov[0,1]/var) if var!=0 else np.nan

def compute_alpha_capm(rp, rm, rf):
    rp,rm,rf=_align(rp,rm,rf)
    if len(rp)<10: return {}
    ep=(rp-rf).values.astype(float); em=(rm-rf).values.astype(float)
    sl,ic,rv,pv,se=stats.linregress(em,ep)
    sl,ic,rv,pv,se=float(sl),float(ic),float(rv),float(pv),float(se)
    t=ic/se if se!=0 else np.nan
    return {"alpha":float((1+ic)**252-1),"alpha_pct":float((1+ic)**252-1)*100,
            "beta":sl,"r_squared":rv**2,"t_stat":t,"p_value":pv}

def compute_sharpe(rp,rf):
    rp,rf=_align(rp,rf); ex=(rp-rf).dropna()
    if len(ex)<10: return np.nan
    mu,sg=float(ex.mean()),float(ex.std())
    return float(mu/sg*np.sqrt(252)) if sg!=0 else np.nan

def compute_sortino(rp,rf):
    rp,rf=_align(rp,rf); ex=(rp-rf).dropna()
    if len(ex)<10: return np.nan
    dn=ex[ex<0]
    if len(dn)<2: return np.nan
    dv=float(dn.std()*np.sqrt(252))
    return float(ex.mean()*252/dv) if dv!=0 else np.nan

def compute_treynor(rp,rm,rf):
    rp,rm,rf=_align(rp,rm,rf)
    beta=compute_beta(rp,rm)
    if np.isnan(beta) or beta==0: return np.nan
    return float((float(rp.mean())-float(rf.mean()))*252/beta)

def compute_calmar(rp):
    r=_sq(rp).dropna()
    if len(r)<10: return np.nan
    a=ann_ret(r); dd=compute_max_drawdown(r)
    return float(a/abs(dd/100)) if not np.isnan(dd) and dd!=0 else np.nan

def compute_omega(rp, threshold=0.0):
    r=_sq(rp).dropna().values.astype(float)
    if len(r)<10: return np.nan
    g=np.sum(np.maximum(r-threshold,0)); l=np.sum(np.maximum(threshold-r,0))
    return float(g/l) if l!=0 else np.nan

def compute_information_ratio(rp,rb):
    rp,rb=_align(rp,rb); active=(rp-rb).dropna()
    if len(active)<10: return np.nan
    te=float(active.std()*np.sqrt(252))
    return float(active.mean()*252/te) if te!=0 else np.nan

def compute_tracking_error(rp,rb):
    rp,rb=_align(rp,rb); active=(rp-rb).dropna()
    return float(active.std()*np.sqrt(252))*100 if len(active)>=10 else np.nan

def compute_drawdown_series(rp):
    r=_sq(rp).dropna(); w=(1+r).cumprod(); pk=w.cummax()
    return (w-pk)/pk*100

def compute_max_drawdown(rp):
    dd=compute_drawdown_series(rp)
    return float(dd.min()) if len(dd)>0 else np.nan

def compute_var_hist(rp, conf=0.95):
    r=_sq(rp).dropna().values.astype(float)
    return float(np.percentile(r,(1-conf)*100))*100 if len(r)>=10 else np.nan

def compute_var_param(rp, conf=0.95):
    r=_sq(rp).dropna().values.astype(float)
    if len(r)<10: return np.nan
    return float(r.mean()+stats.norm.ppf(1-conf)*r.std())*100

def compute_cvar(rp, conf=0.95):
    r=_sq(rp).dropna().values.astype(float)
    if len(r)<10: return np.nan
    vp=np.percentile(r,(1-conf)*100); tail=r[r<=vp]
    return float(tail.mean())*100 if len(tail)>0 else np.nan

def compute_skewness(rp):
    r=_sq(rp).dropna()
    return float(r.skew()) if len(r)>3 else np.nan

def compute_kurtosis(rp):
    r=_sq(rp).dropna()
    return float(r.kurtosis()) if len(r)>3 else np.nan

def rolling_volatility(rp, window=63):
    return _sq(rp).dropna().rolling(window).std()*np.sqrt(252)*100

def rolling_beta(rp,rm,window=63):
    rp2,rm2=_align(rp,rm)
    combined=pd.concat([rp2.rename("p"),rm2.rename("m")],axis=1).dropna()
    out=pd.Series(index=combined.index,dtype=float)
    for i in range(window,len(combined)+1):
        w=combined.iloc[i-window:i].values.astype(float)
        vm=np.var(w[:,1],ddof=1)
        out.iloc[i-1]=float(np.cov(w[:,0],w[:,1])[0,1]/vm) if vm!=0 else np.nan
    return out

def rolling_correlation(rp,rm,window=63):
    rp2,rm2=_align(rp,rm)
    return rp2.rolling(window).corr(rm2)

def rolling_sharpe(rp,rf,window=252):
    rp2,rf2=_align(rp,rf); ex=rp2-rf2
    rm=ex.rolling(window).mean(); rs=ex.rolling(window).std()
    return (rm/rs*np.sqrt(252)).replace([np.inf,-np.inf],np.nan)

def detect_volatility_regimes(rp,window=21,threshold_pct=75.0):
    r=_sq(rp).dropna(); rv=r.rolling(window).std()*np.sqrt(252)
    return rv>=np.nanpercentile(rv.dropna(),threshold_pct)

def regime_analysis(rp,rm,rf,mask,name="Portfolio"):
    rp2,rm2,rf2=_align(rp,rm,rf)
    msk=_sq(mask).reindex(rp2.index).ffill().bfill()
    out={}
    for label,cond in [("High Volatility",msk==True),("Low Volatility",msk==False)]:
        idx=cond[cond].index
        sr=rp2.loc[rp2.index.intersection(idx)]
        smr=rm2.loc[rm2.index.intersection(idx)]
        sfr=rf2.loc[rf2.index.intersection(idx)]
        if len(sr)<20: continue
        out[label]={"n_days":len(sr),"ann_return_pct":ann_ret(sr)*100,
                    "ann_vol_pct":ann_vol(sr)*100,"sharpe":compute_sharpe(sr,sfr),
                    "max_dd_pct":compute_max_drawdown(sr),"beta":compute_beta(sr,smr)}
    return out

def full_risk_summary(name,rp,rm,rf,benchmark=None,confidence=0.95):
    r=_sq(rp).dropna(); m=_sq(rm).dropna(); f=_sq(rf).dropna()
    capm=compute_alpha_capm(r,m,f)
    return {
        "Name":name,
        "Ann. Return (%)":ann_ret(r)*100,
        "Total Return (%)":compute_total_return(r),
        "Ann. Volatility (%)":ann_vol(r)*100,
        "Sharpe Ratio":compute_sharpe(r,f),
        "Sortino Ratio":compute_sortino(r,f),
        "Treynor Ratio":compute_treynor(r,m,f),
        "Calmar Ratio":compute_calmar(r),
        "Omega Ratio":compute_omega(r),
        "Information Ratio":compute_information_ratio(r,benchmark) if benchmark is not None else np.nan,
        "Tracking Error (%)":compute_tracking_error(r,benchmark) if benchmark is not None else np.nan,
        "Beta":capm.get("beta",np.nan),
        "CAPM Alpha (%)":capm.get("alpha_pct",np.nan),
        "Alpha t-stat":capm.get("t_stat",np.nan),
        "Alpha p-value":capm.get("p_value",np.nan),
        "R-squared":capm.get("r_squared",np.nan),
        "Max Drawdown (%)":compute_max_drawdown(r),
        "VaR Hist (%)":compute_var_hist(r,confidence),
        "VaR Param (%)":compute_var_param(r,confidence),
        "CVaR (%)":compute_cvar(r,confidence),
        "Skewness":compute_skewness(r),
        "Excess Kurtosis":compute_kurtosis(r),
        "n_obs":len(r),
    }
