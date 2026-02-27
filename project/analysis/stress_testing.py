"""Historical crisis stress testing module"""
import numpy as np
import pandas as pd

CRISES = {
    "GFC 2008":          {"start":"2008-09-01","end":"2009-03-31","label":"Global Financial Crisis","desc":"-55% S&P 500 in 6 months","color":"#FF4560"},
    "EU Debt 2011":      {"start":"2011-07-01","end":"2011-10-31","label":"EU Sovereign Debt Crisis","desc":"Greek/Italian contagion, -20% Euro Stoxx","color":"#FF8C00"},
    "China Shock 2015":  {"start":"2015-08-10","end":"2015-09-30","label":"China Devaluation","desc":"CNY devaluation, -11% S&P in 6 days","color":"#FFB800"},
    "COVID-19 2020":     {"start":"2020-02-19","end":"2020-03-23","label":"COVID-19 Crash","desc":"-34% S&P 500 in 33 trading days","color":"#A855F7"},
    "Inflation 2022":    {"start":"2022-01-03","end":"2022-10-12","label":"Inflation / Rate Shock","desc":"Fed +425bp, -25% S&P 500","color":"#4B9EFF"},
}

def _sq(s):
    if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
    return s.squeeze() if hasattr(s,"squeeze") else s

def _crisis_stats(rp, start, end, rm=None):
    r=_sq(rp).dropna()
    w=r.loc[(r.index>=pd.to_datetime(start))&(r.index<=pd.to_datetime(end))]
    if len(w)<3:
        return {"total_ret_pct":np.nan,"ann_vol_pct":np.nan,"max_dd_pct":np.nan,
                "beta":np.nan,"n_days":0,"available":False}
    total=float((1+w).prod()-1)*100
    vol=float(w.std()*np.sqrt(252))*100
    wealth=(1+w).cumprod(); dd=float(((wealth-wealth.cummax())/wealth.cummax()).min())*100
    beta=np.nan
    if rm is not None:
        m=_sq(rm).dropna()
        mw=m.loc[(m.index>=pd.to_datetime(start))&(m.index<=pd.to_datetime(end))]
        common=w.index.intersection(mw.index)
        if len(common)>=5:
            rv=w.loc[common].values.astype(float); mv=mw.loc[common].values.astype(float)
            vm=float(np.var(mv,ddof=1))
            if vm!=0: beta=float(np.cov(rv,mv)[0,1]/vm)
    # Recovery: days after end to recover to pre-crisis level
    pre_end=r.loc[r.index<pd.to_datetime(start)]
    pre_level=float((1+pre_end).prod()) if len(pre_end)>0 else 1.0
    post=r.loc[r.index>pd.to_datetime(end)]
    post_w=(1+post).cumprod()*float((1+w).prod())
    recovered=post_w[post_w>=pre_level]
    rec_days=int((recovered.index[0]-pd.to_datetime(end)).days) if len(recovered)>0 else None
    return {"total_ret_pct":total,"ann_vol_pct":vol,"max_dd_pct":dd,
            "beta":beta,"n_days":len(w),"recovery_days":rec_days,"available":True}

def run_stress_tests(rp, rm=None, extra_scenarios=None):
    scenarios=dict(CRISES)
    if extra_scenarios: scenarios.update(extra_scenarios)
    out={}
    for name,cfg in scenarios.items():
        m=_crisis_stats(rp,cfg["start"],cfg["end"],rm)
        m.update({"label":cfg.get("label",name),"desc":cfg.get("desc",""),
                  "color":cfg.get("color","#7B8098"),"start":cfg["start"],"end":cfg["end"]})
        out[name]=m
    return out

def stress_table(esg_res, cls_res):
    rows=[]
    for crisis in esg_res:
        e=esg_res[crisis]; c=cls_res.get(crisis,{})
        if not e.get("available") and not c.get("available"): continue
        rows.append({"Scénario":e.get("label",crisis),
                     "Période":f"{e['start']} → {e['end']}",
                     "ESG (%)":e.get("total_ret_pct",np.nan),
                     "Classic (%)":c.get("total_ret_pct",np.nan),
                     "ESG MaxDD (%)":e.get("max_dd_pct",np.nan),
                     "Cls MaxDD (%)":c.get("max_dd_pct",np.nan),
                     "ESG Beta":e.get("beta",np.nan),
                     "Cls Beta":c.get("beta",np.nan),
                     "ESG Recov (j)":e.get("recovery_days",None),
                     "Cls Recov (j)":c.get("recovery_days",None)})
    return pd.DataFrame(rows)
