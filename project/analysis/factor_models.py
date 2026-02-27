"""Factor models: CAPM, FF3, FF5, FF5+MOM — HAC-robust OLS
Functions: run_capm, run_ff3, run_ff5, run_ff5_momentum, rolling_alpha, factor_decomposition
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _squeeze(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze() if hasattr(s, "squeeze") else s


def _prep(rp, ff, cols):
    r = _squeeze(rp).rename("Rp")
    avail = [c for c in cols + ["RF"] if c in ff.columns]
    if "RF" not in avail:
        raise ValueError("RF column missing from ff_factors")
    df = pd.concat([r, ff[avail]], axis=1).dropna()
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} aligned obs — need 30+")
    y  = (df["Rp"] - df["RF"]).values.astype(float)
    fc = [c for c in cols if c in df.columns]
    X  = df[fc].values.astype(float)
    return y, X, fc


def _ols(y, X, fnames, maxlags=5):
    Xc  = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    p   = res.params
    tv  = res.tvalues
    pv  = res.pvalues
    ci  = res.conf_int()
    ic  = float(p[0])
    ia  = float((1 + ic) ** 252 - 1) * 100
    it  = float(tv[0])
    ip  = float(pv[0])
    factors = {}
    for i, fn in enumerate(fnames, 1):
        if i < len(p):
            factors[fn] = {
                "loading":     float(p[i]),
                "t_stat":      float(tv[i]),
                "p_value":     float(pv[i]),
                "ci_low":      float(ci.iloc[i, 0]),
                "ci_high":     float(ci.iloc[i, 1]),
                "significant": bool(abs(float(tv[i])) > 2),
            }
    return {
        "alpha_annual_pct":  ia,
        "alpha_daily":       ic,
        "alpha_tstat":       it,
        "alpha_pvalue":      ip,
        "alpha_significant": bool(abs(it) > 2 and ip < 0.05),
        "r_squared":         float(res.rsquared),
        "r_squared_adj":     float(res.rsquared_adj),
        "n_obs":             int(res.nobs),
        "factors":           factors,
    }


def run_capm(rp, ff):
    y, X, fc = _prep(rp, ff, ["Mkt-RF"])
    return _ols(y, X, fc)


def run_ff3(rp, ff):
    y, X, fc = _prep(rp, ff, ["Mkt-RF", "SMB", "HML"])
    return _ols(y, X, fc)


def run_ff5(rp, ff):
    y, X, fc = _prep(rp, ff, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    return _ols(y, X, fc)


def run_ff5_momentum(rp, ff):
    y, X, fc = _prep(rp, ff, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    return _ols(y, X, fc)


def rolling_alpha(rp, ff, model="FF5", window=252):
    fn_map   = {"CAPM": run_capm, "FF3": run_ff3, "FF5": run_ff5}
    fn       = fn_map.get(model, run_ff5)
    cols_map = {
        "CAPM": ["Mkt-RF"],
        "FF3":  ["Mkt-RF", "SMB", "HML"],
        "FF5":  ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }
    cols  = cols_map.get(model, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
    avail = [c for c in cols + ["RF"] if c in ff.columns]
    r     = _squeeze(rp)
    df    = pd.concat([r.rename("Rp"), ff[avail]], axis=1).dropna()
    out   = pd.Series(index=df.index, dtype=float)
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window:i]
        try:
            res = fn(chunk["Rp"], chunk[avail])
            out.iloc[i - 1] = res.get("alpha_annual_pct", np.nan)
        except Exception:
            out.iloc[i - 1] = np.nan
    return out


def factor_decomposition(ff_result, ff_factors):
    rows   = []
    fmeans = ff_factors.mean()
    for fn, fd in ff_result.get("factors", {}).items():
        b      = fd.get("loading", 0)
        mr     = float(fmeans.get(fn, 0))
        contrib = b * mr * 252 * 100
        rows.append({
            "Factor":                  fn,
            "Beta":                    round(b, 4),
            "Mean Factor Ret (% ann)": round(mr * 252 * 100, 3),
            "Contribution (bps)":      round(contrib * 100, 1),
            "Significant":             fd.get("significant", False),
        })
    return pd.DataFrame(rows)
