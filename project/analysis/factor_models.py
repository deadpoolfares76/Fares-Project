"""
Factor Models Module
CAPM, Fama-French 3, 5, and 5+Momentum factor regressions.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


TRADING_DAYS = 252


def _align_data(returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Align portfolio returns with factor data on common dates."""
    df = pd.concat([returns.rename("Rp"), factors], axis=1).dropna()
    return df


def run_capm(portfolio_returns: pd.Series,
             factors: pd.DataFrame) -> dict:
    """
    OLS regression: excess_return ~ MKT
    Expects factors to contain 'Mkt-RF' and 'RF'.
    """
    df = _align_data(portfolio_returns, factors)
    df["excess_Rp"] = df["Rp"] - df["RF"]
    X = sm.add_constant(df["Mkt-RF"])
    y = df["excess_Rp"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return _extract_results(model, "CAPM")


def run_ff3(portfolio_returns: pd.Series,
            factors: pd.DataFrame) -> dict:
    """
    OLS regression: excess_return ~ MKT + SMB + HML
    """
    df = _align_data(portfolio_returns, factors)
    df["excess_Rp"] = df["Rp"] - df["RF"]
    factor_cols = ["Mkt-RF", "SMB", "HML"]
    X = sm.add_constant(df[factor_cols])
    y = df["excess_Rp"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return _extract_results(model, "Fama-French 3")


def run_ff5(portfolio_returns: pd.Series,
            factors: pd.DataFrame) -> dict:
    """
    OLS regression: excess_return ~ MKT + SMB + HML + RMW + CMA
    Requires FF5 factors (RMW, CMA present in factors).
    """
    df = _align_data(portfolio_returns, factors)
    df["excess_Rp"] = df["Rp"] - df["RF"]
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                   if c in df.columns]
    if len(factor_cols) < 5:
        return {"error": "FF5 factors not all available. Using available ones.",
                "available": factor_cols}
    X = sm.add_constant(df[factor_cols])
    y = df["excess_Rp"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return _extract_results(model, "Fama-French 5")


def run_ff5_momentum(portfolio_returns: pd.Series,
                     factors: pd.DataFrame) -> dict:
    """
    OLS regression: excess_return ~ MKT + SMB + HML + RMW + CMA + MOM
    """
    df = _align_data(portfolio_returns, factors)
    df["excess_Rp"] = df["Rp"] - df["RF"]
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
                   if c in df.columns]
    X = sm.add_constant(df[factor_cols])
    y = df["excess_Rp"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return _extract_results(model, "FF5 + Momentum")


def _extract_results(model: sm.regression.linear_model.RegressionResultsWrapper,
                     model_name: str) -> dict:
    """Extract key stats from statsmodels OLS result."""
    params = model.params
    tvalues = model.tvalues
    pvalues = model.pvalues
    conf_int = model.conf_int()

    # Annualise alpha (intercept)
    alpha_daily = params.get("const", np.nan)
    alpha_annual = ((1 + alpha_daily) ** TRADING_DAYS - 1) * 100  # in %

    results = {
        "model": model_name,
        "alpha_daily": alpha_daily,
        "alpha_annual_pct": alpha_annual,
        "alpha_tstat": tvalues.get("const", np.nan),
        "alpha_pvalue": pvalues.get("const", np.nan),
        "alpha_significant": pvalues.get("const", 1.0) < 0.05,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "factors": {},
    }

    for factor in params.index:
        if factor == "const":
            continue
        results["factors"][factor] = {
            "loading": params[factor],
            "t_stat": tvalues[factor],
            "p_value": pvalues[factor],
            "significant": pvalues[factor] < 0.05,
            "ci_lower": conf_int.loc[factor, 0],
            "ci_upper": conf_int.loc[factor, 1],
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING BETAS & CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────
def rolling_beta(portfolio_returns: pd.Series,
                 market_returns: pd.Series,
                 window: int = 63) -> pd.Series:
    """Compute rolling beta using a fixed window (default 63 trading days ≈ 1Q)."""
    df = pd.concat([portfolio_returns, market_returns], axis=1,
                   keys=["Rp", "Rm"]).dropna()
    X = sm.add_constant(df["Rm"])
    rols = RollingOLS(df["Rp"], X, window=window).fit()
    return rols.params["Rm"].rename("Rolling Beta")


def rolling_correlation(portfolio_returns: pd.Series,
                         market_returns: pd.Series,
                         window: int = 63) -> pd.Series:
    """Rolling Pearson correlation between portfolio and market."""
    df = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualised rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def regime_analysis(portfolio_returns: pd.Series,
                    market_returns: pd.Series,
                    risk_free_rate: pd.Series,
                    regime_mask: pd.Series,
                    name: str = "Portfolio") -> dict:
    """
    Compare metrics in HIGH vs LOW volatility regimes.
    regime_mask: boolean Series, True = high vol.
    """
    from analysis.risk_metrics import (compute_sharpe, compute_max_drawdown,
                                        compute_cvar, compute_beta)

    results = {}
    for label, mask in [("High Volatility", regime_mask),
                         ("Low Volatility", ~regime_mask)]:
        p = portfolio_returns[mask]
        m = market_returns[mask]
        rf = risk_free_rate[mask]
        if len(p) < 10:
            continue
        results[label] = {
            "n_days": len(p),
            "ann_return_pct": p.mean() * TRADING_DAYS * 100,
            "ann_vol_pct": p.std() * np.sqrt(TRADING_DAYS) * 100,
            "beta": compute_beta(p, m),
            "sharpe": compute_sharpe(p, rf),
            "max_dd_pct": compute_max_drawdown(p) * 100,
            "cvar_95_pct": compute_cvar(p, 0.95) * 100,
        }
    return results
