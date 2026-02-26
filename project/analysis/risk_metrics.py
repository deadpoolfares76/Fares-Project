"""
Risk Metrics Module
Computes all risk/return metrics for a portfolio return series.
"""
import numpy as np
import pandas as pd
from scipy import stats

def _squeeze_series(s) -> pd.Series:
    """Ensure s is a proper 1-D pandas Series."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if hasattr(s, 'squeeze'):
        s = s.squeeze()
    return s



# ─────────────────────────────────────────────────────────────────────────────
# ANNUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
TRADING_DAYS = 252


def annualize_return(daily_ret: float) -> float:
    return (1 + daily_ret) ** TRADING_DAYS - 1


def annualize_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# BASIC RETURN & RISK MEASURES
# ─────────────────────────────────────────────────────────────────────────────
def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """R = (V_t - V_{t-1}) / V_{t-1}"""
    return prices.pct_change().dropna()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """r = ln(V_t / V_{t-1})"""
    return np.log(prices / prices.shift(1)).dropna()


def compute_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """σ = sqrt(Var(R))"""
    vol = returns.std()
    return annualize_vol(vol) if annualize else vol


def compute_variance(returns: pd.Series) -> float:
    """Var(R) = E[(R - μ)²]"""
    return float(returns.var())


def compute_beta(portfolio_returns: pd.Series,
                 market_returns: pd.Series) -> float:
    """β = Cov(R_p, R_m) / Var(R_m)"""
    aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    cov_matrix = aligned.cov()
    var_market = aligned.iloc[:, 1].var()
    if var_market == 0:
        return np.nan
    return cov_matrix.iloc[0, 1] / var_market


def compute_alpha_capm(portfolio_returns: pd.Series,
                       market_returns: pd.Series,
                       risk_free_rate: pd.Series) -> dict:
    """
    Estimate CAPM alpha and beta via OLS.
    Returns dict with alpha, beta, t-stat, p-value, r-squared.
    """
    # Ensure all inputs are 1-D Series to avoid ambiguous array truth-value errors
    p = portfolio_returns.squeeze() if hasattr(portfolio_returns, "squeeze") else portfolio_returns
    m = market_returns.squeeze() if hasattr(market_returns, "squeeze") else market_returns
    rf = risk_free_rate.squeeze() if hasattr(risk_free_rate, "squeeze") else risk_free_rate
    if isinstance(rf, pd.DataFrame):
        rf = rf.iloc[:, 0]
    df = pd.concat([p.rename("Rp"), m.rename("Rm"), rf.rename("Rf")], axis=1).dropna()
    excess_p = df["Rp"] - df["Rf"]
    excess_m = df["Rm"] - df["Rf"]
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        excess_m.values.astype(float), excess_p.values.astype(float)
    )
    # Cast everything to plain Python floats immediately
    slope, intercept, r_val, p_val, std_err = (
        float(slope), float(intercept), float(r_val), float(p_val), float(std_err)
    )
    alpha_annual = annualize_return(intercept)
    t_stat = intercept / std_err if std_err != 0.0 else np.nan
    return {
        "alpha": alpha_annual,
        "alpha_daily": intercept,
        "beta": slope,
        "r_squared": r_val ** 2,
        "t_stat_alpha": t_stat,
        "p_value_alpha": p_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE RATIOS
# ─────────────────────────────────────────────────────────────────────────────
def compute_sharpe(portfolio_returns: pd.Series,
                   risk_free_rate: pd.Series) -> float:
    """Sharpe = (R_p - R_f) / σ_p  (annualised)"""
    portfolio_returns = _squeeze_series(portfolio_returns)
    risk_free_rate = _squeeze_series(risk_free_rate)
    df = pd.concat([portfolio_returns.rename("Rp"), risk_free_rate.rename("Rf")],
                   axis=1).dropna()
    excess = df["Rp"] - df["Rf"]
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def compute_sortino(portfolio_returns: pd.Series,
                    risk_free_rate: pd.Series) -> float:
    """Sortino = (R_p - R_f) / σ_downside  (annualised)"""
    portfolio_returns = _squeeze_series(portfolio_returns)
    risk_free_rate = _squeeze_series(risk_free_rate)
    df = pd.concat([portfolio_returns.rename("Rp"), risk_free_rate.rename("Rf")],
                   axis=1).dropna()
    excess = df["Rp"] - df["Rf"]
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    ann_excess = excess.mean() * TRADING_DAYS
    return ann_excess / downside_vol


def compute_treynor(portfolio_returns: pd.Series,
                    market_returns: pd.Series,
                    risk_free_rate: pd.Series) -> float:
    """Treynor = (R_p - R_f) / β_p  (annualised)"""
    portfolio_returns = _squeeze_series(portfolio_returns)
    market_returns = _squeeze_series(market_returns)
    risk_free_rate = _squeeze_series(risk_free_rate)
    df = pd.concat([portfolio_returns.rename("Rp"), market_returns.rename("Rm"),
                    risk_free_rate.rename("Rf")], axis=1).dropna()
    excess_p = df["Rp"] - df["Rf"]
    beta = compute_beta(df["Rp"], df["Rm"])
    if beta == 0 or np.isnan(beta):
        return np.nan
    return (excess_p.mean() * TRADING_DAYS) / beta


def compute_calmar(portfolio_returns: pd.Series) -> float:
    """Calmar = Annualised Return / |Max Drawdown|"""
    ann_ret = portfolio_returns.mean() * TRADING_DAYS
    mdd = compute_max_drawdown(portfolio_returns)
    if mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)


def compute_information_ratio(portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series) -> float:
    """IR = Mean(Active Return) / Tracking Error"""
    active = portfolio_returns - benchmark_returns
    te = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return np.nan
    return (active.mean() * TRADING_DAYS) / te


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN & EXTREME LOSS METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Returns the drawdown series (negative values)."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum Drawdown (negative float)."""
    dd = compute_drawdown_series(returns)
    return float(dd.min())


def compute_var(returns: pd.Series, confidence: float = 0.95,
                method: str = "historical") -> float:
    """
    Value at Risk at given confidence level.
    method: 'historical' or 'parametric'
    Returns a negative number (loss).
    """
    if method == "historical":
        return float(np.percentile(returns, (1 - confidence) * 100))
    else:  # parametric / Gaussian
        mu = returns.mean()
        sigma = returns.std()
        return float(stats.norm.ppf(1 - confidence, mu, sigma))


def compute_cvar(returns: pd.Series, confidence: float = 0.95,
                 method: str = "historical") -> float:
    """
    Conditional VaR / Expected Shortfall.
    Returns the mean of losses beyond VaR (negative number).
    """
    var = compute_var(returns, confidence, method)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL MOMENTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_skewness(returns: pd.Series) -> float:
    """Skew = E[(R-μ)³] / σ³"""
    return float(returns.skew())


def compute_kurtosis(returns: pd.Series) -> float:
    """Kurt = E[(R-μ)⁴] / σ⁴  (excess kurtosis, Normal = 0)"""
    return float(returns.kurtosis())


def compute_jarque_bera(returns: pd.Series) -> dict:
    """Jarque-Bera normality test."""
    jb_stat, jb_pval = stats.jarque_bera(returns.dropna())
    return {"statistic": jb_stat, "p_value": jb_pval,
            "normal": jb_pval > 0.05}


def compute_hit_ratio(portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
    """Percentage of periods where portfolio outperforms benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns],
                        axis=1).dropna()
    outperform = (aligned.iloc[:, 0] > aligned.iloc[:, 1]).sum()
    return outperform / len(aligned)


# ─────────────────────────────────────────────────────────────────────────────
# FULL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def full_risk_summary(name: str,
                      returns: pd.Series,
                      market_returns: pd.Series,
                      risk_free_rate: pd.Series,
                      benchmark_returns: pd.Series = None,
                      confidence: float = 0.95) -> dict:
    """
    Compute all risk/return metrics and return as a dict.
    """
    capm = compute_alpha_capm(returns, market_returns, risk_free_rate)

    summary = {
        "Name": name,
        # ── Returns
        "Ann. Return (%)": returns.mean() * TRADING_DAYS * 100,
        "Total Return (%)": ((1 + returns).prod() - 1) * 100,
        # ── Risk
        "Ann. Volatility (%)": compute_volatility(returns) * 100,
        "Variance (daily)": compute_variance(returns),
        "Beta": capm["beta"],
        "Max Drawdown (%)": compute_max_drawdown(returns) * 100,
        # ── Ratios
        "Sharpe Ratio": compute_sharpe(returns, risk_free_rate),
        "Sortino Ratio": compute_sortino(returns, risk_free_rate),
        "Treynor Ratio": compute_treynor(returns, market_returns, risk_free_rate),
        "Calmar Ratio": compute_calmar(returns),
        # ── CAPM
        "CAPM Alpha (%)": capm["alpha"] * 100,
        "CAPM R²": capm["r_squared"],
        "Alpha t-stat": capm["t_stat_alpha"],
        # ── Tail risk
        f"VaR {confidence*100:.0f}% (%)": compute_var(returns, confidence) * 100,
        f"CVaR {confidence*100:.0f}% (%)": compute_cvar(returns, confidence) * 100,
        # ── Moments
        "Skewness": compute_skewness(returns),
        "Excess Kurtosis": compute_kurtosis(returns),
    }

    if benchmark_returns is not None:
        summary["Info. Ratio"] = compute_information_ratio(returns, benchmark_returns)
        summary["Hit Ratio (%)"] = compute_hit_ratio(returns, benchmark_returns) * 100

    return summary
