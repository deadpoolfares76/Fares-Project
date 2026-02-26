"""
Data Loader Module
Handles extraction from yfinance and Kenneth French Data Library.
Includes 3-level fallback for Fama-French factors when Dartmouth server is unreachable.
"""
import io
import zipfile
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT TICKER MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ESG_TICKERS = {
    "MSCI World ESG Leaders": "SUSL",
    "MSCI USA ESG Select": "SUSA",
    "S&P 500 ESG": "ESGU",
    "MSCI Europe ESG": "ESGD",
    "MSCI EM ESG": "ESGE",
}

DEFAULT_CLASSIC_TICKERS = {
    "MSCI World (Classic)": "URTH",
    "S&P 500": "SPY",
    "MSCI USA": "EFA",
    "MSCI Europe": "VGK",
    "MSCI EM": "EEM",
}

RISK_FREE_TICKER = "^IRX"
MARKET_TICKER = "SPY"

# Kenneth French URLs (try both HTTP and HTTPS mirrors)
_FF_URLS = {
    "FF3": [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
        "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    ],
    "FF5": [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ],
    "MOM": [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
        "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    ],
}

# ETF proxies used to construct factor approximations when server is down
_PROXY_TICKERS = {
    "market": "SPY",    # Market (MKT-RF proxy)
    "small":  "IWM",    # Small-cap (for SMB)
    "large":  "IVV",    # Large-cap (for SMB)
    "value":  "IVE",    # Value (for HML)
    "growth": "IVW",    # Growth (for HML)
    "qual":   "QUAL",   # Quality / profitability (for RMW)
    "lowvol": "USMV",   # Low investment (for CMA)
    "mom":    "MTUM",   # Momentum (for MOM)
    "rf":     "^IRX",   # Risk-free rate
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _to_series(obj, name: str = None) -> pd.Series:
    """Safely coerce a yfinance result to a 1-D pd.Series."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    s = obj.squeeze()
    if not isinstance(s, pd.Series):
        s = pd.Series([float(s)], name=name)
    return s.rename(name) if name else s


def _squeeze_series(s) -> pd.Series:
    """Ensure s is a proper 1-D pandas Series."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if hasattr(s, "squeeze"):
        s = s.squeeze()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def download_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for a list of tickers."""
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            if len(tickers) == 1:
                prices.columns = tickers
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(
                name=tickers[0] if len(tickers) == 1 else "price")
        return prices.dropna(how="all")
    except Exception as e:
        raise RuntimeError(f"Failed to download prices: {e}")


def get_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute returns from prices. method: 'log' or 'simple'."""
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# RISK-FREE RATE
# ─────────────────────────────────────────────────────────────────────────────
def get_risk_free_rate(start: str, end: str, freq: str = "daily") -> pd.Series:
    """Download 3-month T-Bill rate. Always returns a 1-D pd.Series."""
    try:
        raw = yf.download(RISK_FREE_TICKER, start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        rf_annual = _to_series(raw, "RF") / 100.0
        if freq == "daily":
            return ((1 + rf_annual) ** (1 / 252) - 1).rename("RF")
        return rf_annual
    except Exception:
        idx = pd.date_range(start=start, end=end, freq="B")
        return pd.Series(0.04 / 252, index=idx, name="RF")


# ─────────────────────────────────────────────────────────────────────────────
# FAMA-FRENCH — LEVEL 1: pandas-datareader
# ─────────────────────────────────────────────────────────────────────────────
def _ff_via_datareader(model: str, start_dt, end_dt) -> pd.DataFrame:
    """Try pandas-datareader (uses Dartmouth internally)."""
    import pandas_datareader.data as web
    _DATASET_KEYS = {
        "FF3": "F-F_Research_Data_Factors_daily",
        "FF5": "F-F_Research_Data_5_Factors_2x3_daily",
        "MOM": "F-F_Momentum_Factor_daily",
    }
    base_key = "FF3" if model == "FF3" else "FF5"
    ff = web.DataReader(_DATASET_KEYS[base_key], "famafrench",
                        start=start_dt, end=end_dt)
    ff_df = ff[0] / 100.0

    if "FF5+MOM" in model:
        mom = web.DataReader(_DATASET_KEYS["MOM"], "famafrench",
                             start=start_dt, end=end_dt)
        mom_df = mom[0] / 100.0
        mom_df.columns = ["Mom"]
        ff_df = ff_df.join(mom_df, how="inner")

    ff_df.index = pd.to_datetime(ff_df.index)
    return ff_df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FAMA-FRENCH — LEVEL 2: Direct HTTP download + parse CSV
# ─────────────────────────────────────────────────────────────────────────────
def _download_ff_zip(key: str, timeout: int = 20) -> pd.DataFrame:
    """Download and parse a French Data Library CSV zip directly."""
    last_err = None
    for url in _FF_URLS[key]:
        try:
            resp = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = [n for n in z.namelist()
                            if n.lower().endswith(".csv")][0]
                with z.open(csv_name) as f:
                    raw = f.read().decode("utf-8", errors="replace")
            return _parse_ff_csv(raw)
        except Exception as e:
            last_err = e
            continue
    raise ConnectionError(f"All URLs for {key} failed. Last error: {last_err}")


def _parse_ff_csv(raw: str) -> pd.DataFrame:
    """Parse Kenneth French CSV format (skips header text, handles footer)."""
    lines = raw.splitlines()
    # Find the first line that looks like a data header (contains 'Mkt' or 'SMB')
    header_idx = 0
    for i, line in enumerate(lines):
        if any(kw in line for kw in ["Mkt", "SMB", "HML", "Mom", "RF"]):
            header_idx = i
            break

    # Read from header_idx onwards, stop at empty lines (footer)
    data_lines = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("Copyright"):
            break
        data_lines.append(stripped)

    csv_text = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d",
                               errors="coerce")
    df = df[df.index.notna()]
    df.columns = [c.strip() for c in df.columns]
    # Rename to standard names
    rename_map = {
        "Mkt-RF": "Mkt-RF",
        "Mom   ": "Mom",
        "Mom": "Mom",
    }
    df.rename(columns=rename_map, inplace=True)
    return df / 100.0  # percent → decimal


def _ff_via_direct_http(model: str, start_dt, end_dt) -> pd.DataFrame:
    """Download FF factors via direct HTTP requests with manual CSV parsing."""
    base_key = "FF3" if model == "FF3" else "FF5"
    ff_df = _download_ff_zip(base_key)

    if "FF5+MOM" in model:
        mom_df = _download_ff_zip("MOM")
        # Standardise momentum column name
        mom_col = [c for c in mom_df.columns if "mom" in c.lower()]
        if mom_col:
            mom_df = mom_df[[mom_col[0]]].rename(columns={mom_col[0]: "Mom"})
        ff_df = ff_df.join(mom_df, how="inner")

    # Filter date range
    ff_df = ff_df.loc[
        (ff_df.index >= pd.to_datetime(start_dt)) &
        (ff_df.index <= pd.to_datetime(end_dt))
    ]
    return ff_df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FAMA-FRENCH — LEVEL 3: ETF-based proxy factors
# ─────────────────────────────────────────────────────────────────────────────
def _ff_via_etf_proxies(model: str, start: str, end: str) -> pd.DataFrame:
    """
    Construct approximate Fama-French factors from liquid ETF spreads.
    These are approximations — useful when the French server is unreachable.

    SMB  ≈ Return(IWM) - Return(IVV)     [small - large]
    HML  ≈ Return(IVE) - Return(IVW)     [value - growth]
    RMW  ≈ Return(QUAL) - Return(SPY)    [quality - market]
    CMA  ≈ Return(USMV) - Return(SPY)    [low-inv - market]
    Mom  ≈ Return(MTUM) - Return(SPY)    [momentum - market]
    RF   ≈ ^IRX / 252
    """
    needed = {
        "market": "SPY", "small": "IWM", "large": "IVV",
        "value": "IVE", "growth": "IVW", "qual": "QUAL",
        "lowvol": "USMV", "mom": "MTUM", "rf": "^IRX",
    }
    tickers = list(set(needed.values()))

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    ret = np.log(raw / raw.shift(1)).dropna()

    def col(name):
        t = needed[name]
        return ret[t] if t in ret.columns else pd.Series(0.0, index=ret.index)

    # Risk-free daily rate
    rf_annual = raw["^IRX"] / 100.0 if "^IRX" in raw.columns else pd.Series(0.04, index=raw.index)
    rf_daily = ((1 + rf_annual) ** (1 / 252) - 1).reindex(ret.index).ffill()

    factors = pd.DataFrame(index=ret.index)
    factors["Mkt-RF"] = col("market") - rf_daily
    factors["SMB"]    = col("small")  - col("large")
    factors["HML"]    = col("value")  - col("growth")
    factors["RF"]     = rf_daily

    if model in ("FF5", "FF5+MOM"):
        factors["RMW"] = col("qual")   - col("market")
        factors["CMA"] = col("lowvol") - col("market")

    if "MOM" in model:
        factors["Mom"] = col("mom") - col("market")

    return factors.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — get_fama_french_factors (with 3-level fallback)
# ─────────────────────────────────────────────────────────────────────────────
def get_fama_french_factors(model: str = "FF5",
                             start: str = "2015-01-01",
                             end: str = None) -> tuple[pd.DataFrame, str]:
    """
    Download Fama-French factors with 3-level fallback:
      1. pandas-datareader (official Dartmouth connection)
      2. Direct HTTP download + manual CSV parse (bypasses datareader timeout)
      3. ETF proxy factors from yfinance (always available)

    Returns
    -------
    (DataFrame of factors, source_label)
    source_label ∈ {"datareader", "direct_http", "etf_proxy"}
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)

    # ── Level 1: pandas-datareader
    try:
        df = _ff_via_datareader(model, start_dt, end_dt)
        return df, "datareader"
    except Exception as e1:
        pass

    # ── Level 2: Direct HTTP
    try:
        df = _ff_via_direct_http(model, start_dt, end_dt)
        return df, "direct_http"
    except Exception as e2:
        pass

    # ── Level 3: ETF proxies (always works, needs only yfinance)
    try:
        df = _ff_via_etf_proxies(model, start, end)
        return df, "etf_proxy"
    except Exception as e3:
        raise RuntimeError(
            f"All 3 methods to load Fama-French factors failed.\n"
            f"  datareader: connection error\n"
            f"  direct_http: connection error\n"
            f"  etf_proxy: {e3}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTAL DATA (Yahoo Finance)
# ─────────────────────────────────────────────────────────────────────────────
def get_fundamentals(ticker: str) -> dict:
    """Extract fundamental data for a single ticker via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        bs    = stock.balance_sheet
        income = stock.income_stmt

        market_cap       = info.get("marketCap", np.nan)
        enterprise_value = info.get("enterpriseValue", np.nan)
        ebitda           = info.get("ebitda", np.nan)
        trailing_pe      = info.get("trailingPE", np.nan)
        forward_pe       = info.get("forwardPE", np.nan)
        ev_ebitda = enterprise_value / ebitda if (ebitda and ebitda != 0) else np.nan

        ebit = None
        if income is not None and not income.empty:
            for label in ["EBIT", "Operating Income", "Operating Income or Loss"]:
                if label in income.index:
                    ebit = float(income.loc[label].iloc[0])
                    break
        ev_ebit = enterprise_value / ebit if (ebit and ebit != 0) else np.nan

        roe = info.get("returnOnEquity", np.nan)
        roa = info.get("returnOnAssets", np.nan)

        roic = np.nan
        nopat = None
        invested_capital = None
        if income is not None and not income.empty and bs is not None and not bs.empty:
            try:
                tax_rate = info.get("effectiveTaxRate", 0.25) or 0.25
                if ebit is not None:
                    nopat = ebit * (1 - tax_rate)
                te = float(bs.loc["Stockholders Equity"].iloc[0]) if "Stockholders Equity" in bs.index else np.nan
                td_labels   = ["Total Debt", "Long Term Debt"]
                cash_labels = ["Cash And Cash Equivalents", "Cash"]
                td   = next((float(bs.loc[l].iloc[0]) for l in td_labels   if l in bs.index), np.nan)
                cash = next((float(bs.loc[l].iloc[0]) for l in cash_labels if l in bs.index), np.nan)
                invested_capital = te + td - cash
                if nopat and invested_capital and invested_capital != 0:
                    roic = nopat / invested_capital
            except Exception:
                pass

        beta          = float(info.get("beta", 1.0) or 1.0)
        total_debt_raw = info.get("totalDebt", np.nan)
        cost_of_debt   = info.get("averageInterestRate", np.nan)

        net_debt = np.nan
        if total_debt_raw is not None:
            try:
                cash_val = float(info.get("totalCash", 0) or 0)
                net_debt = float(total_debt_raw) - cash_val
            except Exception:
                pass

        net_debt_ebitda = (net_debt / ebitda
                           if (ebitda and ebitda != 0 and not np.isnan(net_debt))
                           else np.nan)

        return {
            "name":             info.get("longName", ticker),
            "sector":           info.get("sector", "Unknown"),
            "industry":         info.get("industry", "Unknown"),
            "market_cap":       market_cap,
            "enterprise_value": enterprise_value,
            "ebitda":           ebitda,
            "ebit":             ebit,
            "ev_ebitda":        ev_ebitda,
            "ev_ebit":          ev_ebit,
            "trailing_pe":      trailing_pe,
            "forward_pe":       forward_pe,
            "roe":              roe,
            "roa":              roa,
            "roic":             roic,
            "beta":             beta,
            "total_equity":     info.get("marketCap", np.nan),
            "total_debt":       total_debt_raw,
            "net_debt":         net_debt,
            "net_debt_ebitda":  net_debt_ebitda,
            "cost_of_debt":     cost_of_debt,
            "nopat":            nopat,
            "invested_capital": invested_capital,
        }
    except Exception as e:
        return {"error": str(e), "name": ticker}


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_volatility_regimes(returns: pd.Series, window: int = 21,
                               threshold_pct: float = 75.0) -> pd.Series:
    """Classify dates as HIGH or LOW volatility. Returns boolean Series."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    cutoff = np.nanpercentile(rolling_vol.dropna(), threshold_pct)
    return rolling_vol >= cutoff
