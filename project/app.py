"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ESG vs Classic Portfolio Analyser — Streamlit Application          ║
║         Performance · Risk · Factor Models · Fundamentals · AI Agent       ║
╚══════════════════════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ── Path setup so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

# ── Internal modules
from data.data_loader import (
    download_prices, get_returns, get_risk_free_rate,
    get_fama_french_factors, get_fundamentals,
    detect_volatility_regimes,
    DEFAULT_ESG_TICKERS, DEFAULT_CLASSIC_TICKERS,
)
from analysis.risk_metrics import (
    full_risk_summary, compute_drawdown_series,
    compute_var, compute_cvar,
)
from analysis.factor_models import (
    run_capm, run_ff3, run_ff5, run_ff5_momentum,
    rolling_beta, rolling_correlation, rolling_volatility,
    regime_analysis,
)
from analysis.fundamental import (
    compute_wacc_from_fundamentals,
    compute_lbo_metrics,
    fundamental_summary_df,
)
from ai_agent.analyzer import run_ai_analysis, generate_regime_commentary, test_api_connection
from utils.plotting import (
    plot_cumulative_returns, plot_drawdowns,
    plot_rolling, plot_return_distribution,
    plot_radar, plot_factor_loadings,
    plot_metrics_heatmap, plot_wacc_waterfall,
    plot_correlation_matrix, plot_regime_comparison,
    COLORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — persiste les résultats entre les reruns Streamlit
# ─────────────────────────────────────────────────────────────────────────────
if "analysis_done"       not in st.session_state: st.session_state.analysis_done       = False
if "esg_summary"         not in st.session_state: st.session_state.esg_summary         = None
if "classic_summary"     not in st.session_state: st.session_state.classic_summary     = None
if "esg_ret"             not in st.session_state: st.session_state.esg_ret             = None
if "classic_ret"         not in st.session_state: st.session_state.classic_ret         = None
if "market_ret"          not in st.session_state: st.session_state.market_ret          = None
if "rf_series"           not in st.session_state: st.session_state.rf_series           = None
if "ff_results"          not in st.session_state: st.session_state.ff_results          = {}
if "ff_available"        not in st.session_state: st.session_state.ff_available        = False
if "ff_source"           not in st.session_state: st.session_state.ff_source           = None
if "esg_dd"              not in st.session_state: st.session_state.esg_dd              = None
if "classic_dd"          not in st.session_state: st.session_state.classic_dd          = None
if "r_vol_esg"           not in st.session_state: st.session_state.r_vol_esg           = None
if "r_vol_classic"       not in st.session_state: st.session_state.r_vol_classic       = None
if "r_beta_esg"          not in st.session_state: st.session_state.r_beta_esg          = None
if "r_beta_classic"      not in st.session_state: st.session_state.r_beta_classic      = None
if "r_corr_esg"          not in st.session_state: st.session_state.r_corr_esg          = None
if "r_corr_classic"      not in st.session_state: st.session_state.r_corr_classic      = None
if "regime_esg"          not in st.session_state: st.session_state.regime_esg          = None
if "regime_classic"      not in st.session_state: st.session_state.regime_classic      = None
if "regime_mask"         not in st.session_state: st.session_state.regime_mask         = None
if "prices"              not in st.session_state: st.session_state.prices              = None
if "ai_analysis_text"    not in st.session_state: st.session_state.ai_analysis_text    = None
if "ai_analysis_source"  not in st.session_state: st.session_state.ai_analysis_source  = None
if "last_config"         not in st.session_state: st.session_state.last_config         = None

st.set_page_config(
    page_title="ESG Analytics Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global dark theme overrides */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0E1117; }

/* ── Metric cards */
.metric-card {
    background: #1E2130;
    border: 1px solid #2A2D3E;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00C49A; }
.metric-label { font-size: 0.78rem; color: #8B8FA8; text-transform: uppercase;
                letter-spacing: 0.08em; margin-bottom: 6px; }
.metric-value { font-size: 1.65rem; font-weight: 700; color: #E0E0E0; }
.metric-delta-pos { font-size: 0.82rem; color: #00C49A; margin-top: 4px; }
.metric-delta-neg { font-size: 0.82rem; color: #FF6B6B; margin-top: 4px; }

/* ── Section headers */
.section-header {
    border-left: 4px solid #00C49A;
    padding-left: 14px;
    margin: 30px 0 18px 0;
    color: #E0E0E0;
    font-size: 1.3rem;
    font-weight: 600;
}

/* ── AI output */
.ai-box {
    background: #1A1D2E;
    border: 1px solid #2F3450;
    border-radius: 12px;
    padding: 24px;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #C8CCD8;
}

/* ── Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #1E2130;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: #8B8FA8;
}
.stTabs [aria-selected="true"] {
    background-color: #2A2D3E !important;
    color: #00C49A !important;
}

/* ── Sidebar */
section[data-testid="stSidebar"] { background-color: #13151F; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value, delta=None, fmt: str = ".2f") -> str:
    val_str = f"{value:{fmt}}" if isinstance(value, (int, float)) and not np.isnan(float(value if value else 0)) else "N/A"
    delta_html = ""
    if delta is not None:
        try:
            d = float(delta)
            cls = "metric-delta-pos" if d >= 0 else "metric-delta-neg"
            sign = "▲" if d >= 0 else "▼"
            delta_html = f'<div class="{cls}">{sign} {abs(d):.2f}</div>'
        except (TypeError, ValueError):
            pass
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val_str}</div>
        {delta_html}
    </div>"""


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    return download_prices(tickers, start, end)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_ff_factors(model: str, start: str, end: str) -> tuple:
    return get_fama_french_factors(model, start, end)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_rf(start: str, end: str) -> pd.Series:
    return get_risk_free_rate(start, end)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_fundamentals(ticker: str) -> dict:
    return get_fundamentals(ticker)


def fmt_pct(v, decimals=2):
    try:
        if np.isnan(float(v)):
            return "N/A"
        return f"{float(v):.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_num(v, decimals=3):
    try:
        if np.isnan(float(v)):
            return "N/A"
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 ESG Analytics Platform")
    st.markdown("---")

    st.markdown("### 📅 Time Period")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=date(2018, 1, 1),
                                    min_value=date(2010, 1, 1))
    with col_e:
        end_date = st.date_input("End", value=date.today())

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    st.markdown("### 📊 Portfolios")

    # ESG portfolio
    esg_preset = st.selectbox("ESG Portfolio (preset)",
                               options=list(DEFAULT_ESG_TICKERS.keys()),
                               index=0)
    esg_custom = st.text_input("Or enter custom ESG ticker", "")
    esg_ticker = esg_custom.upper().strip() if esg_custom else DEFAULT_ESG_TICKERS[esg_preset]
    esg_name = esg_custom if esg_custom else esg_preset

    # Classic portfolio
    classic_preset = st.selectbox("Classic Portfolio (preset)",
                                   options=list(DEFAULT_CLASSIC_TICKERS.keys()),
                                   index=1)
    classic_custom = st.text_input("Or enter custom Classic ticker", "")
    classic_ticker = classic_custom.upper().strip() if classic_custom else DEFAULT_CLASSIC_TICKERS[classic_preset]
    classic_name = classic_custom if classic_custom else classic_preset

    st.markdown("### ⚙️ Parameters")
    return_method = st.radio("Return method", ["Log", "Simple"], horizontal=True)
    confidence_var = st.slider("VaR / CVaR confidence", 0.90, 0.99, 0.95, 0.01)
    rolling_window = st.slider("Rolling window (days)", 21, 252, 63, 21)

    st.markdown("### 🧮 Factor Model")
    ff_model = st.selectbox("Fama-French Model",
                             ["FF5 + Momentum", "FF5", "FF3", "CAPM"])

    st.markdown("### 🤖 Agent IA")
    api_key_input = st.text_input(
        "Clé API Anthropic (optionnelle)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Sans clé : analyse locale gratuite. "
             "Avec clé : analyse enrichie par Claude. "
             "console.anthropic.com"
    )
    use_ai = True  # Always enabled — local mode requires no key
    if api_key_input:
        if st.button("🔌 Tester la connexion API", use_container_width=True):
            ok, msg = test_api_connection(api_key_input)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("### 📋 Fundamental Ticker")
    fund_ticker = st.text_input("Fundamentals ticker", "AAPL",
                                 help="Single stock for WACC & LBO analysis")

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 30px 0 20px 0;">
    <h1 style="color:#E0E0E0; font-size:2.2rem; margin-bottom:8px;">
        🌱 ESG vs Classic Portfolio Analyser
    </h1>
    <p style="color:#8B8FA8; font-size:1rem; max-width:700px;">
        Compare risk-adjusted performance of ESG and conventional portfolios using
        multi-factor models, tail risk analysis, and AI-powered insights.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGIC : Run analysis or restore from session_state
# ─────────────────────────────────────────────────────────────────────────────
current_config = (esg_ticker, classic_ticker, start_str, end_str, return_method, ff_model)
config_changed = (current_config != st.session_state.last_config)

# Show welcome screen only when nothing has been computed yet
if not run_button and not st.session_state.analysis_done:
    st.info("👈 Configurez vos portefeuilles dans la barre latérale et cliquez sur **🚀 Run Analysis**.")
    st.markdown("""
    #### Fonctionnalités
    - **Télécharge** les prix via `yfinance` et les facteurs Fama-French (3 niveaux de fallback)
    - **Calcule** Sharpe, Sortino, Treynor, VaR, CVaR, Max Drawdown, Skewness, Kurtosis, Beta...
    - **Régresse** CAPM, FF3, FF5, FF5+Momentum pour tester l'alpha ESG
    - **Analyse** les régimes de volatilité : comportement ESG en période de stress
    - **Extrait** fondamentaux (P/E, EV/EBITDA, ROE, ROIC) et calcule le WACC
    - **Génère** une analyse IA complète (locale gratuite ou via Claude API)
    """)
    st.stop()

# ── (Re)compute when Run is clicked or config changed
# ret_method must be defined before the if block so tabs can use it
ret_method = "log" if return_method == "Log" else "simple"

if run_button or config_changed:
    st.session_state.last_config = current_config

    with st.spinner("📡 Téléchargement des données de marché..."):
        try:
            tickers = [esg_ticker, classic_ticker, "SPY"]
            prices = cached_prices(tickers, start_str, end_str)
            if esg_ticker not in prices.columns or classic_ticker not in prices.columns:
                st.error(f"Impossible de récupérer les données pour {esg_ticker} ou {classic_ticker}.")
                st.stop()

            ret_method = "log" if return_method == "Log" else "simple"
            all_returns = get_returns(prices, ret_method)
            esg_ret     = all_returns[esg_ticker].dropna()
            classic_ret = all_returns[classic_ticker].dropna()
            market_ret  = all_returns["SPY"].dropna()
            common_idx  = esg_ret.index.intersection(classic_ret.index).intersection(market_ret.index)
            esg_ret     = esg_ret.loc[common_idx]
            classic_ret = classic_ret.loc[common_idx]
            market_ret  = market_ret.loc[common_idx]
            rf_series   = cached_rf(start_str, end_str).reindex(common_idx).ffill().fillna(0.0001)
            st.session_state.prices      = prices
            st.session_state.esg_ret     = esg_ret
            st.session_state.classic_ret = classic_ret
            st.session_state.market_ret  = market_ret
            st.session_state.rf_series   = rf_series
        except Exception as e:
            st.error(f"❌ Erreur de données : {e}")
            st.stop()

    with st.spinner("📚 Chargement des facteurs Fama-French..."):
        ff_model_map = {"CAPM":"FF3","FF3":"FF3","FF5":"FF5","FF5 + Momentum":"FF5+MOM"}
        try:
            ff_factors, ff_source = cached_ff_factors(ff_model_map.get(ff_model,"FF5"), start_str, end_str)
            ff_available = True
            _SRC = {"datareader":"✅ French Data Library (officiel)",
                    "direct_http":"✅ French Data Library (direct)",
                    "etf_proxy":"⚠️ Facteurs ETF proxy (Dartmouth inaccessible)"}
            st.sidebar.markdown(f"**FF Source :** {_SRC.get(ff_source, ff_source)}")
        except Exception as e:
            st.warning(f"Facteurs Fama-French non disponibles : {e}")
            ff_available = False
            ff_factors   = None
            ff_source    = None
        st.session_state.ff_available = ff_available
        st.session_state.ff_source    = ff_source

    with st.spinner("🔢 Calcul des métriques de risque..."):
        try:
            esg_summary     = full_risk_summary(esg_name, esg_ret, market_ret, rf_series,
                                                 benchmark_returns=classic_ret, confidence=confidence_var)
            classic_summary = full_risk_summary(classic_name, classic_ret, market_ret, rf_series,
                                                 benchmark_returns=esg_ret, confidence=confidence_var)
        except Exception as e:
            import traceback as tb
            st.error(f"❌ Erreur métriques : {e}")
            st.code(tb.format_exc())
            st.stop()

        esg_dd      = compute_drawdown_series(esg_ret)
        classic_dd  = compute_drawdown_series(classic_ret)
        r_vol_esg   = rolling_volatility(esg_ret, rolling_window)
        r_vol_cls   = rolling_volatility(classic_ret, rolling_window)
        r_beta_esg  = rolling_beta(esg_ret, market_ret, rolling_window)
        r_beta_cls  = rolling_beta(classic_ret, market_ret, rolling_window)
        r_corr_esg  = rolling_correlation(esg_ret, market_ret, rolling_window)
        r_corr_cls  = rolling_correlation(classic_ret, market_ret, rolling_window)
        regime_mask = detect_volatility_regimes(market_ret, window=21, threshold_pct=75)
        regime_esg  = regime_analysis(esg_ret,     market_ret, rf_series, regime_mask, esg_name)
        regime_cls  = regime_analysis(classic_ret, market_ret, rf_series, regime_mask, classic_name)

        st.session_state.esg_summary    = esg_summary
        st.session_state.classic_summary= classic_summary
        st.session_state.esg_dd         = esg_dd
        st.session_state.classic_dd     = classic_dd
        st.session_state.r_vol_esg      = r_vol_esg
        st.session_state.r_vol_classic  = r_vol_cls
        st.session_state.r_beta_esg     = r_beta_esg
        st.session_state.r_beta_classic = r_beta_cls
        st.session_state.r_corr_esg     = r_corr_esg
        st.session_state.r_corr_classic = r_corr_cls
        st.session_state.regime_mask    = regime_mask
        st.session_state.regime_esg     = regime_esg
        st.session_state.regime_classic = regime_cls

    ff_results = {}
    if ff_available and ff_factors is not None:
        with st.spinner("🔬 Régressions factorielles..."):
            try:
                if ff_model == "CAPM":
                    ff_results = {"CAPM": run_capm(esg_ret, ff_factors),
                                  f"CAPM ({classic_name})": run_capm(classic_ret, ff_factors)}
                elif ff_model == "FF3":
                    ff_results = {esg_name: run_ff3(esg_ret, ff_factors),
                                  classic_name: run_ff3(classic_ret, ff_factors)}
                elif ff_model == "FF5":
                    ff_results = {esg_name: run_ff5(esg_ret, ff_factors),
                                  classic_name: run_ff5(classic_ret, ff_factors)}
                else:
                    ff_results = {esg_name: run_ff5_momentum(esg_ret, ff_factors),
                                  classic_name: run_ff5_momentum(classic_ret, ff_factors)}
            except Exception as e:
                st.warning(f"Erreur modèle factoriel : {e}")
    st.session_state.ff_results   = ff_results
    st.session_state.analysis_done = True
    # Reset saved AI text when recomputing
    st.session_state.ai_analysis_text   = None
    st.session_state.ai_analysis_source = None

# ── Restore from session_state (for all subsequent reruns)
prices          = st.session_state.prices
esg_ret         = st.session_state.esg_ret
classic_ret     = st.session_state.classic_ret
market_ret      = st.session_state.market_ret
rf_series       = st.session_state.rf_series
esg_summary     = st.session_state.esg_summary
classic_summary = st.session_state.classic_summary
esg_dd          = st.session_state.esg_dd
classic_dd      = st.session_state.classic_dd
r_vol_esg       = st.session_state.r_vol_esg
r_vol_classic   = st.session_state.r_vol_classic
r_beta_esg      = st.session_state.r_beta_esg
r_beta_classic  = st.session_state.r_beta_classic
r_corr_esg      = st.session_state.r_corr_esg
r_corr_classic  = st.session_state.r_corr_classic
regime_mask     = st.session_state.regime_mask
regime_esg      = st.session_state.regime_esg
regime_classic  = st.session_state.regime_classic
ff_results      = st.session_state.ff_results
ff_available    = st.session_state.ff_available
ff_source       = st.session_state.ff_source


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab_overview, tab_risk, tab_factor,
 tab_regimes, tab_fundamental, tab_lbo, tab_ai) = st.tabs([
    "📊 Overview",
    "⚠️ Risk Analysis",
    "🔬 Factor Models",
    "🌊 Volatility Regimes",
    "🏦 Fundamentals",
    "💼 LBO / PE",
    "🤖 AI Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    section_header("Portfolio Performance Dashboard")

    # ── KPI Row
    cols = st.columns(6)
    kpis = [
        ("ESG Ann. Return", esg_summary["Ann. Return (%)"], "%"),
        ("Classic Ann. Return", classic_summary["Ann. Return (%)"], "%"),
        ("ESG Sharpe", esg_summary["Sharpe Ratio"], ""),
        ("Classic Sharpe", classic_summary["Sharpe Ratio"], ""),
        ("ESG Max DD", esg_summary["Max Drawdown (%)"], "%"),
        ("Classic Max DD", classic_summary["Max Drawdown (%)"], "%"),
    ]
    for col, (label, val, suffix) in zip(cols, kpis):
        try:
            fval = float(val)
            delta = None
            if "ESG" in label and "Classic" not in label:
                # Compute diff vs classic equivalent
                classic_key = label.replace("ESG", "Classic")
                classic_val = next((v for l, v, _ in kpis if l == classic_key), None)
                if classic_val is not None:
                    delta = float(val) - float(classic_val)
            display = f"{fval:.2f}{suffix}"
        except (TypeError, ValueError):
            display = "N/A"
            delta = None

        is_negative = False
        try:
            is_negative = float(val) < 0
        except Exception:
            pass

        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{'#FF6B6B' if is_negative else '#E0E0E0'}">{display}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cumulative returns chart
    prices_norm = prices[[esg_ticker, classic_ticker, "SPY"]].dropna()
    esg_r = get_returns(prices_norm[[esg_ticker]], ret_method).iloc[:, 0]
    classic_r = get_returns(prices_norm[[classic_ticker]], ret_method).iloc[:, 0]
    mkt_r = get_returns(prices_norm[["SPY"]], ret_method).iloc[:, 0]

    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig_cum = plot_cumulative_returns(esg_r, classic_r, mkt_r,
                                          esg_name, classic_name)
        st.plotly_chart(fig_cum, use_container_width=True)
    with col_r:
        fig_radar = plot_radar(esg_summary, classic_summary, esg_name, classic_name)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Metrics comparison table
    section_header("Key Metrics Comparison")
    metric_rows = [
        ("Annual Return (%)", esg_summary["Ann. Return (%)"], classic_summary["Ann. Return (%)"]),
        ("Total Return (%)", esg_summary["Total Return (%)"], classic_summary["Total Return (%)"]),
        ("Ann. Volatility (%)", esg_summary["Ann. Volatility (%)"], classic_summary["Ann. Volatility (%)"]),
        ("Beta", esg_summary["Beta"], classic_summary["Beta"]),
        ("Sharpe Ratio", esg_summary["Sharpe Ratio"], classic_summary["Sharpe Ratio"]),
        ("Sortino Ratio", esg_summary["Sortino Ratio"], classic_summary["Sortino Ratio"]),
        ("Treynor Ratio", esg_summary["Treynor Ratio"], classic_summary["Treynor Ratio"]),
        ("Calmar Ratio", esg_summary.get("Calmar Ratio"), classic_summary.get("Calmar Ratio")),
        ("Max Drawdown (%)", esg_summary["Max Drawdown (%)"], classic_summary["Max Drawdown (%)"]),
        (f"VaR {int(confidence_var*100)}% (%)", esg_summary[f"VaR {int(confidence_var*100)}% (%)"],
         classic_summary[f"VaR {int(confidence_var*100)}% (%)"]),
        (f"CVaR {int(confidence_var*100)}% (%)", esg_summary[f"CVaR {int(confidence_var*100)}% (%)"],
         classic_summary[f"CVaR {int(confidence_var*100)}% (%)"]),
        ("CAPM Alpha (%)", esg_summary["CAPM Alpha (%)"], classic_summary["CAPM Alpha (%)"]),
        ("Skewness", esg_summary["Skewness"], classic_summary["Skewness"]),
        ("Excess Kurtosis", esg_summary["Excess Kurtosis"], classic_summary["Excess Kurtosis"]),
    ]

    def arrow(e, c):
        try:
            return "🟢" if float(e) > float(c) else ("🔴" if float(e) < float(c) else "⚪")
        except Exception:
            return "⚪"

    tbl = pd.DataFrame(
        [(m, fmt_num(e, 3), fmt_num(c, 3), arrow(e, c))
         for m, e, c in metric_rows],
        columns=["Metric", esg_name, classic_name, "ESG Edge"]
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    section_header("Drawdown & Tail Risk")

    c1, c2 = st.columns(2)
    with c1:
        fig_dd = plot_drawdowns(esg_dd, classic_dd, esg_name, classic_name)
        st.plotly_chart(fig_dd, use_container_width=True)
    with c2:
        fig_dist = plot_return_distribution(esg_ret, classic_ret, esg_name, classic_name,
                                             confidence_var)
        st.plotly_chart(fig_dist, use_container_width=True)

    section_header("Rolling Risk Metrics")
    metric_choice = st.selectbox(
        "Select rolling metric",
        ["Annualised Volatility", "Beta vs Market", "Correlation vs Market"],
        key="rolling_choice"
    )

    if metric_choice == "Annualised Volatility":
        fig_roll = plot_rolling(r_vol_esg, r_vol_classic, "Rolling Volatility",
                                 "Ann. Volatility", esg_name, classic_name, pct=True)
    elif metric_choice == "Beta vs Market":
        fig_roll = plot_rolling(r_beta_esg, r_beta_classic, "Rolling Beta vs Market",
                                 "Beta", esg_name, classic_name)
    else:
        fig_roll = plot_rolling(r_corr_esg, r_corr_classic, "Rolling Correlation vs Market",
                                 "Correlation", esg_name, classic_name)

    st.plotly_chart(fig_roll, use_container_width=True)

    # ── VaR / CVaR detail
    section_header("Tail Risk Detail")
    var_cols = st.columns(3)
    for conf in [0.90, 0.95, 0.99]:
        idx = int((conf - 0.90) / 0.05)  # 0,1,2
        if idx < 3:
            e_var = compute_var(esg_ret, conf)
            c_var = compute_var(classic_ret, conf)
            e_cvar = compute_cvar(esg_ret, conf)
            c_cvar = compute_cvar(classic_ret, conf)
            with var_cols[idx]:
                st.markdown(f"**VaR {int(conf*100)}%**")
                st.markdown(f"""
                | | {esg_name} | {classic_name} |
                |--|--|--|
                | VaR | {fmt_pct(e_var*100)} | {fmt_pct(c_var*100)} |
                | CVaR | {fmt_pct(e_cvar*100)} | {fmt_pct(c_cvar*100)} |
                """)

    # ── Returns correlation
    section_header("Portfolio Correlations")
    returns_for_corr = pd.concat([esg_ret, classic_ret, market_ret],
                                  axis=1, keys=[esg_name, classic_name, "Market (SPY)"])
    fig_corr = plot_correlation_matrix(returns_for_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Statistical moments detail
    section_header("Return Distribution Statistics")
    moments_data = {
        "Statistic": ["Mean (daily %)", "Std Dev (daily %)", "Skewness",
                      "Excess Kurtosis", "Min (%)", "Max (%)",
                      "Jarque-Bera p-value"],
        esg_name: [
            fmt_pct(esg_ret.mean() * 100, 4),
            fmt_pct(esg_ret.std() * 100, 4),
            fmt_num(esg_ret.skew()),
            fmt_num(esg_ret.kurtosis()),
            fmt_pct(esg_ret.min() * 100),
            fmt_pct(esg_ret.max() * 100),
            fmt_num(__import__('scipy').stats.jarque_bera(esg_ret)[1]),
        ],
        classic_name: [
            fmt_pct(classic_ret.mean() * 100, 4),
            fmt_pct(classic_ret.std() * 100, 4),
            fmt_num(classic_ret.skew()),
            fmt_num(classic_ret.kurtosis()),
            fmt_pct(classic_ret.min() * 100),
            fmt_pct(classic_ret.max() * 100),
            fmt_num(__import__('scipy').stats.jarque_bera(classic_ret)[1]),
        ],
    }
    st.dataframe(pd.DataFrame(moments_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FACTOR MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_factor:
    if not ff_available or not ff_results:
        st.warning("Factor data unavailable. Check internet connection.")
    else:
        section_header(f"Factor Model Results — {ff_model}")
        if ff_source == "etf_proxy":
            st.warning(
                "⚠️ ETF Proxy Factors — Dartmouth server unreachable. "
                "Factors approximated from ETF spreads: "
                "Mkt-RF=SPY-RF, SMB=IWM-IVV, HML=IVE-IVW, "
                "RMW=QUAL-SPY, CMA=USMV-SPY, Mom=MTUM-SPY. "
                "Results are directionally informative but approximate."
            )

        # ── Alpha comparison
        names = list(ff_results.keys())
        results_list = list(ff_results.values())

        for name_f, res in zip(names, results_list):
            if "error" in res:
                st.error(f"{name_f}: {res['error']}")
                continue

            alpha = res.get("alpha_annual_pct", np.nan)
            tstat = res.get("alpha_tstat", np.nan)
            pval = res.get("alpha_pvalue", np.nan)
            r2 = res.get("r_squared", np.nan)
            sig = res.get("alpha_significant", False)

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric(f"{name_f} — Alpha (annual)", fmt_pct(alpha))
            col_b.metric("Alpha t-stat", fmt_num(tstat))
            col_c.metric("p-value", fmt_num(pval))
            col_d.metric("R²", fmt_num(r2))
            if sig:
                st.success(f"✅ Alpha is **statistically significant** (p < 0.05) for {name_f}")
            else:
                st.info(f"ℹ️ Alpha is **not statistically significant** (p = {fmt_num(pval)}) for {name_f}")

            st.markdown("**Factor Loadings:**")
            if res.get("factors"):
                factor_rows = []
                for factor, fdata in res["factors"].items():
                    factor_rows.append({
                        "Factor": factor,
                        "Loading (β)": fmt_num(fdata["loading"]),
                        "t-stat": fmt_num(fdata["t_stat"]),
                        "p-value": fmt_num(fdata["p_value"]),
                        "95% CI Lower": fmt_num(fdata["ci_lower"]),
                        "95% CI Upper": fmt_num(fdata["ci_upper"]),
                        "Significant": "✅" if fdata["significant"] else "❌",
                    })
                st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
            st.markdown("---")

        # ── Factor loadings chart
        if len(names) >= 2 and "error" not in results_list[0] and "error" not in results_list[1]:
            section_header("Factor Loadings Comparison")
            fig_fl = plot_factor_loadings(results_list[0], results_list[1], names[0], names[1])
            st.plotly_chart(fig_fl, use_container_width=True)

        # ── Fama-French interpretation
        with st.expander("📘 Factor Interpretation Guide"):
            st.markdown("""
| Factor | Description | High ESG Loading → |
|--------|-------------|-------------------|
| **Mkt-RF** | Market excess return | Systematic market sensitivity |
| **SMB** | Small Minus Big | Small-cap tilt (negative = large-cap bias common in ESG) |
| **HML** | High Minus Low (value) | Negative = growth tilt (ESG often avoids deep value/energy) |
| **RMW** | Robust Minus Weak (profitability) | Positive = quality tilt in ESG firms |
| **CMA** | Conservative Minus Aggressive (investment) | Positive = conservative capex (ESG governance effect) |
| **Mom** | Momentum | Tests if ESG performance is driven by momentum |

A **positive, significant alpha** after controlling for all factors would constitute evidence of a genuine ESG risk premium (or mis-pricing).
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VOLATILITY REGIMES
# ══════════════════════════════════════════════════════════════════════════════
with tab_regimes:
    section_header("Volatility Regime Analysis")

    n_high = int(regime_mask.sum())
    n_low = int((~regime_mask).sum())
    pct_high = n_high / (n_high + n_low) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("High Volatility Days", f"{n_high:,}")
    col2.metric("Low Volatility Days", f"{n_low:,}")
    col3.metric("% in High Vol Regime", f"{pct_high:.1f}%")

    # ── Regime metrics comparison
    section_header("Performance by Regime")
    if regime_esg and regime_classic:
        # Build comparison df
        all_metrics_reg = ["ann_return_pct", "ann_vol_pct", "beta", "sharpe",
                            "max_dd_pct", "cvar_95_pct"]
        metric_labels = {
            "ann_return_pct": "Ann. Return (%)",
            "ann_vol_pct": "Ann. Volatility (%)",
            "beta": "Beta",
            "sharpe": "Sharpe Ratio",
            "max_dd_pct": "Max Drawdown (%)",
            "cvar_95_pct": "CVaR 95% (%)",
        }
        regime_rows = []
        for regime in ["High Volatility", "Low Volatility"]:
            for mkey, mlabel in metric_labels.items():
                e_val = regime_esg.get(regime, {}).get(mkey, np.nan)
                c_val = regime_classic.get(regime, {}).get(mkey, np.nan)
                regime_rows.append({
                    "Regime": regime,
                    "Metric": mlabel,
                    esg_name: fmt_num(e_val),
                    classic_name: fmt_num(c_val),
                })
        reg_df = pd.DataFrame(regime_rows)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        # ── Regime charts
        reg_metric = st.selectbox(
            "Metric to visualise by regime",
            ["sharpe", "ann_return", "ann_vol", "beta", "max_dd", "cvar"],
            key="regime_metric_viz"
        )
        fig_regime = plot_regime_comparison(regime_esg, regime_classic, reg_metric,
                                             esg_name, classic_name)
        st.plotly_chart(fig_regime, use_container_width=True)

    # ── High vol period overlay
    section_header("Volatility Overlay Chart")
    import plotly.graph_objects as go
    fig_vol = go.Figure()
    wealth_esg = 100 * (1 + esg_ret).cumprod()
    wealth_cls = 100 * (1 + classic_ret).cumprod()

    fig_vol.add_trace(go.Scatter(
        x=wealth_esg.index, y=wealth_esg,
        name=esg_name, line=dict(color=COLORS["esg"], width=2)
    ))
    fig_vol.add_trace(go.Scatter(
        x=wealth_cls.index, y=wealth_cls,
        name=classic_name, line=dict(color=COLORS["classic"], width=2)
    ))

    # Shade high-vol periods
    high_vol_dates = regime_mask[regime_mask].index
    if len(high_vol_dates) > 0:
        # Group consecutive dates
        gaps = (high_vol_dates.to_series().diff() > pd.Timedelta(days=5))
        groups = gaps.cumsum()
        for _, grp in high_vol_dates.to_series().groupby(groups):
            fig_vol.add_vrect(
                x0=grp.iloc[0], x1=grp.iloc[-1],
                fillcolor="rgba(255,200,0,0.08)",
                line_width=0,
                annotation_text="High Vol" if grp.iloc[0] == high_vol_dates[0] else "",
                annotation_position="top left",
            )

    fig_vol.update_layout(
        title="Cumulative Returns with High Volatility Periods Highlighted",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], title="Indexed Value"),
        legend=dict(bgcolor="rgba(30,33,48,0.8)"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── AI regime commentary
    if use_ai and api_key_input:
        section_header("🤖 AI Regime Commentary")
        with st.spinner("Génération du commentaire régimes..."):
            commentary, cmt_source = generate_regime_commentary(
                regime_esg, regime_classic,
                api_key=api_key_input
            )
        src_badge2 = ("🤖 **Claude AI**" if cmt_source == "claude_api"
                      else "🔢 **Analyse locale automatique**")
        st.caption(f"Source : {src_badge2}")
        st.markdown(commentary)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_fundamental:
    section_header(f"Fundamental Analysis — {fund_ticker.upper()}")

    with st.spinner(f"Loading fundamentals for {fund_ticker.upper()}..."):
        fund_data = cached_fundamentals(fund_ticker.upper())

    if "error" in fund_data:
        st.error(f"Error loading fundamentals: {fund_data['error']}")
    else:
        col_info, col_val, col_prof = st.columns(3)

        with col_info:
            st.markdown("**Company Info**")
            st.markdown(f"""
- **Name**: {fund_data.get('name', 'N/A')}
- **Sector**: {fund_data.get('sector', 'N/A')}
- **Industry**: {fund_data.get('industry', 'N/A')}
- **Market Cap**: ${(fund_data.get('market_cap') or 0)/1e9:.1f}B
- **Beta**: {fmt_num(fund_data.get('beta'))}
            """)

        with col_val:
            st.markdown("**Valuation Ratios**")
            st.markdown(f"""
- **EV/EBITDA**: {fmt_num(fund_data.get('ev_ebitda'))}
- **EV/EBIT**: {fmt_num(fund_data.get('ev_ebit'))}
- **Trailing P/E**: {fmt_num(fund_data.get('trailing_pe'))}
- **Forward P/E**: {fmt_num(fund_data.get('forward_pe'))}
- **Enterprise Value**: ${(fund_data.get('enterprise_value') or 0)/1e9:.1f}B
            """)

        with col_prof:
            st.markdown("**Profitability**")
            st.markdown(f"""
- **ROE**: {fmt_pct((fund_data.get('roe') or 0) * 100)}
- **ROA**: {fmt_pct((fund_data.get('roa') or 0) * 100)}
- **ROIC**: {fmt_pct((fund_data.get('roic') or 0) * 100) if fund_data.get('roic') else 'N/A'}
- **EBITDA**: ${(fund_data.get('ebitda') or 0)/1e6:.0f}M
- **Net Debt/EBITDA**: {fmt_num(fund_data.get('net_debt_ebitda'))}
            """)

        # ── WACC
        section_header("WACC Calculation")
        wacc_cols = st.columns([1, 2])
        with wacc_cols[0]:
            rf_input = st.number_input("Risk-Free Rate (%)", 0.0, 15.0, 4.0, 0.1) / 100
            erp_input = st.number_input("Equity Risk Premium (%)", 1.0, 15.0, 5.5, 0.1) / 100
            tax_input = st.number_input("Tax Rate (%)", 0.0, 50.0, 25.0, 1.0) / 100

            wacc_result = compute_wacc_from_fundamentals(
                fund_data, risk_free_rate=rf_input,
                erp=erp_input, tax_rate=tax_input
            )

            st.markdown(f"""
**WACC Results**
| Component | Value |
|-----------|-------|
| Cost of Equity (Re) | {fmt_pct(wacc_result.get('cost_of_equity_pct', 0))} |
| After-tax Cost of Debt | {fmt_pct(wacc_result.get('after_tax_cod_pct', 0))} |
| Equity Weight (E/V) | {fmt_pct(wacc_result.get('equity_weight', 0)*100)} |
| Debt Weight (D/V) | {fmt_pct(wacc_result.get('debt_weight', 0)*100)} |
| **WACC** | **{fmt_pct(wacc_result.get('wacc_pct', 0))}** |
            """)

        with wacc_cols[1]:
            fig_wacc = plot_wacc_waterfall(wacc_result,
                                            f"WACC Breakdown — {fund_ticker.upper()}")
            st.plotly_chart(fig_wacc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LBO / PE
# ══════════════════════════════════════════════════════════════════════════════
with tab_lbo:
    section_header("LBO / Private Equity Analysis")

    st.markdown("**Model Inputs** — adjust parameters to simulate a leveraged buyout.")

    c1, c2, c3 = st.columns(3)
    with c1:
        lbo_ebitda = st.number_input("EBITDA ($M)", 10.0, 10000.0, 100.0, 10.0)
        entry_multiple = st.number_input("Entry EV/EBITDA", 4.0, 30.0, 8.0, 0.5)
        exit_multiple = st.number_input("Exit EV/EBITDA", 4.0, 30.0, 8.5, 0.5)
    with c2:
        debt_ratio = st.slider("Leverage (Debt/EV)", 0.0, 0.9, 0.6, 0.05)
        holding_yrs = st.slider("Holding Period (years)", 1, 10, 5)
        int_rate = st.number_input("Interest Rate (%)", 1.0, 20.0, 7.0, 0.5) / 100
    with c3:
        ebitda_growth = st.number_input("EBITDA CAGR (%)", -10.0, 40.0, 5.0, 1.0) / 100
        lbo_tax = st.number_input("Tax Rate (%) ", 0.0, 50.0, 25.0, 1.0) / 100
        capex_pct = st.slider("CapEx / EBITDA (%)", 0, 40, 15) / 100

    lbo = compute_lbo_metrics(
        ebitda=lbo_ebitda * 1e6,
        entry_ev_multiple=entry_multiple,
        exit_ev_multiple=exit_multiple,
        debt_ratio=debt_ratio,
        holding_years=holding_yrs,
        interest_rate=int_rate,
        ebitda_growth_rate=ebitda_growth,
        tax_rate=lbo_tax,
        capex_to_ebitda=capex_pct,
    )

    # ── Results
    section_header("LBO Results")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("MOIC", fmt_num(lbo["moic"], 2) + "×")
    kpi2.metric("IRR", fmt_pct(lbo.get("irr_pct", np.nan)))
    kpi3.metric("Entry Equity ($M)", f"${lbo['entry_equity']/1e6:,.1f}M")
    kpi4.metric("Exit Equity ($M)", f"${lbo['exit_equity']/1e6:,.1f}M")

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown(f"""
**Entry Metrics**
| | Value |
|--|--|
| Enterprise Value | ${lbo['entry_ev']/1e6:,.1f}M |
| Equity | ${lbo['entry_equity']/1e6:,.1f}M |
| Debt | ${lbo['entry_debt']/1e6:,.1f}M |
| Entry Net Debt/EBITDA | {lbo['entry_debt_ebitda']:.1f}× |
        """)
    with col_out:
        st.markdown(f"""
**Exit Metrics (Year {lbo['holding_years']})**
| | Value |
|--|--|
| Enterprise Value | ${lbo['exit_ev']/1e6:,.1f}M |
| Equity | ${lbo['exit_equity']/1e6:,.1f}M |
| Remaining Debt | ${lbo['exit_debt']/1e6:,.1f}M |
| Exit Net Debt/EBITDA | {lbo.get('exit_debt_ebitda', 0):.1f}× |
| Exit EBITDA | ${lbo['exit_ebitda']/1e6:,.1f}M |
        """)

    # ── Sensitivity: MOIC vs entry multiple
    section_header("Sensitivity Analysis — MOIC by Entry Multiple")
    multiples = np.arange(5, 16, 0.5)
    moics = [
        compute_lbo_metrics(lbo_ebitda * 1e6, m, exit_multiple, debt_ratio,
                             holding_yrs, int_rate, ebitda_growth, lbo_tax, capex_pct)["moic"]
        for m in multiples
    ]
    import plotly.graph_objects as go
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=multiples, y=moics, mode="lines+markers",
                                   line=dict(color=COLORS["esg"], width=2.5),
                                   marker=dict(size=6)))
    fig_sens.add_hline(y=2.0, line_dash="dash", line_color=COLORS["classic"],
                        annotation_text="2.0× threshold")
    fig_sens.update_layout(
        title="MOIC Sensitivity to Entry EV/EBITDA Multiple",
        xaxis_title="Entry EV/EBITDA", yaxis_title="MOIC (×)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"]),
    )
    st.plotly_chart(fig_sens, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    section_header("🤖 Agent IA — Analyse Automatique")

    period_str = f"{start_str} to {end_str}"

    if not api_key_input:
        st.info(
            "💡 **Mode analyse locale activé** — aucune clé API requise. "
            "L'analyse est générée automatiquement à partir des métriques calculées. "
            "Entrez une clé Anthropic dans la sidebar pour une analyse enrichie par Claude."
        )
    else:
        st.success("🤖 **Mode Claude AI activé** — analyse enrichie par l'IA générative.")

    st.markdown(
        f"**Analyse :** {esg_name} (ESG) vs {classic_name} (Classique) | "
        f"**Période :** {period_str} | **Modèle :** {ff_model}"
    )

    if st.button("▶️ Générer l'analyse complète", type="primary"):
        with st.spinner("Génération de l'analyse en cours..."):
            try:
                analysis, analysis_source = run_ai_analysis(
                    esg_summary=esg_summary,
                    classic_summary=classic_summary,
                    esg_name=esg_name,
                    classic_name=classic_name,
                    ff_results=ff_results if ff_results else None,
                    period=period_str,
                    api_key=api_key_input,
                    model="claude-sonnet-4-6",
                )
                st.session_state.ai_analysis_text   = analysis
                st.session_state.ai_analysis_source = analysis_source
            except Exception as e:
                import traceback as tb
                st.error(f"❌ Erreur : {e}")
                st.code(tb.format_exc())

    # Afficher l'analyse persistée (survive au rerun)
    if st.session_state.ai_analysis_text:
        src_label = ("🤖 Claude AI" if st.session_state.ai_analysis_source == "claude_api"
                     else "🔢 Analyse locale automatique")
        st.caption(f"Source : {src_label}")
        st.markdown(st.session_state.ai_analysis_text)

    st.markdown("---")
    section_header("Analyses Rapides (nécessite une clé Claude AI)")

    if not api_key_input:
        st.info("💡 Entrez une clé API Anthropic dans la sidebar pour activer les analyses rapides.")
    else:
        qa_col1, qa_col2 = st.columns(2)

        with qa_col1:
            if st.button("📊 Résumé profil de risque"):
                prompt_risk = (
                    f"En 3 points concis, compare le profil de risque:\n"
                    f"- {esg_name}: Sharpe={fmt_num(esg_summary['Sharpe Ratio'])}, "
                    f"Vol={fmt_pct(esg_summary['Ann. Volatility (%)'])}, "
                    f"MaxDD={fmt_pct(esg_summary['Max Drawdown (%)'])}, Beta={fmt_num(esg_summary['Beta'])}\n"
                    f"- {classic_name}: Sharpe={fmt_num(classic_summary['Sharpe Ratio'])}, "
                    f"Vol={fmt_pct(classic_summary['Ann. Volatility (%)'])}, "
                    f"MaxDD={fmt_pct(classic_summary['Max Drawdown (%)'])}, Beta={fmt_num(classic_summary['Beta'])}\n"
                    f"Réponse en français, ton professionnel."
                )
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key_input)
                    resp = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=400,
                        messages=[{"role": "user", "content": prompt_risk}]
                    )
                    st.markdown(resp.content[0].text)
                except Exception as ex:
                    st.error(str(ex))

        with qa_col2:
            if st.button("🔬 Interprétation alpha factoriel"):
                if ff_results:
                    esg_ff = ff_results.get(esg_name, ff_results.get(list(ff_results.keys())[0], {}))
                    alpha = esg_ff.get("alpha_annual_pct", "N/A")
                    pval  = esg_ff.get("alpha_pvalue", "N/A")
                    prompt_alpha = (
                        f"En 2-3 phrases, interprète pour un investisseur institutionnel:\n"
                        f"Alpha {ff_model} de {esg_name} = {alpha}%/an, p-value = {pval}.\n"
                        f"Est-ce économiquement et statistiquement significatif ? Réponse en français."
                    )
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key_input)
                        resp = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=300,
                            messages=[{"role": "user", "content": prompt_alpha}]
                        )
                        st.markdown(resp.content[0].text)
                    except Exception as ex:
                        st.error(str(ex))
                else:
                    st.info("Lancez d'abord les modèles factoriels.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """<div style="text-align:center; color:#4A4D5E; font-size:0.8rem; padding:16px 0;">
    ESG Analytics Platform · Data: Yahoo Finance & Kenneth French Data Library ·
    Factor Models: CAPM, FF3, FF5, FF5+MOM · AI: Anthropic Claude
    </div>""",
    unsafe_allow_html=True
)
