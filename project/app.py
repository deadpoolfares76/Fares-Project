"""
ESG Analytics Platform — Institutional Edition v2.0
9 tabs: Executive | Risk | Factor | Stress | Monte Carlo | Regimes | Fundamentals | LBO | AI Memo
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, date

from utils.css_theme import (
    DARK_CSS, PLOTLY_LAYOUT,
    C_ESG, C_CLS, C_MKT, C_GOLD, C_PUR, C_ORG,
    C_BG, C_BDR, C_TXT, C_MUT,
)
from data.data_loader import (
    download_prices, get_returns, get_risk_free_rate,
    get_fama_french_factors, get_fundamentals,
    DEFAULT_ESG_TICKERS, DEFAULT_CLASSIC_TICKERS,
)
from analysis.risk_metrics import (
    full_risk_summary, compute_drawdown_series,
    rolling_volatility, rolling_beta, rolling_correlation, rolling_sharpe,
    detect_volatility_regimes, regime_analysis,
)
from analysis.factor_models import (
    run_capm, run_ff3, run_ff5, run_ff5_momentum,
    rolling_alpha, factor_decomposition,
)
from analysis.stress_testing import run_stress_tests, stress_table, CRISES
from analysis.monte_carlo import simulate_gbm, simulate_correlated
from analysis.fundamental import compute_wacc, compute_cost_equity, compute_lbo
from ai_agent.analyzer import run_ai_analysis, test_api_connection

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Analytics | Institutional",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "done": False, "last_cfg": None,
    "prices": None, "esg_ret": None, "cls_ret": None,
    "mkt_ret": None, "rf": None,
    "esg_sum": None, "cls_sum": None,
    "esg_dd": None, "cls_dd": None,
    "r_vol_e": None, "r_vol_c": None,
    "r_beta_e": None, "r_beta_c": None,
    "r_corr_e": None, "r_corr_c": None,
    "r_sh_e": None, "r_sh_c": None,
    "regime_mask": None, "reg_e": None, "reg_c": None,
    "ff_res": {}, "ff_avail": False, "ff_src": None, "ff_factors": None,
    "stress_e": None, "stress_c": None,
    "ai_txt": None, "ai_src": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def f(v, d=2, sfx="", na="—"):
    try:
        x = float(v)
        return na if np.isnan(x) else f"{x:.{d}f}{sfx}"
    except Exception:
        return na

def pl(fig, title="", h=380):
    fig.update_layout(**PLOTLY_LAYOUT, title_text=title, height=h)
    return fig

def line_trace(x, y, name, color, width=2, dash=None, fill=None):
    kw = {}
    if dash:
        kw["dash"] = dash
    if fill:
        kw["fill"] = fill
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=width, **kw),
        hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>%{{y:.2f}}<extra></extra>",
    )

def crisis_vrect(fig, start, end, color, label=""):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    fig.add_vrect(
        x0=s, x1=e, fillcolor=color, opacity=0.07,
        layer="below", line_width=0,
        annotation_text=label, annotation_position="top left",
        annotation_font=dict(size=8, color=color),
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#E8EAF0;padding:.3rem 0;'>◈ ESG Analytics</div>"
        "<div style='font-size:.62rem;color:#7B8098;letter-spacing:.1em;text-transform:uppercase;"
        "margin-bottom:1rem;'>Institutional Platform v2.0</div>",
        unsafe_allow_html=True,
    )

    def sb(label):
        st.markdown(
            f"<div style='font-size:.6rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;"
            f"color:#FFB800;padding:.35rem 0;border-bottom:1px solid #1E2440;margin:.6rem 0 .4rem 0;'>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )

    sb("Portfolio Configuration")
    esg_choice = st.selectbox("ESG Universe", list(DEFAULT_ESG_TICKERS.keys()), index=2)
    cls_choice = st.selectbox("Classic Benchmark", list(DEFAULT_CLASSIC_TICKERS.keys()), index=1)
    esg_ticker = DEFAULT_ESG_TICKERS[esg_choice]
    cls_ticker = DEFAULT_CLASSIC_TICKERS[cls_choice]
    esg_name   = esg_choice
    cls_name   = cls_choice

    if st.checkbox("Custom tickers"):
        esg_ticker = st.text_input("ESG Ticker",     value=esg_ticker).upper().strip()
        cls_ticker = st.text_input("Classic Ticker", value=cls_ticker).upper().strip()
        esg_name = esg_ticker
        cls_name = cls_ticker

    sb("Time Period")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("From", value=date(2018, 1, 1), min_value=date(2005, 1, 1))
    with c2:
        end_date = st.date_input("To", value=date.today())
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    sb("Model Parameters")
    ret_method_ui = st.selectbox("Returns",       ["Log", "Simple"])
    ff_model_ui   = st.selectbox("Factor Model",  ["CAPM", "FF3", "FF5", "FF5 + Momentum"], index=2)
    conf_var      = st.slider("VaR / CVaR Confidence", 0.90, 0.99, 0.95, 0.01, key="sb_conf")
    roll_win      = st.slider("Rolling Window (days)",  21,   252,   63,   21,  key="sb_roll")

    sb("AI Agent")
    api_key = st.text_input("Anthropic API Key (optional)", type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY", ""))
    if api_key:
        if st.button("Test API Connection", use_container_width=True):
            ok, msg = test_api_connection(api_key)
            (st.success if ok else st.error)(msg)

    st.markdown("---")
    run_btn = st.button("🚀  Run Analysis", type="primary", use_container_width=True)

# ── Data pipeline ─────────────────────────────────────────────────────────────
ret_method = "log" if ret_method_ui == "Log" else "simple"
ff_map     = {"CAPM": "FF3", "FF3": "FF3", "FF5": "FF5", "FF5 + Momentum": "FF5+MOM"}
cur_cfg    = (esg_ticker, cls_ticker, start_str, end_str, ret_method_ui, ff_model_ui)
cfg_chg    = (cur_cfg != st.session_state.last_cfg and st.session_state.done)

if not run_btn and not st.session_state.done:
    st.markdown(
        "<div style='text-align:center;padding:5rem 2rem;'>"
        "<div style='font-size:2.5rem;margin-bottom:.5rem;color:#00D4AA;'>◈</div>"
        "<div style='font-size:1.5rem;font-weight:700;color:#E8EAF0;margin-bottom:.4rem;'>"
        "ESG Analytics Platform</div>"
        "<div style='font-size:.85rem;color:#7B8098;margin-bottom:2rem;line-height:1.6;'>"
        "Institutional-grade ESG vs Classic portfolio analysis<br>"
        "Factor models · Stress testing · Monte Carlo · AI Investment Memo</div>"
        "<div style='font-size:.75rem;color:#4A4E65;'>Configure portfolios in the sidebar and click "
        "<span style='color:#00D4AA;font-weight:600;'>🚀 Run Analysis</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if run_btn or cfg_chg:
    st.session_state.last_cfg = cur_cfg
    st.session_state.ai_txt   = None
    st.session_state.ai_src   = None

    with st.spinner("Downloading market data…"):
        try:
            tickers = list({esg_ticker, cls_ticker, "SPY"})
            prices  = download_prices(tickers, start_str, end_str)
            if esg_ticker not in prices.columns or cls_ticker not in prices.columns:
                st.error(f"Price data unavailable for {esg_ticker} or {cls_ticker}.")
                st.stop()
            all_ret  = get_returns(prices, ret_method)
            esg_ret  = all_ret[esg_ticker].dropna()
            cls_ret  = all_ret[cls_ticker].dropna()
            mkt_ret  = all_ret["SPY"].dropna()
            common   = esg_ret.index.intersection(cls_ret.index).intersection(mkt_ret.index)
            esg_ret  = esg_ret.loc[common]
            cls_ret  = cls_ret.loc[common]
            mkt_ret  = mkt_ret.loc[common]
            rf       = get_risk_free_rate(start_str, end_str).reindex(common).ffill().fillna(0.04 / 252)
            st.session_state.prices  = prices
            st.session_state.esg_ret = esg_ret
            st.session_state.cls_ret = cls_ret
            st.session_state.mkt_ret = mkt_ret
            st.session_state.rf      = rf
        except Exception as err:
            import traceback as _tb
            st.error(f"Data error: {err}")
            st.code(_tb.format_exc())
            st.stop()

    with st.spinner("Loading Fama-French factors…"):
        try:
            ff_factors, ff_src = get_fama_french_factors(ff_map[ff_model_ui], start_str, end_str)
            ff_avail = True
        except Exception:
            ff_factors = None
            ff_src     = None
            ff_avail   = False
        st.session_state.ff_factors = ff_factors
        st.session_state.ff_src     = ff_src
        st.session_state.ff_avail   = ff_avail

    with st.spinner("Computing risk metrics…"):
        try:
            esg_sum = full_risk_summary(esg_name, esg_ret, mkt_ret, rf,
                                         benchmark=cls_ret, confidence=conf_var)
            cls_sum = full_risk_summary(cls_name, cls_ret, mkt_ret, rf,
                                         benchmark=esg_ret, confidence=conf_var)
        except Exception as err:
            import traceback as _tb
            st.error(f"Risk metrics error: {err}")
            st.code(_tb.format_exc())
            st.stop()

        esg_dd   = compute_drawdown_series(esg_ret)
        cls_dd   = compute_drawdown_series(cls_ret)
        r_vol_e  = rolling_volatility(esg_ret, roll_win)
        r_vol_c  = rolling_volatility(cls_ret,  roll_win)
        r_beta_e = rolling_beta(esg_ret, mkt_ret, roll_win)
        r_beta_c = rolling_beta(cls_ret,  mkt_ret, roll_win)
        r_corr_e = rolling_correlation(esg_ret, mkt_ret, roll_win)
        r_corr_c = rolling_correlation(cls_ret,  mkt_ret, roll_win)
        rw_sh    = min(roll_win, len(esg_ret) // 2)
        r_sh_e   = rolling_sharpe(esg_ret, rf, window=max(rw_sh, 20))
        r_sh_c   = rolling_sharpe(cls_ret,  rf, window=max(rw_sh, 20))
        reg_mask = detect_volatility_regimes(mkt_ret, window=21, threshold_pct=75)
        reg_e    = regime_analysis(esg_ret, mkt_ret, rf, reg_mask, esg_name)
        reg_c    = regime_analysis(cls_ret,  mkt_ret, rf, reg_mask, cls_name)

        for k, v in [
            ("esg_sum", esg_sum), ("cls_sum", cls_sum),
            ("esg_dd", esg_dd), ("cls_dd", cls_dd),
            ("r_vol_e", r_vol_e), ("r_vol_c", r_vol_c),
            ("r_beta_e", r_beta_e), ("r_beta_c", r_beta_c),
            ("r_corr_e", r_corr_e), ("r_corr_c", r_corr_c),
            ("r_sh_e", r_sh_e), ("r_sh_c", r_sh_c),
            ("regime_mask", reg_mask), ("reg_e", reg_e), ("reg_c", reg_c),
        ]:
            st.session_state[k] = v

    with st.spinner("Running factor regressions…"):
        ff_res = {}
        if ff_avail and ff_factors is not None:
            try:
                fn = {"CAPM": run_capm, "FF3": run_ff3, "FF5": run_ff5,
                      "FF5 + Momentum": run_ff5_momentum}[ff_model_ui]
                ff_res = {
                    esg_name: fn(esg_ret, ff_factors),
                    cls_name: fn(cls_ret,  ff_factors),
                }
            except Exception:
                pass
        st.session_state.ff_res = ff_res

    with st.spinner("Running stress tests…"):
        try:
            stress_e = run_stress_tests(esg_ret, mkt_ret)
            stress_c = run_stress_tests(cls_ret,  mkt_ret)
        except Exception:
            stress_e = {}
            stress_c = {}
        st.session_state.stress_e = stress_e
        st.session_state.stress_c = stress_c

    st.session_state.done = True

# ── Restore ───────────────────────────────────────────────────────────────────
prices    = st.session_state.prices
esg_ret   = st.session_state.esg_ret
cls_ret   = st.session_state.cls_ret
mkt_ret   = st.session_state.mkt_ret
rf        = st.session_state.rf
esg_sum   = st.session_state.esg_sum
cls_sum   = st.session_state.cls_sum
esg_dd    = st.session_state.esg_dd
cls_dd    = st.session_state.cls_dd
r_vol_e   = st.session_state.r_vol_e
r_vol_c   = st.session_state.r_vol_c
r_beta_e  = st.session_state.r_beta_e
r_beta_c  = st.session_state.r_beta_c
r_corr_e  = st.session_state.r_corr_e
r_corr_c  = st.session_state.r_corr_c
r_sh_e    = st.session_state.r_sh_e
r_sh_c    = st.session_state.r_sh_c
reg_mask  = st.session_state.regime_mask
reg_e     = st.session_state.reg_e
reg_c     = st.session_state.reg_c
ff_res    = st.session_state.ff_res
ff_avail  = st.session_state.ff_avail
ff_src    = st.session_state.ff_src
ff_factors= st.session_state.ff_factors
stress_e  = st.session_state.stress_e
stress_c  = st.session_state.stress_c

# ── Top header bar ────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns([3, 2, 2])
with h1:
    st.markdown(
        f"<div style='padding:.4rem 0'>"
        f"<div style='font-size:.62rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#7B8098;'>Portfolio Analysis</div>"
        f"<div style='font-size:1.15rem;font-weight:700;color:#E8EAF0;margin:.15rem 0;'>"
        f"<span style='color:{C_ESG};'>{esg_name}</span>"
        f"<span style='color:#7B8098;font-size:.85rem;font-weight:400;'> vs </span>"
        f"<span style='color:{C_CLS};'>{cls_name}</span></div>"
        f"<div style='font-size:.68rem;color:#7B8098;'>{start_str} → {end_str} "
        f"&nbsp;|&nbsp; {len(esg_ret)} trading days</div></div>",
        unsafe_allow_html=True,
    )
with h2:
    er = f(esg_sum.get("Ann. Return (%)"), 2, "%")
    cr = f(cls_sum.get("Ann. Return (%)"), 2, "%")
    st.markdown(
        f"<div style='text-align:center;padding:.4rem'>"
        f"<div style='font-size:.58rem;color:#7B8098;letter-spacing:.08em;text-transform:uppercase;'>Annualised Return</div>"
        f"<div style='font-size:1.05rem;font-weight:700;color:{C_ESG};'>{er}</div>"
        f"<div style='font-size:.78rem;color:{C_CLS};'>{cr}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with h3:
    now = datetime.now().strftime("%d %b %Y  %H:%M")
    src_lbl = {"datareader": "FF Official", "direct_http": "FF Direct",
               "etf_proxy": "ETF Proxy"}.get(ff_src, "—")
    st.markdown(
        f"<div style='text-align:right;padding:.4rem'>"
        f"<div style='font-size:.62rem;color:#7B8098;'>{now}</div>"
        f"<div style='font-size:.62rem;color:#7B8098;margin-top:.15rem;'>"
        f"Factor src: <span style='color:{C_GOLD};'>{src_lbl}</span></div>"
        f"<div style='font-size:.62rem;color:#7B8098;'>"
        f"Model: <span style='color:{C_MKT};'>{ff_model_ui}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
st.markdown(
    "<hr style='border:none;border-top:1px solid #1E2440;margin:.4rem 0 .8rem 0;'>",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "01 EXECUTIVE", "02 RISK & PERF", "03 FACTOR MODELS",
    "04 STRESS TEST", "05 MONTE CARLO", "06 REGIMES",
    "07 FUNDAMENTALS", "08 LBO", "09 AI MEMO",
])

# ══════════════════════════════════════════════════════════════════════════════
# 01 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    sec("Key Performance Indicators")

    def delta_str(e_val, c_val, higher_better=True, sfx=""):
        try:
            ev, cv = float(e_val), float(c_val)
            d = ev - cv
            arrow = "↑" if (d > 0) == higher_better else "↓"
            return f"{arrow} {abs(d):.2f}{sfx} vs Classic"
        except Exception:
            return None

    kpi_rows = [
        [
            ("Sharpe Ratio",       esg_sum.get("Sharpe Ratio"),        cls_sum.get("Sharpe Ratio"),        True,  ""),
            ("Sortino Ratio",      esg_sum.get("Sortino Ratio"),       cls_sum.get("Sortino Ratio"),       True,  ""),
            ("Calmar Ratio",       esg_sum.get("Calmar Ratio"),        cls_sum.get("Calmar Ratio"),        True,  ""),
            ("Omega Ratio",        esg_sum.get("Omega Ratio"),         cls_sum.get("Omega Ratio"),         True,  ""),
        ],
        [
            ("Ann. Return",        esg_sum.get("Ann. Return (%)"),     cls_sum.get("Ann. Return (%)"),     True,  "%"),
            ("Ann. Volatility",    esg_sum.get("Ann. Volatility (%)"), cls_sum.get("Ann. Volatility (%)"), False, "%"),
            ("Max Drawdown",       esg_sum.get("Max Drawdown (%)"),    cls_sum.get("Max Drawdown (%)"),    False, "%"),
            ("CAPM Alpha",         esg_sum.get("CAPM Alpha (%)"),      cls_sum.get("CAPM Alpha (%)"),      True,  "%"),
        ],
        [
            ("Beta",               esg_sum.get("Beta"),                cls_sum.get("Beta"),                False, ""),
            ("CVaR 95%",           esg_sum.get("CVaR (%)"),            cls_sum.get("CVaR (%)"),            False, "%"),
            ("Tracking Error",     esg_sum.get("Tracking Error (%)"),  cls_sum.get("Tracking Error (%)"),  False, "%"),
            ("Information Ratio",  esg_sum.get("Information Ratio"),   cls_sum.get("Information Ratio"),   True,  ""),
        ],
    ]
    for row in kpi_rows:
        cols = st.columns(4)
        for col, (label, ev, cv, hb, sfx) in zip(cols, row):
            try:
                disp = f"{float(ev):.2f}{sfx}" if not np.isnan(float(ev)) else "—"
            except Exception:
                disp = "—"
            col.metric(f"ESG — {label}", disp, delta_str(ev, cv, hb, sfx))

    st.markdown("---")
    sec("Cumulative Wealth Index — Rebased 100")

    esg_w = (1 + esg_ret).cumprod() * 100
    cls_w = (1 + cls_ret).cumprod() * 100
    mkt_w = (1 + mkt_ret).cumprod() * 100

    fig_w = go.Figure()
    for crisis_key, c_info in CRISES.items():
        cs, ce = pd.to_datetime(c_info["start"]), pd.to_datetime(c_info["end"])
        if cs >= esg_w.index[0] and ce <= esg_w.index[-1]:
            crisis_vrect(fig_w, cs, ce, c_info["color"], crisis_key.split(" ")[0])

    fig_w.add_trace(go.Scatter(
        x=esg_w.index, y=esg_w.values, name=esg_name,
        line=dict(color=C_ESG, width=2.2),
        hovertemplate="<b>" + esg_name + "</b><br>%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
    ))
    fig_w.add_trace(go.Scatter(
        x=cls_w.index, y=cls_w.values, name=cls_name,
        line=dict(color=C_CLS, width=2.2),
        hovertemplate="<b>" + cls_name + "</b><br>%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
    ))
    fig_w.add_trace(go.Scatter(
        x=mkt_w.index, y=mkt_w.values, name="S&P 500",
        line=dict(color=C_MKT, width=1.5, dash="dot"),
        hovertemplate="<b>S&P 500</b><br>%{x|%d %b %Y}<br>%{y:.1f}<extra></extra>",
    ))

    # Max drawdown annotations
    for series, name_ann, color in [(esg_dd, esg_name, C_ESG), (cls_dd, cls_name, C_CLS)]:
        if len(series) > 0:
            trough_idx = series.idxmin()
            w = esg_w if color == C_ESG else cls_w
            trough_y = float(w.loc[trough_idx]) if trough_idx in w.index else None
            if trough_y:
                fig_w.add_annotation(
                    x=trough_idx, y=trough_y,
                    text=f"MaxDD<br>{f(series.min(), 1)}%",
                    showarrow=True, arrowhead=2, arrowcolor=color,
                    font=dict(color=color, size=9), bgcolor="#12151F",
                    bordercolor=color, borderwidth=1, borderpad=3,
                )
    pl(fig_w, "", h=420)
    st.plotly_chart(fig_w, use_container_width=True)

    # Drawdown comparison
    sec("Drawdown Analysis")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=esg_dd.index, y=esg_dd.values, name=esg_name,
        fill="tozeroy", fillcolor=f"rgba(0,212,170,0.08)",
        line=dict(color=C_ESG, width=1.5),
    ))
    fig_dd.add_trace(go.Scatter(
        x=cls_dd.index, y=cls_dd.values, name=cls_name,
        fill="tozeroy", fillcolor=f"rgba(255,69,96,0.08)",
        line=dict(color=C_CLS, width=1.5),
    ))
    pl(fig_dd, "Drawdown (%)", h=260)
    st.plotly_chart(fig_dd, use_container_width=True)

    sec("Summary Comparison Table")
    compare_keys = [
        ("Ann. Return (%)", "Ann. Return", True),
        ("Ann. Volatility (%)", "Ann. Volatility", False),
        ("Sharpe Ratio", "Sharpe", True),
        ("Sortino Ratio", "Sortino", True),
        ("Calmar Ratio", "Calmar", True),
        ("Omega Ratio", "Omega", True),
        ("Information Ratio", "Info. Ratio", True),
        ("Tracking Error (%)", "Tracking Error", False),
        ("Beta", "Beta", False),
        ("CAPM Alpha (%)", "CAPM Alpha", True),
        ("Max Drawdown (%)", "Max Drawdown", False),
        ("CVaR (%)", "CVaR 95%", False),
        ("Skewness", "Skewness", True),
        ("Excess Kurtosis", "Kurtosis", False),
    ]
    rows = []
    for key, label, hb in compare_keys:
        ev = esg_sum.get(key, np.nan)
        cv = cls_sum.get(key, np.nan)
        try:
            evf, cvf = float(ev), float(cv)
            winner = (esg_name if ((hb and evf > cvf) or (not hb and evf < cvf)) else cls_name) if not (np.isnan(evf) or np.isnan(cvf)) else "—"
        except Exception:
            winner = "—"
        rows.append({
            "Metric": label,
            esg_name: f(ev, 3),
            cls_name: f(cv, 3),
            "Edge": "✅ ESG" if winner == esg_name else ("📊 Classic" if winner == cls_name else "—"),
        })
    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 02 — RISK & PERFORMANCE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    sec("Rolling Risk Metrics")

    tab_a, tab_b, tab_c = st.tabs(["Rolling Sharpe", "Rolling Beta", "Rolling Volatility"])

    with tab_a:
        fig_sh = go.Figure()
        fig_sh.add_hline(y=1.0, line_dash="dash", line_color=C_GOLD,
                         annotation_text="Sharpe=1", annotation_font_size=9)
        fig_sh.add_hline(y=0.0, line_dash="dot",  line_color=C_MUT, line_width=1)
        if r_sh_e is not None:
            fig_sh.add_trace(go.Scatter(
                x=r_sh_e.index, y=r_sh_e.values, name=esg_name,
                line=dict(color=C_ESG, width=1.8),
                hovertemplate="<b>" + esg_name + "</b><br>%{x|%d %b %Y}<br>Sharpe: %{y:.3f}<extra></extra>",
            ))
        if r_sh_c is not None:
            fig_sh.add_trace(go.Scatter(
                x=r_sh_c.index, y=r_sh_c.values, name=cls_name,
                line=dict(color=C_CLS, width=1.8),
                hovertemplate="<b>" + cls_name + "</b><br>%{x|%d %b %Y}<br>Sharpe: %{y:.3f}<extra></extra>",
            ))
        pl(fig_sh, f"Rolling Sharpe Ratio ({roll_win}d window)", h=340)
        st.plotly_chart(fig_sh, use_container_width=True)
        st.caption(f"Rolling Sharpe over {roll_win}-day window. Dotted gold line = Sharpe 1.0 threshold.")

    with tab_b:
        fig_bt = go.Figure()
        fig_bt.add_hline(y=1.0, line_dash="dash", line_color=C_GOLD,
                         annotation_text="β=1", annotation_font_size=9)
        if r_beta_e is not None:
            fig_bt.add_trace(go.Scatter(
                x=r_beta_e.index, y=r_beta_e.values, name=esg_name,
                line=dict(color=C_ESG, width=1.8),
                hovertemplate="<b>" + esg_name + "</b><br>%{x|%d %b %Y}<br>Beta: %{y:.3f}<extra></extra>",
            ))
        if r_beta_c is not None:
            fig_bt.add_trace(go.Scatter(
                x=r_beta_c.index, y=r_beta_c.values, name=cls_name,
                line=dict(color=C_CLS, width=1.8),
                hovertemplate="<b>" + cls_name + "</b><br>%{x|%d %b %Y}<br>Beta: %{y:.3f}<extra></extra>",
            ))
        pl(fig_bt, f"Rolling Beta vs S&P 500 ({roll_win}d)", h=340)
        st.plotly_chart(fig_bt, use_container_width=True)

    with tab_c:
        fig_vl = go.Figure()
        if r_vol_e is not None:
            fig_vl.add_trace(go.Scatter(
                x=r_vol_e.index, y=r_vol_e.values, name=esg_name,
                line=dict(color=C_ESG, width=1.8),
                hovertemplate="<b>" + esg_name + "</b><br>%{x|%d %b %Y}<br>Vol: %{y:.2f}%<extra></extra>",
            ))
        if r_vol_c is not None:
            fig_vl.add_trace(go.Scatter(
                x=r_vol_c.index, y=r_vol_c.values, name=cls_name,
                line=dict(color=C_CLS, width=1.8),
                hovertemplate="<b>" + cls_name + "</b><br>%{x|%d %b %Y}<br>Vol: %{y:.2f}%<extra></extra>",
            ))
        pl(fig_vl, f"Rolling Annualised Volatility — % ({roll_win}d)", h=340)
        st.plotly_chart(fig_vl, use_container_width=True)

    sec("Return Distribution")
    c1, c2 = st.columns(2)

    def dist_fig(ret, name, color):
        r = ret.dropna().values * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=r, nbinsx=60, name=name,
            marker=dict(color=color, opacity=0.7),
            histnorm="probability density",
        ))
        import scipy.stats as spst
        mu, sg = r.mean(), r.std()
        xs = np.linspace(r.min(), r.max(), 300)
        ys = spst.norm.pdf(xs, mu, sg)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name="Normal fit",
            line=dict(color=C_GOLD, width=1.5, dash="dash"),
        ))
        var95 = float(np.percentile(r, 5))
        fig.add_vline(x=var95, line_dash="dot", line_color=C_CLS,
                      annotation_text=f"VaR95: {var95:.2f}%",
                      annotation_font=dict(color=C_CLS, size=9))
        pl(fig, f"{name} — Return Distribution", h=300)
        return fig

    with c1:
        st.plotly_chart(dist_fig(esg_ret, esg_name, C_ESG), use_container_width=True)
    with c2:
        st.plotly_chart(dist_fig(cls_ret, cls_name, C_CLS), use_container_width=True)

    sec("Advanced Risk Metrics")
    adv_keys = [
        ("Sharpe Ratio",       "Sharpe"),
        ("Sortino Ratio",      "Sortino"),
        ("Calmar Ratio",       "Calmar"),
        ("Omega Ratio",        "Omega"),
        ("Information Ratio",  "Info. Ratio"),
        ("Tracking Error (%)", "Tracking Error (%)"),
        ("VaR Hist (%)",       "VaR Hist 95%"),
        ("VaR Param (%)",      "VaR Param 95%"),
        ("CVaR (%)",           "CVaR 95%"),
        ("Skewness",           "Skewness"),
        ("Excess Kurtosis",    "Kurtosis"),
    ]
    adv_rows = [{"Metric": lbl, esg_name: f(esg_sum.get(k), 3), cls_name: f(cls_sum.get(k), 3)}
                for k, lbl in adv_keys]
    st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 03 — FACTOR MODELS
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    if not ff_avail or not ff_res:
        st.warning(
            "⚠️ **Factor data unavailable.** All 3 download attempts failed (Dartmouth server + HTTP + ETF proxy). "
            "This is usually a **network/firewall issue** on your machine. "
            "Try: (1) check your internet connection, (2) disable VPN/proxy, "
            "(3) click **🚀 Run Analysis** again — the ETF proxy fallback should eventually succeed."
        )
    else:
        for ptf_name, res in ff_res.items():
            color = C_ESG if ptf_name == esg_name else C_CLS
            sec(f"Factor Regression — {ptf_name}  ({ff_model_ui})")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Alpha (ann.)",
                      f"{f(res.get('alpha_annual_pct'), 2)}%",
                      "Significant ✅" if res.get("alpha_significant") else "Not sig.")
            m2.metric("t-stat α",   f(res.get("alpha_tstat"), 3))
            m3.metric("p-value α",  f(res.get("alpha_pvalue"), 4))
            m4.metric("R²",         f(res.get("r_squared"),   3))

            factors = res.get("factors", {})
            if factors:
                fnames   = list(factors.keys())
                loadings = [factors[fn]["loading"]  for fn in fnames]
                tstats   = [factors[fn]["t_stat"]   for fn in fnames]
                ci_lo    = [factors[fn]["ci_low"]   for fn in fnames]
                ci_hi    = [factors[fn]["ci_high"]  for fn in fnames]
                sigs     = [factors[fn]["significant"] for fn in fnames]
                bar_cols = [C_ESG if s else C_MUT for s in sigs]

                fig_f = go.Figure()
                fig_f.add_trace(go.Bar(
                    x=fnames, y=loadings, name="Factor Loading",
                    marker_color=bar_cols, opacity=0.85,
                    error_y=dict(
                        type="data", symmetric=False,
                        array=[hi - lo for lo, hi in zip(ci_lo, ci_hi)],
                        arrayminus=[lo - lo for lo in ci_lo],
                        color=C_MUT, thickness=1.5, width=4,
                    ),
                    hovertemplate="<b>%{x}</b><br>β=%{y:.4f}<extra></extra>",
                ))
                fig_f.add_hline(y=0, line_color=C_MUT, line_width=1)
                pl(fig_f, "Factor Loadings (β) with 95% HAC Confidence Intervals", h=300)
                st.plotly_chart(fig_f, use_container_width=True)

                frows = [{"Factor": fn,
                          "Loading (β)": f(factors[fn]["loading"], 4),
                          "t-stat": f(factors[fn]["t_stat"], 3),
                          "p-value": f(factors[fn]["p_value"], 4),
                          "CI 95% low": f(factors[fn]["ci_low"], 4),
                          "CI 95% high": f(factors[fn]["ci_high"], 4),
                          "Significant": "✅" if factors[fn]["significant"] else "—"}
                         for fn in fnames]
                st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)

                if ff_factors is not None:
                    dec = factor_decomposition(res, ff_factors)
                    if not dec.empty:
                        with st.expander("Alpha Decomposition — Factor Contributions"):
                            st.dataframe(dec, use_container_width=True, hide_index=True)

        # Rolling alpha
        if ff_factors is not None:
            sec("Rolling Alpha — Persistence Analysis")
            roll_model = "FF5" if "FF5" in ff_model_ui else ("FF3" if "FF3" in ff_model_ui else "CAPM")
            fig_ra = go.Figure()
            fig_ra.add_hline(y=0, line_color=C_MUT, line_width=1, line_dash="dot")
            for ptf_ret, ptf_name, color in [(esg_ret, esg_name, C_ESG), (cls_ret, cls_name, C_CLS)]:
                try:
                    ra = rolling_alpha(ptf_ret, ff_factors, model=roll_model, window=min(roll_win, len(ptf_ret)//2))
                    fig_ra.add_trace(go.Scatter(
                        x=ra.index, y=ra.values, name=ptf_name,
                        line=dict(color=color, width=1.8),
                        hovertemplate="<b>" + ptf_name + "</b><br>%{x|%d %b %Y}<br>Alpha: %{y:.2f}%<extra></extra>",
                    ))
                except Exception:
                    pass
            pl(fig_ra, f"Rolling Alpha %/year ({roll_model}, {roll_win}d window)", h=320)
            st.plotly_chart(fig_ra, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 04 — STRESS TESTING
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    sec("Historical Crisis Scenarios")

    if not stress_e or not stress_c:
        st.info("No stress test data available. Rerun the analysis.")
    else:
        df_st = stress_table(stress_e, stress_c)
        if not df_st.empty:
            # Summary heatmap
            ret_cols = ["ESG (%)", "Classic (%)"]
            heat_data = []
            crisis_labels = []
            for _, row in df_st.iterrows():
                try:
                    heat_data.append([float(row["ESG (%)"]), float(row["Classic (%)"])])
                    crisis_labels.append(str(row["Scénario"]))
                except Exception:
                    pass

            if heat_data:
                heat_arr = np.array(heat_data)
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heat_arr,
                    x=[esg_name, cls_name],
                    y=crisis_labels,
                    colorscale=[[0, C_CLS], [0.5, "#1E2440"], [1, C_ESG]],
                    text=[[f"{v:.1f}%" for v in row] for row in heat_data],
                    texttemplate="%{text}",
                    textfont=dict(size=11, family="JetBrains Mono, monospace"),
                    zmid=0,
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
                ))
                pl(fig_heat, "Crisis Returns Heatmap (%)", h=300)
                st.plotly_chart(fig_heat, use_container_width=True)

        sec("Crisis-by-Crisis Detail")
        for crisis_key, c_info in CRISES.items():
            e = stress_e.get(crisis_key, {})
            c = stress_c.get(crisis_key, {})
            if not e.get("available") and not c.get("available"):
                continue

            with st.expander(
                f"{e.get('label', crisis_key)}  ·  {c_info['start']} → {c_info['end']}  "
                f"·  {e.get('desc', '')}",
                expanded=False,
            ):
                cols = st.columns(4)
                cols[0].metric(
                    f"ESG Return", f"{f(e.get('total_ret_pct'), 2)}%",
                    delta=f"{f(float(e.get('total_ret_pct', np.nan) or np.nan) - float(c.get('total_ret_pct', np.nan) or np.nan), 2)}% vs Classic"
                    if e.get("available") and c.get("available") else None,
                )
                cols[1].metric("Classic Return",  f"{f(c.get('total_ret_pct'), 2)}%")
                cols[2].metric("ESG MaxDD",        f"{f(e.get('max_dd_pct'), 2)}%")
                cols[3].metric("Classic MaxDD",    f"{f(c.get('max_dd_pct'), 2)}%")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric("ESG Beta",               f(e.get("beta"), 3))
                r2.metric("Classic Beta",            f(c.get("beta"), 3))
                r3.metric("ESG Recovery (days)",     str(e.get("recovery_days") or "Not recovered"))
                r4.metric("Classic Recovery (days)", str(c.get("recovery_days") or "Not recovered"))

                # Crisis window chart
                cs = pd.to_datetime(c_info["start"]) - pd.Timedelta(days=30)
                ce = pd.to_datetime(c_info["end"])   + pd.Timedelta(days=90)
                esg_sub = esg_ret.loc[(esg_ret.index >= cs) & (esg_ret.index <= ce)]
                cls_sub = cls_ret.loc[(cls_ret.index >= cs) & (cls_ret.index <= ce)]
                if len(esg_sub) > 0 and len(cls_sub) > 0:
                    ew = (1 + esg_sub).cumprod() * 100
                    cw = (1 + cls_sub).cumprod() * 100
                    fig_cr = go.Figure()
                    crisis_vrect(fig_cr, c_info["start"], c_info["end"], c_info["color"])
                    fig_cr.add_trace(go.Scatter(x=ew.index, y=ew.values, name=esg_name,
                                                line=dict(color=C_ESG, width=2)))
                    fig_cr.add_trace(go.Scatter(x=cw.index, y=cw.values, name=cls_name,
                                                line=dict(color=C_CLS, width=2)))
                    pl(fig_cr, "", h=240)
                    st.plotly_chart(fig_cr, use_container_width=True)

        sec("All Crises — Comparison Table")
        if not df_st.empty:
            st.dataframe(df_st, use_container_width=True, hide_index=True)

        # Custom scenario
        sec("Custom Scenario")
        cc1, cc2 = st.columns(2)
        with cc1:
            cust_start = st.date_input("Start", value=date(2020, 1, 1), key="cust_s")
        with cc2:
            cust_end   = st.date_input("End",   value=date(2020, 6, 1), key="cust_e")
        if st.button("Run Custom Scenario"):
            try:
                ce = run_stress_tests(
                    esg_ret, mkt_ret,
                    extra_scenarios={"Custom": {
                        "start": cust_start.strftime("%Y-%m-%d"),
                        "end":   cust_end.strftime("%Y-%m-%d"),
                        "label": "Custom Scenario", "desc": "", "color": C_GOLD,
                    }},
                )
                cc = run_stress_tests(
                    cls_ret, mkt_ret,
                    extra_scenarios={"Custom": {
                        "start": cust_start.strftime("%Y-%m-%d"),
                        "end":   cust_end.strftime("%Y-%m-%d"),
                        "label": "Custom Scenario", "desc": "", "color": C_GOLD,
                    }},
                )
                e_r = ce.get("Custom", {})
                c_r = cc.get("Custom", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ESG Return",    f"{f(e_r.get('total_ret_pct'), 2)}%")
                c2.metric("Classic Return", f"{f(c_r.get('total_ret_pct'), 2)}%")
                c3.metric("ESG MaxDD",      f"{f(e_r.get('max_dd_pct'), 2)}%")
                c4.metric("Classic MaxDD",  f"{f(c_r.get('max_dd_pct'), 2)}%")
            except Exception as ex:
                st.error(str(ex))

# ══════════════════════════════════════════════════════════════════════════════
# 05 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    sec("Monte Carlo GBM Simulation")
    st.caption(
        "S(t) = S(0) · exp[(μ − σ²/2)·t + σ·W(t)]  —  "
        "Parameters calibrated on historical daily returns."
    )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        n_sims   = st.select_slider("Simulations", [500, 1000, 2000, 5000], value=1000)
    with mc2:
        horizon  = st.select_slider("Horizon (years)", [1, 2, 3, 5, 10], value=5)
    with mc3:
        corr_mode = st.checkbox("Correlated paths (Cholesky)", value=True)

    if st.button("▶ Run Monte Carlo", type="primary"):
        with st.spinner(f"Simulating {n_sims:,} paths × {horizon}y…"):
            try:
                if corr_mode:
                    mc_res = simulate_correlated(esg_ret, cls_ret, n_sims=n_sims,
                                                  horizon_years=float(horizon), seed=42)
                    mc_e = mc_res["esg"]
                    mc_c = mc_res["classic"]
                    corr  = mc_res.get("correlation", np.nan)
                    t_ax  = mc_res["time_axis"]
                else:
                    mc_e = simulate_gbm(esg_ret, n_sims=n_sims, horizon_years=float(horizon), seed=42)
                    mc_c = simulate_gbm(cls_ret,  n_sims=n_sims, horizon_years=float(horizon), seed=43)
                    corr  = np.nan
                    t_ax  = mc_e["time_axis"]

                # Metrics strip
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("ESG Median Wealth",    f"{f(mc_e['metrics']['median'], 1)}")
                m2.metric("Classic Median",        f"{f(mc_c['metrics']['median'], 1)}")
                m3.metric("ESG Prob. Loss",        f"{f(mc_e['metrics']['prob_loss_pct'], 1)}%")
                m4.metric("Classic Prob. Loss",    f"{f(mc_c['metrics']['prob_loss_pct'], 1)}%")
                m5.metric("ESG VaR 5th pctile",   f"{f(mc_e['metrics']['var5'], 1)}")
                m6.metric("Correlation (ρ)",       f(corr, 3))

                # Fan chart
                fig_mc = go.Figure()
                years = t_ax / 252
                pct_labels = {5: ("5th", 0.35), 25: ("25th", 0.55), 50: ("Median", 1.0),
                              75: ("75th", 0.55), 95: ("95th", 0.35)}
                # ESG fan
                for pctile, (lbl, op) in pct_labels.items():
                    y_vals = mc_e["percentiles"][pctile]
                    show = pctile == 50
                    fig_mc.add_trace(go.Scatter(
                        x=years, y=y_vals,
                        name=f"ESG {lbl}" if show else f"ESG {lbl}",
                        showlegend=show,
                        line=dict(color=C_ESG, width=2.5 if pctile == 50 else 0.8,
                                  dash="solid" if pctile == 50 else "dot"),
                        opacity=op,
                        hovertemplate=f"ESG {lbl}: %{{y:.1f}}<extra></extra>",
                    ))
                # Classic fan
                for pctile, (lbl, op) in pct_labels.items():
                    y_vals = mc_c["percentiles"][pctile]
                    show = pctile == 50
                    fig_mc.add_trace(go.Scatter(
                        x=years, y=y_vals,
                        name=f"Classic {lbl}" if show else f"Classic {lbl}",
                        showlegend=show,
                        line=dict(color=C_CLS, width=2.5 if pctile == 50 else 0.8,
                                  dash="solid" if pctile == 50 else "dot"),
                        opacity=op,
                        hovertemplate=f"Classic {lbl}: %{{y:.1f}}<extra></extra>",
                    ))
                fig_mc.add_hline(y=100, line_dash="dash", line_color=C_MUT,
                                 annotation_text="Initial capital (100)", annotation_font_size=9)
                pl(fig_mc, f"Monte Carlo — {n_sims:,} paths, {horizon}y horizon (GBM)", h=440)
                fig_mc.update_layout(xaxis_title="Years", yaxis_title="Portfolio Value (rebased 100)")
                st.plotly_chart(fig_mc, use_container_width=True)

                # Terminal distribution
                sec("Terminal Wealth Distribution")
                td1, td2 = st.columns(2)
                for col, mc_data, name, color in [
                    (td1, mc_e, esg_name, C_ESG),
                    (td2, mc_c, cls_name, C_CLS),
                ]:
                    with col:
                        term = mc_data["terminal"]
                        fig_td = go.Figure()
                        fig_td.add_trace(go.Histogram(
                            x=term, nbinsx=60, name=name,
                            marker_color=color, opacity=0.75,
                            hovertemplate="%{x:.1f}: %{y}<extra></extra>",
                        ))
                        fig_td.add_vline(x=100, line_dash="dash", line_color=C_MUT,
                                         annotation_text="Capital=100")
                        fig_td.add_vline(x=float(np.median(term)), line_dash="dash",
                                         line_color=color,
                                         annotation_text=f"Median={f(np.median(term), 0)}",
                                         annotation_font=dict(color=color))
                        pl(fig_td, f"{name} — Terminal Wealth", h=280)
                        col.plotly_chart(fig_td, use_container_width=True)

                        col.markdown(
                            f"**Annualised drift:** {f(mc_data['metrics']['ann_ret_pct'], 2)}%  "
                            f"| **Ann. vol:** {f(mc_data['metrics']['ann_vol_pct'], 2)}%  "
                            f"| **P(loss):** {f(mc_data['metrics']['prob_loss_pct'], 1)}%  "
                            f"| **P(double):** {f(mc_data['metrics']['prob_double_pct'], 1)}%"
                        )

            except Exception as ex:
                import traceback as _tb
                st.error(f"Monte Carlo error: {ex}")
                st.code(_tb.format_exc())
    else:
        st.info("Configure parameters above and click **▶ Run Monte Carlo** to launch the simulation.")

# ══════════════════════════════════════════════════════════════════════════════
# 06 — VOLATILITY REGIMES
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    sec("Volatility Regime Detection")
    st.caption("High-volatility regime: rolling 21-day annualised vol above the 75th percentile.")

    # Regime overlay on cumulative wealth
    esg_w = (1 + esg_ret).cumprod() * 100
    cls_w = (1 + cls_ret).cumprod() * 100
    mkt_rv = mkt_ret.rolling(21).std() * np.sqrt(252) * 100

    fig_reg = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
    )
    fig_reg.add_trace(go.Scatter(x=esg_w.index, y=esg_w.values, name=esg_name,
                                  line=dict(color=C_ESG, width=1.8)), row=1, col=1)
    fig_reg.add_trace(go.Scatter(x=cls_w.index, y=cls_w.values, name=cls_name,
                                  line=dict(color=C_CLS, width=1.8)), row=1, col=1)
    fig_reg.add_trace(go.Scatter(x=mkt_rv.index, y=mkt_rv.values, name="Market Vol",
                                  fill="tozeroy", fillcolor="rgba(255,140,0,0.08)",
                                  line=dict(color=C_ORG, width=1.2)), row=2, col=1)
    if reg_mask is not None:
        vol_thresh = float(mkt_ret.rolling(21).std().dropna().quantile(0.75) * np.sqrt(252) * 100)
        fig_reg.add_hline(y=vol_thresh, line_dash="dash", line_color=C_ORG,
                           annotation_text=f"Regime threshold ({vol_thresh:.1f}%)",
                           annotation_font=dict(color=C_ORG, size=9), row=2, col=1)
    fig_reg.update_layout(**PLOTLY_LAYOUT, height=420, title_text="Cumulative Wealth with Volatility Regimes")
    fig_reg.update_yaxes(title_text="Index (100)", row=1, col=1)
    fig_reg.update_yaxes(title_text="Vol %", row=2, col=1)
    st.plotly_chart(fig_reg, use_container_width=True)

    sec("Performance by Regime")
    all_reg = [("High Volatility", C_ORG), ("Low Volatility", C_MKT)]
    for label, color in all_reg:
        e_r = reg_e.get(label, {})
        c_r = reg_c.get(label, {})
        if not e_r and not c_r:
            continue
        st.markdown(
            f"<div style='font-size:.7rem;font-weight:600;color:{color};"
            f"text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem 0;'>"
            f"{'🌩 ' if 'High' in label else '📈 '}{label} — {e_r.get('n_days', 0)} trading days</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(6)
        fields = [
            ("Ann. Return (%)",  "ann_return_pct", True),
            ("Ann. Vol (%)",     "ann_vol_pct",    False),
            ("Sharpe",           "sharpe",         True),
            ("Max DD (%)",       "max_dd_pct",     False),
            ("Beta",             "beta",           False),
        ]
        for col, (lbl, key, hb) in zip(cols, fields):
            ev = e_r.get(key, np.nan)
            col.metric(
                f"ESG {lbl}",
                f(ev, 3),
                delta=f"{f(float(ev or np.nan) - float(c_r.get(key, np.nan) or np.nan), 3)} vs Classic"
                if not np.isnan(float(ev or np.nan)) and not np.isnan(float(c_r.get(key, np.nan) or np.nan))
                else None,
            )

# ══════════════════════════════════════════════════════════════════════════════
# 07 — FUNDAMENTALS & WACC
# ══════════════════════════════════════════════════════════════════════════════
with t7:
    sec("Fundamental Data")
    f1, f2 = st.columns(2)
    for col, ticker, name in [(f1, esg_ticker, esg_name), (f2, cls_ticker, cls_name)]:
        with col:
            st.markdown(
                f"<div style='font-size:.7rem;font-weight:600;color:#7B8098;"
                f"text-transform:uppercase;margin-bottom:.4rem;'>{name} ({ticker})</div>",
                unsafe_allow_html=True,
            )
            with st.spinner(f"Fetching {ticker}…"):
                try:
                    fd = get_fundamentals(ticker)
                    if "error" in fd:
                        st.warning(f"Could not fetch: {fd['error']}")
                    else:
                        st.markdown(f"**{fd.get('name', ticker)}** — {fd.get('sector', '—')}")
                        r1, r2 = st.columns(2)
                        mc = fd.get("market_cap")
                        ev = fd.get("enterprise_value")
                        r1.metric("Market Cap", f"${mc/1e9:.1f}B" if mc else "—")
                        r2.metric("EV",          f"${ev/1e9:.1f}B" if ev else "—")
                        r3, r4 = st.columns(2)
                        r3.metric("EV/EBITDA",   f(fd.get("ev_ebitda"), 2, "x"))
                        r4.metric("Trailing P/E", f(fd.get("trailing_pe"), 2, "x"))
                        r5, r6 = st.columns(2)
                        r5.metric("ROE",  f(float(fd.get("roe") or np.nan)*100, 2, "%"))
                        r6.metric("Beta", f(fd.get("beta"), 3))
                except Exception as ex:
                    st.warning(f"Fundamental data error: {ex}")

    sec("WACC Calculator")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        wacc_rf   = st.slider("Risk-Free Rate (%)",  0.0, 8.0, 4.2, 0.1, key="wacc_rf") / 100
        wacc_erp  = st.slider("Equity Risk Premium (%)", 2.0, 10.0, 5.5, 0.1, key="wacc_erp") / 100
    with wc2:
        wacc_beta = st.slider("Beta",         0.3, 2.5, 1.0, 0.05, key="wacc_beta")
        wacc_kd   = st.slider("Cost of Debt (%)", 1.0, 12.0, 5.0, 0.1, key="wacc_kd") / 100
    with wc3:
        wacc_tax  = st.slider("Tax Rate (%)", 10.0, 45.0, 25.0, 1.0, key="wacc_tax") / 100
        wacc_lev  = st.slider("D/(D+E) (%)",  0.0, 80.0, 30.0, 1.0, key="wacc_lev") / 100

    ke    = compute_cost_equity(wacc_rf, wacc_erp, wacc_beta)
    wacc  = compute_wacc(1 - wacc_lev, wacc_lev, ke, wacc_kd, wacc_tax)
    wm1, wm2, wm3 = st.columns(3)
    wm1.metric("Cost of Equity (Ke)",  f"{ke*100:.2f}%")
    wm2.metric("WACC",                  f"{wacc*100:.2f}%")
    wm3.metric("After-tax Cost of Debt", f"{wacc_kd*(1-wacc_tax)*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 08 — LBO MODEL
# ══════════════════════════════════════════════════════════════════════════════
with t8:
    sec("LBO / Private Equity Model")
    l1, l2, l3 = st.columns(3)
    with l1:
        lbo_ebitda    = st.number_input("EBITDA (M€)", value=100.0, step=5.0)
        lbo_entry_mul = st.slider("Entry EV/EBITDA", 4.0, 20.0, 10.0, 0.5, key="lbo_entry")
        lbo_exit_mul  = st.slider("Exit EV/EBITDA",  4.0, 20.0, 12.0, 0.5, key="lbo_exit")
    with l2:
        lbo_lev       = st.slider("Leverage (D/EV %)",    20.0, 80.0, 60.0, 5.0, key="lbo_lev")   / 100
        lbo_interest  = st.slider("Interest Rate (%)",     2.0, 15.0, 6.0,  0.5, key="lbo_int")   / 100
        lbo_years     = st.slider("Holding Period (yrs)",  3,    10,   5,    1,   key="lbo_yrs")
    with l3:
        lbo_growth    = st.slider("EBITDA Growth (%/yr)", -5.0, 20.0, 5.0, 0.5, key="lbo_grw")   / 100
        lbo_tax       = st.slider("Tax Rate (%)",         10.0, 45.0, 25.0, 1.0, key="lbo_tax")   / 100
        lbo_capex     = st.slider("CapEx / EBITDA (%)",    5.0, 40.0, 15.0, 1.0, key="lbo_capex") / 100

    if st.button("▶ Run LBO Model", type="primary"):
        res_lbo = compute_lbo(
            lbo_ebitda, lbo_entry_mul, lbo_exit_mul, lbo_lev,
            lbo_years, lbo_interest, lbo_growth, lbo_tax, lbo_capex,
        )
        if "error" in res_lbo:
            st.error(res_lbo["error"])
        else:
            lb1, lb2, lb3, lb4, lb5 = st.columns(5)
            lb1.metric("Entry EV (M€)",  f"{f(res_lbo.get('entry_ev'), 1)}")
            lb2.metric("Equity In (M€)", f"{f(res_lbo.get('equity_in'), 1)}")
            lb3.metric("Exit EV (M€)",   f"{f(res_lbo.get('exit_ev'), 1)}")
            lb4.metric("MOIC",           f"{f(res_lbo.get('moic'), 2)}x")
            lb5.metric("IRR",            f"{f(float(res_lbo.get('irr', np.nan) or np.nan)*100, 1)}%")

            # Sensitivity table MOIC × exit multiple
            sec("Sensitivity — MOIC vs Exit Multiple & Leverage")
            exit_mults = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
            lev_vals   = [0.40, 0.50, 0.60, 0.70, 0.80]
            sens_rows  = []
            for lv in lev_vals:
                row = {"Leverage": f"{int(lv*100)}%"}
                for em in exit_mults:
                    try:
                        r = compute_lbo(lbo_ebitda, lbo_entry_mul, em, lv,
                                        lbo_years, lbo_interest, lbo_growth, lbo_tax, lbo_capex)
                        row[f"Exit {em:.0f}x"] = f"{f(r.get('moic'), 2)}x"
                    except Exception:
                        row[f"Exit {em:.0f}x"] = "—"
                sens_rows.append(row)
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 09 — AI INVESTMENT MEMO
# ══════════════════════════════════════════════════════════════════════════════
with t9:
    sec("AI Investment Memo")
    period_str = f"{start_str} → {end_str}"

    if not api_key:
        st.info(
            "💡 **Local analysis mode** — no API key required. "
            "The memo is generated automatically from computed metrics. "
            "Add an Anthropic key in the sidebar for Claude AI analysis."
        )
    else:
        st.success("🤖 **Claude AI mode** — enhanced analysis via Anthropic API.")

    st.markdown(
        f"**Analysis:** <span style='color:{C_ESG};font-weight:600;'>{esg_name}</span> (ESG) vs "
        f"<span style='color:{C_CLS};font-weight:600;'>{cls_name}</span> (Classic)  "
        f"| **Period:** {period_str} | **Model:** {ff_model_ui}",
        unsafe_allow_html=True,
    )

    if st.button("▶ Generate Investment Memo", type="primary"):
        with st.spinner("Generating memo…"):
            try:
                ai_txt, ai_src = run_ai_analysis(
                    esg_sum, cls_sum, esg_name, cls_name,
                    ff_results=ff_res if ff_res else None,
                    period=period_str, api_key=api_key,
                    model="claude-sonnet-4-6",
                )
                st.session_state.ai_txt = ai_txt
                st.session_state.ai_src = ai_src
            except Exception as ex:
                import traceback as _tb
                st.error(f"Analysis error: {ex}")
                st.code(_tb.format_exc())

    if st.session_state.ai_txt:
        src = st.session_state.ai_src
        src_label = "🤖 Claude AI" if src == "claude_api" else "🔢 Local analysis engine"
        st.caption(f"Source: {src_label}")
        st.markdown(
            f"<div class='memo'>{st.session_state.ai_txt}</div>",
            unsafe_allow_html=True,
        )

    # Quick prompts (Claude only)
    if api_key:
        sec("Quick Analytical Prompts (Claude AI)")
        qc1, qc2 = st.columns(2)
        with qc1:
            if st.button("📊 Risk Profile Summary"):
                try:
                    import anthropic
                    prompt = (
                        f"En 3 points quantitatifs, compare le profil de risque:\n"
                        f"- {esg_name}: Sharpe={f(esg_sum.get('Sharpe Ratio'), 3)}, "
                        f"Vol={f(esg_sum.get('Ann. Volatility (%)'), 2)}%, "
                        f"MaxDD={f(esg_sum.get('Max Drawdown (%)'), 2)}%, "
                        f"Beta={f(esg_sum.get('Beta'), 3)}\n"
                        f"- {cls_name}: Sharpe={f(cls_sum.get('Sharpe Ratio'), 3)}, "
                        f"Vol={f(cls_sum.get('Ann. Volatility (%)'), 2)}%, "
                        f"MaxDD={f(cls_sum.get('Max Drawdown (%)'), 2)}%, "
                        f"Beta={f(cls_sum.get('Beta'), 3)}\n"
                        f"Ton professionnel, en français."
                    )
                    client = anthropic.Anthropic(api_key=api_key)
                    resp = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=400,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown(resp.content[0].text)
                except Exception as ex:
                    st.error(str(ex))
        with qc2:
            if st.button("🔬 Alpha Interpretation"):
                if ff_res:
                    try:
                        import anthropic
                        esg_ff = ff_res.get(esg_name, {})
                        prompt = (
                            f"En 2-3 phrases, interprète pour un comité d'investissement:\n"
                            f"Alpha {ff_model_ui} de {esg_name} = "
                            f"{f(esg_ff.get('alpha_annual_pct'), 2)}%/an, "
                            f"t-stat={f(esg_ff.get('alpha_tstat'), 3)}, "
                            f"p={f(esg_ff.get('alpha_pvalue'), 4)}, R²={f(esg_ff.get('r_squared'), 3)}.\n"
                            f"Est-ce économiquement et statistiquement significatif? En français."
                        )
                        client = anthropic.Anthropic(api_key=api_key)
                        resp = client.messages.create(
                            model="claude-sonnet-4-6", max_tokens=300,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.markdown(resp.content[0].text)
                    except Exception as ex:
                        st.error(str(ex))
                else:
                    st.info("Run factor models first.")
