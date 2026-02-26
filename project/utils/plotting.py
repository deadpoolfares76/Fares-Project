"""
Plotting Module — All Plotly charts for the Streamlit app.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Color palette
COLORS = {
    "esg": "#00C49A",        # teal/green for ESG
    "classic": "#FF6B6B",    # coral/red for classic
    "market": "#4C9EFF",     # blue for market benchmark
    "neutral": "#A9A9A9",    # grey
    "bg": "#0E1117",
    "card": "#1E2130",
    "grid": "#2A2D3E",
    "text": "#E0E0E0",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    legend=dict(
        bgcolor="rgba(30,33,48,0.8)",
        bordercolor=COLORS["grid"],
        borderwidth=1,
    ),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
)


def _apply_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    layout = dict(**LAYOUT_DEFAULTS)
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color=COLORS["text"]))
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CUMULATIVE RETURNS
# ─────────────────────────────────────────────────────────────────────────────
def plot_cumulative_returns(returns_esg: pd.Series, returns_classic: pd.Series,
                             returns_market: pd.Series = None,
                             esg_name: str = "ESG",
                             classic_name: str = "Classic",
                             base: float = 100.0) -> go.Figure:
    """Normalised cumulative wealth index starting at `base`."""
    fig = go.Figure()

    def wealth(r):
        return base * (1 + r).cumprod()

    fig.add_trace(go.Scatter(
        x=returns_esg.index, y=wealth(returns_esg),
        name=esg_name, line=dict(color=COLORS["esg"], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=returns_classic.index, y=wealth(returns_classic),
        name=classic_name, line=dict(color=COLORS["classic"], width=2.5)
    ))
    if returns_market is not None:
        fig.add_trace(go.Scatter(
            x=returns_market.index, y=wealth(returns_market),
            name="Market", line=dict(color=COLORS["market"], width=1.5,
                                     dash="dash")
        ))

    _apply_dark_layout(fig, "Cumulative Performance (Rebased to 100)")
    fig.update_yaxes(title_text="Value (indexed)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────
def plot_drawdowns(dd_esg: pd.Series, dd_classic: pd.Series,
                   esg_name: str = "ESG",
                   classic_name: str = "Classic") -> go.Figure:
    """Underwater (drawdown) chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_esg.index, y=dd_esg * 100,
        name=esg_name, fill="tozeroy",
        fillcolor=f"rgba(0,196,154,0.25)",
        line=dict(color=COLORS["esg"], width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=dd_classic.index, y=dd_classic * 100,
        name=classic_name, fill="tozeroy",
        fillcolor=f"rgba(255,107,107,0.25)",
        line=dict(color=COLORS["classic"], width=1.5)
    ))
    _apply_dark_layout(fig, "Drawdown (%)")
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING METRICS
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling(series_esg: pd.Series, series_classic: pd.Series,
                 title: str, ylabel: str,
                 esg_name: str = "ESG",
                 classic_name: str = "Classic",
                 pct: bool = False) -> go.Figure:
    fig = go.Figure()
    mult = 100 if pct else 1
    fig.add_trace(go.Scatter(
        x=series_esg.index, y=series_esg * mult,
        name=esg_name, line=dict(color=COLORS["esg"], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=series_classic.index, y=series_classic * mult,
        name=classic_name, line=dict(color=COLORS["classic"], width=2)
    ))
    _apply_dark_layout(fig, title)
    fig.update_yaxes(title_text=ylabel + (" (%)" if pct else ""))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RETURN DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_return_distribution(returns_esg: pd.Series,
                              returns_classic: pd.Series,
                              esg_name: str = "ESG",
                              classic_name: str = "Classic",
                              confidence: float = 0.95) -> go.Figure:
    """Histogram with VaR markers."""
    from scipy import stats as scipy_stats

    fig = go.Figure()
    bins = 80

    for rets, name, color in [
        (returns_esg, esg_name, COLORS["esg"]),
        (returns_classic, classic_name, COLORS["classic"]),
    ]:
        fig.add_trace(go.Histogram(
            x=rets * 100,
            name=name,
            nbinsx=bins,
            opacity=0.6,
            marker_color=color,
        ))
        var = np.percentile(rets, (1 - confidence) * 100) * 100
        fig.add_vline(
            x=var,
            line=dict(color=color, dash="dash", width=2),
            annotation_text=f"VaR {int(confidence*100)}%: {var:.2f}%",
            annotation_font_color=color,
        )

    fig.update_layout(barmode="overlay")
    _apply_dark_layout(fig, "Daily Return Distribution with VaR")
    fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART — RISK METRICS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar(metrics_esg: dict, metrics_classic: dict,
               esg_name: str = "ESG",
               classic_name: str = "Classic") -> go.Figure:
    """
    Radar/spider chart comparing normalised risk metrics.
    Higher = better for all metrics shown.
    """
    metric_keys = [
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Ann. Return (%)", "Treynor Ratio",
    ]
    labels = []
    vals_esg = []
    vals_classic = []

    for key in metric_keys:
        e = metrics_esg.get(key, 0) or 0
        c = metrics_classic.get(key, 0) or 0
        if np.isnan(e) or np.isnan(c):
            continue
        labels.append(key)
        vals_esg.append(e)
        vals_classic.append(c)

    # Normalise each metric to [0, 1] range
    def normalise(vals):
        arr = np.array(vals, dtype=float)
        min_v, max_v = arr.min(), arr.max()
        if max_v == min_v:
            return np.ones_like(arr) * 0.5
        return (arr - min_v) / (max_v - min_v)

    n_esg = normalise(vals_esg + vals_classic)[:len(vals_esg)]
    n_classic = normalise(vals_esg + vals_classic)[len(vals_esg):]

    labels_closed = labels + [labels[0]]
    esg_closed = list(n_esg) + [n_esg[0]]
    classic_closed = list(n_classic) + [n_classic[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=esg_closed, theta=labels_closed,
        fill="toself", name=esg_name,
        fillcolor=f"rgba(0,196,154,0.2)",
        line=dict(color=COLORS["esg"], width=2)
    ))
    fig.add_trace(go.Scatterpolar(
        r=classic_closed, theta=labels_closed,
        fill="toself", name=classic_name,
        fillcolor=f"rgba(255,107,107,0.2)",
        line=dict(color=COLORS["classic"], width=2)
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=COLORS["grid"],
                            linecolor=COLORS["grid"]),
            angularaxis=dict(gridcolor=COLORS["grid"],
                             linecolor=COLORS["grid"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        title="Risk-Adjusted Performance Radar",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR LOADINGS BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_factor_loadings(ff_results_esg: dict,
                          ff_results_classic: dict,
                          esg_name: str = "ESG",
                          classic_name: str = "Classic") -> go.Figure:
    """Grouped bar chart comparing factor loadings."""
    factors_e = ff_results_esg.get("factors", {})
    factors_c = ff_results_classic.get("factors", {})
    all_factors = sorted(set(list(factors_e.keys()) + list(factors_c.keys())))

    if not all_factors:
        return go.Figure()

    loadings_e = [factors_e.get(f, {}).get("loading", 0) for f in all_factors]
    loadings_c = [factors_c.get(f, {}).get("loading", 0) for f in all_factors]

    fig = go.Figure(data=[
        go.Bar(name=esg_name, x=all_factors, y=loadings_e,
               marker_color=COLORS["esg"], opacity=0.85),
        go.Bar(name=classic_name, x=all_factors, y=loadings_c,
               marker_color=COLORS["classic"], opacity=0.85),
    ])
    fig.update_layout(barmode="group")
    _apply_dark_layout(fig, "Factor Loadings Comparison")
    fig.update_yaxes(title_text="Factor Loading (β)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# METRICS TABLE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_heatmap(summary_df: pd.DataFrame) -> go.Figure:
    """
    Display the comparison metrics as a styled table using Plotly.
    """
    df = summary_df.reset_index()
    headers = list(df.columns)
    cells = [df[col].tolist() for col in headers]

    fig = go.Figure(go.Table(
        header=dict(
            values=headers,
            fill_color=COLORS["card"],
            font=dict(color=COLORS["text"], size=13, family="Inter"),
            align="center",
            height=40,
        ),
        cells=dict(
            values=cells,
            fill_color=[[COLORS["bg"] if i % 2 == 0
                          else COLORS["card"]] * len(df)
                         for i in range(len(headers))],
            font=dict(color=COLORS["text"], size=12, family="Inter"),
            align=["left"] + ["center"] * (len(headers) - 1),
            height=32,
        )
    ))
    _apply_dark_layout(fig, "Portfolio Comparison Metrics")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# WACC WATERFALL
# ─────────────────────────────────────────────────────────────────────────────
def plot_wacc_waterfall(wacc_result: dict, title: str = "WACC Breakdown") -> go.Figure:
    """Waterfall chart showing WACC components."""
    if not wacc_result or "wacc" not in wacc_result:
        return go.Figure()

    fig = go.Figure(go.Waterfall(
        name="WACC",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Equity Component", "Debt Component", "WACC"],
        text=[
            f"{wacc_result.get('equity_contribution_pct', 0):.2f}%",
            f"{wacc_result.get('debt_contribution_pct', 0):.2f}%",
            f"{wacc_result.get('wacc_pct', 0):.2f}%",
        ],
        y=[
            wacc_result.get("equity_contribution_pct", 0),
            wacc_result.get("debt_contribution_pct", 0),
            0,
        ],
        connector=dict(line=dict(color=COLORS["grid"])),
        decreasing=dict(marker=dict(color=COLORS["classic"])),
        increasing=dict(marker=dict(color=COLORS["esg"])),
        totals=dict(marker=dict(color=COLORS["market"])),
    ))
    _apply_dark_layout(fig, title)
    fig.update_yaxes(title_text="Contribution to WACC (%)", ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_matrix(returns_df: pd.DataFrame) -> go.Figure:
    """Heatmap of the correlation matrix."""
    corr = returns_df.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    _apply_dark_layout(fig, "Correlation Matrix")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# REGIME COMPARISON BAR
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_comparison(regime_esg: dict, regime_classic: dict,
                            metric: str = "sharpe",
                            esg_name: str = "ESG",
                            classic_name: str = "Classic") -> go.Figure:
    """Compare a metric across regimes for both portfolios."""
    metric_map = {
        "sharpe": "sharpe",
        "ann_return": "ann_return_pct",
        "ann_vol": "ann_vol_pct",
        "beta": "beta",
        "max_dd": "max_dd_pct",
        "cvar": "cvar_95_pct",
    }
    key = metric_map.get(metric, metric)

    regimes = list(regime_esg.keys())
    vals_esg = [regime_esg.get(r, {}).get(key, 0) for r in regimes]
    vals_classic = [regime_classic.get(r, {}).get(key, 0) for r in regimes]

    fig = go.Figure(data=[
        go.Bar(name=esg_name, x=regimes, y=vals_esg,
               marker_color=COLORS["esg"]),
        go.Bar(name=classic_name, x=regimes, y=vals_classic,
               marker_color=COLORS["classic"]),
    ])
    fig.update_layout(barmode="group")
    _apply_dark_layout(fig, f"{metric.replace('_', ' ').title()} by Volatility Regime")
    return fig
