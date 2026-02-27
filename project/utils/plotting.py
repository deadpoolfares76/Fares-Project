"""
plotting.py — Institutional Plotly Visualisations
Dark fintech theme, rich hover text, dynamic annotations.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

from utils.css_theme import COLORS, PLOTLY_TEMPLATE

# ── Shorthand colours
_ESG   = COLORS["accent_esg"]
_CLS   = COLORS["accent_cls"]
_MKT   = COLORS["accent_mkt"]
_GOLD  = COLORS["accent_gold"]
_PURP  = COLORS["accent_purple"]
_BG    = COLORS["bg_card"]
_BG2   = COLORS["bg_main"]
_BORD  = COLORS["border"]
_TXT   = COLORS["text_primary"]
_TXT2  = COLORS["text_secondary"]


def _base_layout(**kwargs) -> dict:
    base = dict(PLOTLY_TEMPLATE["layout"])
    base.update(kwargs)
    return base


def _fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**_base_layout(height=height))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CUMULATIVE WEALTH
# ─────────────────────────────────────────────────────────────────────────────
def plot_cumulative_wealth(esg_ret: pd.Series, classic_ret: pd.Series,
                            market_ret: pd.Series,
                            esg_name: str = "ESG",
                            classic_name: str = "Classic",
                            crisis_periods: Optional[list] = None) -> go.Figure:
    """Rebased to 100. Annotates Max Drawdown point. Optional crisis shading."""
    esg_w   = (1 + esg_ret).cumprod() * 100
    cls_w   = (1 + classic_ret).cumprod() * 100
    mkt_w   = (1 + market_ret).cumprod() * 100

    # Max drawdown point for ESG
    dd_esg = ((1 + esg_ret).cumprod() - (1 + esg_ret).cumprod().cummax()) / (1 + esg_ret).cumprod().cummax()
    mdd_date = dd_esg.idxmin()

    fig = go.Figure()

    # Crisis shading
    if crisis_periods:
        for cname, cstart, cend, ccolor in crisis_periods:
            fig.add_vrect(x0=cstart, x1=cend,
                          fillcolor=ccolor, opacity=0.08,
                          layer="below", line_width=0,
                          annotation_text=cname,
                          annotation_position="top left",
                          annotation_font={"size": 9, "color": ccolor})

    fig.add_trace(go.Scatter(
        x=esg_w.index, y=esg_w.values,
        name=esg_name, line=dict(color=_ESG, width=2),
        hovertemplate=f"<b>{esg_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Valeur: %{{y:.1f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cls_w.index, y=cls_w.values,
        name=classic_name, line=dict(color=_CLS, width=2),
        hovertemplate=f"<b>{classic_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Valeur: %{{y:.1f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=mkt_w.index, y=mkt_w.values,
        name="S&P 500", line=dict(color=_MKT, width=1.5, dash="dot"),
        hovertemplate="<b>S&P 500</b><br>Date: %{x|%Y-%m-%d}<br>Valeur: %{y:.1f}<extra></extra>",
    ))

    # Max Drawdown annotation
    if mdd_date in esg_w.index:
        mdd_val = esg_w.loc[mdd_date]
        fig.add_annotation(
            x=mdd_date, y=float(mdd_val),
            text=f"Max DD ESG<br>{dd_esg.min()*100:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor=_ESG,
            font=dict(color=_ESG, size=10),
            bgcolor=_BG, bordercolor=_ESG, borderwidth=1,
            ax=40, ay=-40,
        )

    fig.update_layout(**_base_layout(
        title=dict(text="Cumulative Wealth Index — Rebased to 100",
                   font=dict(size=13, color=_TXT2)),
        height=440,
        xaxis_title="", yaxis_title="Wealth Index (Base 100)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, x=0),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_drawdowns(esg_dd: pd.Series, classic_dd: pd.Series,
                    esg_name: str = "ESG", classic_name: str = "Classic") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=esg_dd.index, y=esg_dd.values * 100,
        name=esg_name, fill="tozeroy",
        line=dict(color=_ESG, width=1.5),
        fillcolor=f"rgba(0,212,170,0.15)",
        hovertemplate=f"<b>{esg_name}</b> DD: %{{y:.2f}}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=classic_dd.index, y=classic_dd.values * 100,
        name=classic_name, fill="tozeroy",
        line=dict(color=_CLS, width=1.5),
        fillcolor=f"rgba(255,69,96,0.12)",
        hovertemplate=f"<b>{classic_name}</b> DD: %{{y:.2f}}%<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Drawdown Analysis", font=dict(size=13, color=_TXT2)),
        height=320, yaxis_title="Drawdown (%)", hovermode="x unified",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING METRICS (multi-panel)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_metrics(r_sharpe_esg: pd.Series, r_sharpe_cls: pd.Series,
                          r_vol_esg: pd.Series, r_vol_cls: pd.Series,
                          r_beta_esg: pd.Series, r_beta_cls: pd.Series,
                          r_corr_esg: pd.Series, r_corr_cls: pd.Series,
                          esg_name: str = "ESG",
                          classic_name: str = "Classic") -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         vertical_spacing=0.06,
                         subplot_titles=["Rolling Sharpe", "Rolling Volatility (%)",
                                          "Rolling Beta", "Rolling Correlation vs Market"])

    pairs = [
        (r_sharpe_esg, r_sharpe_cls, 1, _ESG, _CLS, None),
        (r_vol_esg * 100, r_vol_cls * 100, 2, _ESG, _CLS, "%"),
        (r_beta_esg, r_beta_cls, 3, _ESG, _CLS, None),
        (r_corr_esg, r_corr_cls, 4, _ESG, _CLS, None),
    ]
    for s_esg, s_cls, row, c_esg, c_cls, suffix in pairs:
        sfx = suffix or ""
        fig.add_trace(go.Scatter(
            x=s_esg.index, y=s_esg.values,
            name=esg_name, line=dict(color=c_esg, width=1.5),
            showlegend=(row == 1),
            hovertemplate=f"<b>{esg_name}</b>: %{{y:.3f}}{sfx}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=s_cls.index, y=s_cls.values,
            name=classic_name, line=dict(color=c_cls, width=1.5),
            showlegend=(row == 1),
            hovertemplate=f"<b>{classic_name}</b>: %{{y:.3f}}{sfx}<extra></extra>",
        ), row=row, col=1)

    # Zero line on beta and correlation
    for row in [3, 4]:
        fig.add_hline(y=1.0 if row == 3 else 0.0,
                       line=dict(color=_BORD, width=1, dash="dash"), row=row, col=1)

    fig.update_layout(**_base_layout(height=700, hovermode="x unified",
                                      showlegend=True,
                                      legend=dict(orientation="h", y=-0.04, x=0)))
    # Style subplots
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(
                gridcolor=_BORD, linecolor=_BORD,
                tickfont=dict(color=_TXT2)
            )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RETURN DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_return_distribution(esg_ret: pd.Series, classic_ret: pd.Series,
                              var_esg: float, var_cls: float,
                              cvar_esg: float, cvar_cls: float,
                              esg_name: str = "ESG",
                              classic_name: str = "Classic") -> go.Figure:
    from scipy import stats as scipy_stats
    fig = go.Figure()

    for ret, name, color, var, cvar in [
        (esg_ret, esg_name, _ESG, var_esg, cvar_esg),
        (classic_ret, classic_name, _CLS, var_cls, cvar_cls),
    ]:
        r = ret.dropna().values * 100
        fig.add_trace(go.Histogram(
            x=r, name=name, histnorm="probability density",
            opacity=0.45, marker_color=color,
            nbinsx=80,
            hovertemplate=f"<b>{name}</b><br>Bin: %{{x:.2f}}%<br>Density: %{{y:.4f}}<extra></extra>",
        ))
        # Normal fit overlay
        mu, sigma = r.mean(), r.std()
        x_line = np.linspace(r.min(), r.max(), 300)
        y_line = scipy_stats.norm.pdf(x_line, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, name=f"{name} Normal fit",
            line=dict(color=color, width=1.5, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))
        # VaR line
        if not np.isnan(var):
            fig.add_vline(x=var * 100, line=dict(color=color, width=1.5, dash="dot"),
                          annotation_text=f"VaR {name}", annotation_position="top",
                          annotation_font=dict(color=color, size=9))
        # CVaR line
        if not np.isnan(cvar):
            fig.add_vline(x=cvar * 100, line=dict(color=color, width=1, dash="dot"),
                          annotation_text=f"CVaR {name}", annotation_position="top right",
                          annotation_font=dict(color=color, size=9))

    fig.update_layout(**_base_layout(
        title=dict(text="Return Distribution with VaR / CVaR", font=dict(size=13, color=_TXT2)),
        height=380, barmode="overlay",
        xaxis_title="Daily Return (%)", yaxis_title="Probability Density",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR LOADINGS
# ─────────────────────────────────────────────────────────────────────────────
def plot_factor_loadings(esg_results: dict, classic_results: dict,
                          esg_name: str = "ESG",
                          classic_name: str = "Classic") -> go.Figure:
    factors_esg = esg_results.get("factors", {})
    factors_cls = classic_results.get("factors", {}) if classic_results else {}
    all_factors = list(factors_esg.keys())

    esg_loads  = [factors_esg[f]["loading"]   for f in all_factors]
    esg_ci_lo  = [factors_esg[f]["ci_95_lo"]  for f in all_factors]
    esg_ci_hi  = [factors_esg[f]["ci_95_hi"]  for f in all_factors]
    esg_err_lo = [l - e for l, e in zip(esg_loads, esg_ci_lo)]
    esg_err_hi = [h - e for h, e in zip(esg_ci_hi, esg_loads)]

    cls_loads  = [factors_cls.get(f, {}).get("loading", 0)   for f in all_factors]
    cls_ci_lo  = [factors_cls.get(f, {}).get("ci_95_lo", 0)  for f in all_factors]
    cls_ci_hi  = [factors_cls.get(f, {}).get("ci_95_hi", 0)  for f in all_factors]
    cls_err_lo = [l - e for l, e in zip(cls_loads, cls_ci_lo)]
    cls_err_hi = [h - e for h, e in zip(cls_ci_hi, cls_loads)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=esg_name, x=all_factors, y=esg_loads,
        error_y=dict(type="data", symmetric=False,
                     arrayminus=esg_err_lo, array=esg_err_hi, color=_ESG),
        marker_color=_ESG, opacity=0.85,
        hovertemplate="<b>" + esg_name + "</b><br>%{x}: %{y:.4f}<extra></extra>",
    ))
    if factors_cls:
        fig.add_trace(go.Bar(
            name=classic_name, x=all_factors, y=cls_loads,
            error_y=dict(type="data", symmetric=False,
                         arrayminus=cls_err_lo, array=cls_err_hi, color=_CLS),
            marker_color=_CLS, opacity=0.85,
            hovertemplate="<b>" + classic_name + "</b><br>%{x}: %{y:.4f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=_BORD, width=1))
    fig.update_layout(**_base_layout(
        title=dict(text="Factor Loadings with 95% Confidence Intervals",
                   font=dict(size=13, color=_TXT2)),
        height=380, barmode="group",
        xaxis_title="Factor", yaxis_title="Loading (β)",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING ALPHA
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_alpha(r_alpha_esg: pd.Series, r_alpha_cls: pd.Series,
                        esg_name: str = "ESG",
                        classic_name: str = "Classic") -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=float(r_alpha_esg.dropna().max() * 1.1 if r_alpha_esg.dropna().size > 0 else 20),
                  fillcolor="rgba(0,212,170,0.04)", layer="below", line_width=0)
    fig.add_hrect(y0=float(r_alpha_esg.dropna().min() * 1.1 if r_alpha_esg.dropna().size > 0 else -20), y1=0,
                  fillcolor="rgba(255,69,96,0.04)", layer="below", line_width=0)

    for s, name, color in [(r_alpha_esg, esg_name, _ESG), (r_alpha_cls, classic_name, _CLS)]:
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=name, line=dict(color=color, width=1.5),
            hovertemplate=f"<b>{name}</b> α: %{{y:.2f}}%/yr<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=_BORD, width=1, dash="dash"))
    fig.update_layout(**_base_layout(
        title=dict(text="Rolling Alpha — Annualised (%/year)",
                   font=dict(size=13, color=_TXT2)),
        height=320, yaxis_title="Alpha (%/year)", hovermode="x unified",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STRESS TEST HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_stress_heatmap(df: pd.DataFrame, esg_name: str, classic_name: str) -> go.Figure:
    """Heatmap of total returns across all crisis scenarios."""
    cols = [f"{esg_name} Total Ret (%)", f"{classic_name} Total Ret (%)"]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return go.Figure()
    sub = df[cols].dropna(how="all")

    z   = sub.values.astype(float)
    x   = [c.replace(" Total Ret (%)", "") for c in cols]
    y   = list(sub.index)
    txt = np.round(z, 1).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, text=txt, texttemplate="%{text}%",
        colorscale=[
            [0.0,  "#FF4560"], [0.35, "#FF8C42"],
            [0.5,  "#1E2235"], [0.65, "#4B9EFF"],
            [1.0,  "#00D4AA"],
        ],
        zmid=0, showscale=True,
        colorbar=dict(title="%", tickfont=dict(color=_TXT2)),
        hovertemplate="<b>%{y}</b> — %{x}<br>Return: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Crisis Performance Heatmap — Total Return (%)",
                   font=dict(size=13, color=_TXT2)),
        height=max(300, len(y) * 55),
        xaxis=dict(tickfont=dict(color=_TXT), side="top"),
        yaxis=dict(tickfont=dict(color=_TXT2), autorange="reversed"),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CRISIS CUMULATIVE RETURNS
# ─────────────────────────────────────────────────────────────────────────────
def plot_crisis_paths(esg_cumret: Optional[pd.Series],
                       cls_cumret: Optional[pd.Series],
                       mkt_cumret: Optional[pd.Series],
                       crisis_name: str,
                       esg_name: str = "ESG",
                       classic_name: str = "Classic") -> go.Figure:
    fig = go.Figure()
    for series, name, color in [
        (esg_cumret, esg_name, _ESG),
        (cls_cumret, classic_name, _CLS),
        (mkt_cumret, "S&P 500", _MKT),
    ]:
        if series is not None and not series.empty:
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=name, line=dict(color=color, width=2),
                hovertemplate=f"<b>{name}</b>: %{{y:.2f}}%<extra></extra>",
            ))
    fig.add_hline(y=0, line=dict(color=_BORD, width=1, dash="dash"))
    fig.update_layout(**_base_layout(
        title=dict(text=f"Crisis Path — {crisis_name}",
                   font=dict(size=13, color=_TXT2)),
        height=340, yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def plot_monte_carlo(mc_esg, mc_cls,
                      esg_name: str = "ESG",
                      classic_name: str = "Classic",
                      max_paths_shown: int = 200) -> go.Figure:
    """Fan chart: show subset of paths + P5/P25/P50/P75/P95 bands."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                         subplot_titles=[f"{esg_name} — GBM Simulation",
                                          f"{classic_name} — GBM Simulation"])

    for col_idx, (mc, name, c_main) in enumerate(
            [(mc_esg, esg_name, _ESG), (mc_cls, classic_name, _CLS)], 1):

        t = mc.time_axis
        pcts = mc.percentiles
        n_show = min(max_paths_shown, mc.n_simulations)

        # Sample paths (light, transparent)
        rng = np.random.default_rng(99)
        idxs = rng.choice(mc.n_simulations, n_show, replace=False)
        for i in idxs:
            fig.add_trace(go.Scatter(
                x=t, y=mc.paths[i],
                line=dict(color=c_main, width=0.3),
                opacity=0.08, showlegend=False, hoverinfo="skip",
                mode="lines",
            ), row=1, col=col_idx)

        # Percentile bands
        band_pairs = [(5, 95, 0.15), (25, 75, 0.25)]
        for lo, hi, alpha in band_pairs:
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([pcts[hi], pcts[lo][::-1]]),
                fill="toself", fillcolor=f"rgba({_hex_to_rgb(c_main)},{alpha})",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                mode="lines",
            ), row=1, col=col_idx)

        # Median
        fig.add_trace(go.Scatter(
            x=t, y=pcts[50], name=f"{name} P50",
            line=dict(color=c_main, width=2.5),
            hovertemplate=f"<b>{name} Median</b><br>Yr: %{{x:.2f}}<br>Wealth: %{{y:.3f}}x<extra></extra>",
        ), row=1, col=col_idx)

        # P5 / P95
        for p, dash in [(5, "dot"), (95, "dash")]:
            fig.add_trace(go.Scatter(
                x=t, y=pcts[p], name=f"{name} P{p}",
                line=dict(color=c_main, width=1.2, dash=dash),
                hovertemplate=f"<b>{name} P{p}</b>: %{{y:.3f}}x<extra></extra>",
            ), row=1, col=col_idx)

    fig.add_hline(y=1.0, line=dict(color=_BORD, width=1, dash="dash"))
    fig.update_layout(**_base_layout(
        height=460,
        title=dict(text="Monte Carlo Simulation — GBM Wealth Paths (×)",
                   font=dict(size=13, color=_TXT2)),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, x=0),
    ))
    return fig


def plot_mc_terminal_distribution(mc_esg, mc_cls,
                                    esg_name: str = "ESG",
                                    classic_name: str = "Classic") -> go.Figure:
    """Histogram of terminal wealth values for both portfolios."""
    from scipy import stats as scipy_stats
    fig = go.Figure()
    for mc, name, color in [(mc_esg, esg_name, _ESG), (mc_cls, classic_name, _CLS)]:
        d = mc.final_dist
        fig.add_trace(go.Histogram(
            x=d, name=name, histnorm="probability density",
            opacity=0.5, marker_color=color, nbinsx=80,
            hovertemplate=f"<b>{name}</b>: %{{x:.3f}}x — density: %{{y:.4f}}<extra></extra>",
        ))
        # Kernel density overlay
        kde_x = np.linspace(d.min(), d.max(), 300)
        kde_y = scipy_stats.gaussian_kde(d)(kde_x)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y, name=f"{name} KDE",
            line=dict(color=color, width=2), showlegend=False, hoverinfo="skip",
        ))
        # Loss threshold
        fig.add_vline(x=1.0, line=dict(color=_GOLD, width=1.5, dash="dash"),
                       annotation_text="Break-even", annotation_font=dict(color=_GOLD, size=9))
    fig.update_layout(**_base_layout(
        title=dict(text="Terminal Wealth Distribution (×)",
                   font=dict(size=13, color=_TXT2)),
        height=360, barmode="overlay",
        xaxis_title="Terminal Wealth (×)", yaxis_title="Probability Density",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY REGIME
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_cumulative(esg_ret: pd.Series, classic_ret: pd.Series,
                             regime_mask: pd.Series,
                             esg_name: str = "ESG",
                             classic_name: str = "Classic") -> go.Figure:
    common = esg_ret.index.intersection(classic_ret.index).intersection(regime_mask.index)
    esg_w  = (1 + esg_ret.loc[common]).cumprod() * 100
    cls_w  = (1 + classic_ret.loc[common]).cumprod() * 100
    mask   = regime_mask.loc[common]

    fig = go.Figure()

    # Shade high-vol periods
    in_hvol = False
    hvol_start = None
    for dt, is_high in mask.items():
        if is_high and not in_hvol:
            hvol_start = dt
            in_hvol = True
        elif not is_high and in_hvol:
            fig.add_vrect(x0=hvol_start, x1=dt,
                          fillcolor="rgba(255,69,96,0.10)",
                          layer="below", line_width=0)
            in_hvol = False
    if in_hvol and hvol_start is not None:
        fig.add_vrect(x0=hvol_start, x1=mask.index[-1],
                      fillcolor="rgba(255,69,96,0.10)",
                      layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=esg_w.index, y=esg_w.values, name=esg_name,
        line=dict(color=_ESG, width=2),
        hovertemplate=f"<b>{esg_name}</b>: %{{y:.1f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cls_w.index, y=cls_w.values, name=classic_name,
        line=dict(color=_CLS, width=2),
        hovertemplate=f"<b>{classic_name}</b>: %{{y:.1f}}<extra></extra>",
    ))

    # Legend annotation for shading
    fig.add_annotation(
        x=0.01, y=0.04, xref="paper", yref="paper",
        text="🔴 High Volatility Regime", showarrow=False,
        font=dict(color=_CLS, size=9),
        bgcolor=_BG, bordercolor=_BORD, borderwidth=1,
    )

    fig.update_layout(**_base_layout(
        title=dict(text="Cumulative Returns with Volatility Regime Overlay",
                   font=dict(size=13, color=_TXT2)),
        height=380, hovermode="x unified",
        yaxis_title="Wealth Index (Base 100)",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar(esg_summary: dict, classic_summary: dict,
                esg_name: str = "ESG", classic_name: str = "Classic") -> go.Figure:
    metrics = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
               "Omega Ratio", "Information Ratio"]
    categories = ["Sharpe", "Sortino", "Calmar", "Omega", "Info Ratio"]

    def safe_val(d, k):
        v = d.get(k, 0) or 0
        try:
            return max(float(v), 0)
        except Exception:
            return 0.0

    esg_vals = [safe_val(esg_summary, m) for m in metrics]
    cls_vals = [safe_val(classic_summary, m) for m in metrics]

    max_vals = [max(e, c, 0.01) for e, c in zip(esg_vals, cls_vals)]
    esg_norm = [v / mx for v, mx in zip(esg_vals, max_vals)]
    cls_norm = [v / mx for v, mx in zip(cls_vals, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=esg_norm + [esg_norm[0]], theta=categories + [categories[0]],
        fill="toself", name=esg_name,
        line=dict(color=_ESG), fillcolor="rgba(0,212,170,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=cls_norm + [cls_norm[0]], theta=categories + [categories[0]],
        fill="toself", name=classic_name,
        line=dict(color=_CLS), fillcolor="rgba(255,69,96,0.12)",
    ))
    fig.update_layout(**_base_layout(
        polar=dict(
            bgcolor=_BG2,
            radialaxis=dict(visible=True, range=[0, 1.1],
                            gridcolor=_BORD, linecolor=_BORD,
                            tickfont=dict(color=_TXT2, size=8)),
            angularaxis=dict(gridcolor=_BORD, linecolor=_BORD,
                              tickfont=dict(color=_TXT, size=11)),
        ),
        title=dict(text="Risk-Adjusted Ratio Comparison",
                   font=dict(size=13, color=_TXT2)),
        height=380,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_matrix(returns_df: pd.DataFrame) -> go.Figure:
    corr = returns_df.dropna().corr()
    z    = corr.values
    labels = list(corr.columns)
    txt  = np.round(z, 3).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels, text=txt, texttemplate="%{text}",
        colorscale=[
            [0.0, _CLS], [0.5, _BG], [1.0, _ESG],
        ],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Return Correlation Matrix", font=dict(size=13, color=_TXT2)),
        height=360,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# WACC WATERFALL
# ─────────────────────────────────────────────────────────────────────────────
def plot_wacc_waterfall(rf: float, erp: float, beta: float,
                         ke: float, kd: float, tax: float,
                         equity_weight: float, debt_weight: float,
                         wacc: float) -> go.Figure:
    labels = ["Risk-Free Rate", "ERP × Beta", "Cost of Equity",
              "Cost of Debt (after-tax)", "WACC"]
    measures = ["absolute", "relative", "total", "relative", "total"]
    values   = [rf * 100,
                erp * beta * 100,
                0,
                kd * (1 - tax) * debt_weight * 100,
                0]
    base_vals = [0, rf * 100, 0, 0, 0]

    fig = go.Figure(go.Waterfall(
        name="WACC", orientation="v",
        measure=measures,
        x=labels, y=values,
        texttemplate="%{y:.2f}%", textposition="outside",
        connector=dict(line=dict(color=_BORD, width=1)),
        decreasing=dict(marker=dict(color=_MKT)),
        increasing=dict(marker=dict(color=_ESG)),
        totals=dict(marker=dict(color=_GOLD)),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="WACC Decomposition", font=dict(size=13, color=_TXT2)),
        height=380, yaxis_title="Rate (%)",
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "200,200,200"
