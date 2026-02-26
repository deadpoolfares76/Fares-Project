"""
Fundamental Analysis & Corporate Finance Module
WACC, LBO metrics, and financial ratios.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CAPM COST OF EQUITY
# ─────────────────────────────────────────────────────────────────────────────
def cost_of_equity_capm(beta: float, risk_free_rate: float = 0.04,
                         equity_risk_premium: float = 0.055) -> float:
    """
    Re = Rf + β × ERP
    Returns the annualised cost of equity as a decimal.
    """
    return risk_free_rate + beta * equity_risk_premium


# ─────────────────────────────────────────────────────────────────────────────
# WACC
# ─────────────────────────────────────────────────────────────────────────────
def compute_wacc(equity_value: float,
                 debt_value: float,
                 cost_of_equity: float,
                 cost_of_debt: float,
                 tax_rate: float = 0.25) -> dict:
    """
    WACC = (E/V) × Re + (D/V) × Rd × (1 - T)

    Parameters
    ----------
    equity_value : Market capitalisation
    debt_value   : Total financial debt
    cost_of_equity : Re (annual, decimal)
    cost_of_debt   : Rd (annual, decimal)
    tax_rate       : Corporate tax rate

    Returns
    -------
    dict with WACC and component breakdown
    """
    total_value = equity_value + debt_value
    if total_value == 0:
        return {"wacc": np.nan}

    E_ratio = equity_value / total_value
    D_ratio = debt_value / total_value

    equity_contribution = E_ratio * cost_of_equity
    debt_contribution = D_ratio * cost_of_debt * (1 - tax_rate)
    wacc = equity_contribution + debt_contribution

    return {
        "wacc": wacc,
        "wacc_pct": wacc * 100,
        "equity_weight": E_ratio,
        "debt_weight": D_ratio,
        "cost_of_equity": cost_of_equity,
        "cost_of_equity_pct": cost_of_equity * 100,
        "after_tax_cost_of_debt": cost_of_debt * (1 - tax_rate),
        "after_tax_cod_pct": cost_of_debt * (1 - tax_rate) * 100,
        "equity_contribution_pct": equity_contribution * 100,
        "debt_contribution_pct": debt_contribution * 100,
        "tax_rate": tax_rate,
        "leverage_ratio": D_ratio,
    }


def compute_wacc_from_fundamentals(fundamentals: dict,
                                    risk_free_rate: float = 0.04,
                                    erp: float = 0.055,
                                    tax_rate: float = 0.25) -> dict:
    """
    Derive WACC from yfinance fundamentals dict.
    Falls back to sensible defaults when data is missing.
    """
    beta = fundamentals.get("beta", 1.0) or 1.0
    equity = fundamentals.get("total_equity") or fundamentals.get("market_cap")
    debt = fundamentals.get("total_debt", 0) or 0

    # Cost of equity via CAPM
    re = cost_of_equity_capm(beta, risk_free_rate, erp)

    # Cost of debt: use provided or estimate from interest expense
    rd = fundamentals.get("cost_of_debt")
    if rd is None or np.isnan(rd) or rd == 0:
        rd = 0.05  # default 5%

    if equity is None or np.isnan(equity):
        return {"wacc": np.nan, "error": "Missing equity value"}

    return compute_wacc(equity, debt, re, rd, tax_rate)


# ─────────────────────────────────────────────────────────────────────────────
# LBO / PRIVATE EQUITY METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_moic(entry_equity: float, exit_equity: float) -> float:
    """
    Multiple on Invested Capital = Exit Equity / Entry Equity.
    """
    if entry_equity == 0:
        return np.nan
    return exit_equity / entry_equity


def compute_irr(cash_flows: list[float], initial_guess: float = 0.15) -> float:
    """
    Internal Rate of Return: rate r such that NPV(cash_flows, r) = 0.
    cash_flows: list starting with initial outflow (negative) then inflows.
    """
    def npv(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

    try:
        irr = brentq(npv, -0.9999, 10.0, xtol=1e-8, maxiter=1000)
        return irr
    except (ValueError, RuntimeError):
        return np.nan


def compute_lbo_metrics(ebitda: float,
                         entry_ev_multiple: float,
                         exit_ev_multiple: float,
                         debt_ratio: float = 0.6,
                         holding_years: int = 5,
                         interest_rate: float = 0.07,
                         ebitda_growth_rate: float = 0.05,
                         tax_rate: float = 0.25,
                         capex_to_ebitda: float = 0.15,
                         change_in_wc_pct: float = 0.02) -> dict:
    """
    Simplified LBO model.

    Returns entry/exit equity, MOIC, IRR, and leverage metrics.
    """
    # Entry valuation
    entry_ev = ebitda * entry_ev_multiple
    entry_debt = entry_ev * debt_ratio
    entry_equity = entry_ev - entry_debt

    # Year-by-year projections
    debt = entry_debt
    ebitda_t = ebitda
    cash_flows = [-entry_equity]

    for year in range(1, holding_years + 1):
        ebitda_t *= (1 + ebitda_growth_rate)
        ebit = ebitda_t * 0.75  # assume D&A = 25% EBITDA
        interest = debt * interest_rate
        ebt = ebit - interest
        net_income = ebt * (1 - tax_rate) if ebt > 0 else ebt
        fcf = (net_income
               + ebitda_t * 0.25  # D&A add-back
               - ebitda_t * capex_to_ebitda
               - ebitda_t * change_in_wc_pct
               - interest)  # debt service
        debt = max(0, debt - max(0, fcf))

    # Exit valuation
    exit_ev = ebitda_t * exit_ev_multiple
    exit_equity = exit_ev - debt
    cash_flows.append(exit_equity)

    moic = compute_moic(entry_equity, exit_equity)
    irr = compute_irr(cash_flows)

    return {
        "entry_ev": entry_ev,
        "entry_equity": entry_equity,
        "entry_debt": entry_debt,
        "entry_debt_ebitda": entry_debt / ebitda,
        "exit_ev": exit_ev,
        "exit_equity": exit_equity,
        "exit_debt": debt,
        "exit_ebitda": ebitda_t,
        "exit_debt_ebitda": debt / ebitda_t if ebitda_t > 0 else np.nan,
        "moic": moic,
        "irr_pct": irr * 100 if not np.isnan(irr) else np.nan,
        "holding_years": holding_years,
        "leverage_ratio": debt_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def fundamental_summary_df(fundamentals_dict: dict) -> pd.DataFrame:
    """
    Convert fundamentals dict to a clean display DataFrame.
    """
    fields = {
        "Name": "name",
        "Sector": "sector",
        "Industry": "industry",
        "Market Cap ($M)": ("market_cap", 1e6),
        "Enterprise Value ($M)": ("enterprise_value", 1e6),
        "EBITDA ($M)": ("ebitda", 1e6),
        "EV/EBITDA": "ev_ebitda",
        "EV/EBIT": "ev_ebit",
        "Trailing P/E": "trailing_pe",
        "Forward P/E": "forward_pe",
        "ROE (%)": ("roe", 0.01),
        "ROA (%)": ("roa", 0.01),
        "ROIC (%)": ("roic", 0.01),
        "Beta": "beta",
        "Net Debt ($M)": ("net_debt", 1e6),
        "Net Debt/EBITDA": "net_debt_ebitda",
    }
    rows = {}
    for label, key in fields.items():
        if isinstance(key, tuple):
            field, divisor = key
            val = fundamentals_dict.get(field, np.nan)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                if divisor < 1:  # percentage: multiply
                    rows[label] = f"{val / divisor:.1f}%" if val else "N/A"
                else:  # millions: divide
                    rows[label] = f"{val / divisor:,.1f}" if val else "N/A"
            else:
                rows[label] = "N/A"
        else:
            val = fundamentals_dict.get(key, "N/A")
            if isinstance(val, float) and not np.isnan(val):
                rows[label] = f"{val:.2f}"
            elif val is not None:
                rows[label] = val
            else:
                rows[label] = "N/A"

    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])
