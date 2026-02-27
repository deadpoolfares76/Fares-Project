"""WACC, LBO and fundamental analysis"""
import numpy as np
from scipy.optimize import brentq

def compute_wacc(equity, debt, cost_equity, cost_debt, tax_rate):
    total=equity+debt
    if total==0: return np.nan
    return (equity/total)*cost_equity+(debt/total)*cost_debt*(1-tax_rate)

def compute_cost_equity(rf, erp, beta):
    return rf+beta*erp

def compute_lbo(ebitda, entry_multiple, exit_multiple, leverage_ratio,
                holding_years, interest_rate, growth_rate, tax_rate, capex_pct):
    try:
        entry_ev=ebitda*entry_multiple
        equity_in=entry_ev*(1-leverage_ratio)
        debt=entry_ev*leverage_ratio
        current_ebitda=ebitda
        current_debt=debt
        for yr in range(int(holding_years)):
            current_ebitda*=(1+growth_rate)
            ebit=current_ebitda*(1-capex_pct)
            interest=current_debt*interest_rate
            tax=max(0,(ebit-interest)*tax_rate)
            fcf=ebit-interest-tax
            current_debt=max(0,current_debt-fcf)
        exit_ev=current_ebitda*exit_multiple
        equity_out=exit_ev-current_debt
        moic=equity_out/equity_in if equity_in>0 else np.nan
        cfs=[-equity_in]+[0]*(int(holding_years)-1)+[equity_out]
        def npv(r):
            return sum(cf/(1+r)**t for t,cf in enumerate(cfs))
        try:
            irr=brentq(npv,-0.9999,10.0,maxiter=500)
        except Exception:
            irr=np.nan
        return {"entry_ev":entry_ev,"equity_in":equity_in,"debt":debt,
                "exit_ev":exit_ev,"equity_out":equity_out,"moic":moic,"irr":irr,
                "exit_ebitda":current_ebitda,"residual_debt":current_debt}
    except Exception as ex:
        return {"error":str(ex)}
