# ESG Analytics Platform — Institutional Edition v2.0

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
├── app.py                    # Main app (9 tabs)
├── utils/css_theme.py        # Dark theme CSS + Plotly template
├── data/data_loader.py       # yfinance + Fama-French (3 fallbacks)
├── analysis/
│   ├── risk_metrics.py       # Sharpe, Sortino, Calmar, Omega, IR, VaR, CVaR...
│   ├── factor_models.py      # CAPM, FF3, FF5, FF5+MOM (HAC-robust OLS)
│   ├── stress_testing.py     # GFC/COVID/Inflation historical crises
│   ├── monte_carlo.py        # GBM simulation (independent + correlated Cholesky)
│   └── fundamental.py        # WACC + LBO model
└── ai_agent/analyzer.py      # Local rule-based + Claude API memo
```

## Tabs
| # | Tab | Content |
|---|-----|---------|
| 01 | Executive | KPI strip, wealth index, drawdown, comparison table |
| 02 | Risk & Perf | Rolling Sharpe/Beta/Vol, distributions, advanced ratios |
| 03 | Factor Models | CAPM/FF3/FF5/FF5+MOM loadings, alpha decomposition, rolling alpha |
| 04 | Stress Test | GFC/COVID/Inflation crises, recovery, custom scenario |
| 05 | Monte Carlo | GBM fan chart, terminal distribution, correlated paths |
| 06 | Regimes | Volatility regime detection, performance by regime |
| 07 | Fundamentals | Market data, WACC calculator |
| 08 | LBO | LBO model, MOIC/IRR, sensitivity table |
| 09 | AI Memo | Local analysis or Claude API investment memo |
