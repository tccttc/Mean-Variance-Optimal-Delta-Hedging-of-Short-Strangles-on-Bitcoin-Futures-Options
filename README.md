# Mean-Variance Optimal Delta-Hedging of Short Strangles on Bitcoin Futures Options

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025**  
**Prof. Wei JIANG**

---

## 📋 Project Overview

This project implements a **mean-variance optimal delta-hedging strategy** for short strangles on Bitcoin futures options. We sell 10% out-of-the-money (OTM) call and put options on Deribit BTC quarterly futures to harvest volatility premium, while minimizing P&L variance through optimal hedging.

### Key Features

- **Mean-Variance Optimization**: Find optimal hedge weights using convex optimization (cvxpy)
- **EWMA Covariance**: Time-varying covariance estimation with λ=0.94 (RiskMetrics methodology)
- **Delta-Neutral Hedging**: Maintain market-neutral exposure while maximizing risk-adjusted returns
- **Comprehensive Backtesting**: Daily rebalancing from Jan 2022 to Dec 2025
- **Performance Analytics**: Sharpe ratio, max drawdown, VaR, and more
- **Robo-Advisor Prototype**: Personalized hedge recommendations based on risk aversion

---

## 🎯 Strategy Description

### Short Strangle

A short strangle involves:
- **Selling an OTM call** (strike = spot × 1.10)
- **Selling an OTM put** (strike = spot × 0.90)

**Profit Profile**: Collect premium (theta decay), lose if BTC moves significantly in either direction.

### Delta Hedging

The strangle has a net delta exposure that changes with spot price. We hedge this exposure using:
- **BTC Spot**: Direct exposure (δ = 1)
- **BTC Futures**: Slightly different exposure due to basis (δ ≈ 1)
- **USDT (Cash)**: No delta exposure (δ = 0)

### Mean-Variance Optimization

From **Lecture 4**: We minimize portfolio variance subject to constraints:

```
min  w'Σw                    (minimize variance)
s.t. Σwᵢ = 1                 (fully invested)
     δ'w = -δ_strangle       (delta-neutral)
     w ≥ 0                   (long-only, optional)
```

Where:
- `w` = portfolio weights
- `Σ` = EWMA covariance matrix
- `δ` = asset delta vector

---

## 📁 Project Structure

```
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
│
├── src/                 # Source code package
│   ├── __init__.py      # Package initialization
│   ├── data.py          # Data fetching and cleaning
│   ├── returns.py       # Returns calculation
│   ├── covariance.py    # EWMA covariance estimation
│   ├── optimize.py      # Mean-variance optimization (cvxpy)
│   ├── backtest.py      # Backtesting engine
│   ├── plots.py         # Visualization module
│   └── robo_advisor.py  # Robo-advisor prototype
│
├── figures/             # Generated plots (created after running)
├── results/             # CSV outputs (created after running)
├── data/                # Data files (CSV, if any)
└── tests/               # Unit tests (optional)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/btc-strangle-hedging.git
cd btc-strangle-hedging

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Analysis

```bash
python main.py
```

This will:
1. Download BTC spot/futures data from yfinance
2. Run backtest from 2022-01-01 to 2025-12-03
3. Generate performance comparison tables
4. Create visualizations in `figures/`
5. Save results to `results/`
6. Demonstrate robo-advisor functionality

### 3. Command Line Options

```bash
# Custom date range
python main.py --start 2023-01-01 --end 2024-12-31

# Quick run (2023-2025 only)
python main.py --quick

# Skip plot generation
python main.py --no-plots

# Interactive robo-advisor
python main.py --interactive

# Custom notional
python main.py --notional 50000
```

---

## 📊 Output Examples

### Performance Summary Table

| Metric              | Unhedged | Naive Hedge | MV Optimal |
|---------------------|----------|-------------|------------|
| Annualized Return   | 12.5%    | 8.2%        | 9.8%       |
| Annualized Volatility| 85.3%   | 32.1%       | 24.6%      |
| Sharpe Ratio        | 0.09     | 0.10        | 0.19       |
| Max Drawdown        | 72.4%    | 28.6%       | 18.3%      |
| 95% VaR             | 0.045    | 0.018       | 0.012      |

### Volatility Comparison by Period

| Period           | Unhedged Std | Naive Std | MV Std  | Improvement |
|------------------|--------------|-----------|---------|-------------|
| Full Period      | 85.3%        | 32.1%     | 24.6%   | 71.2%       |
| 2022 (Bear/FTX)  | 92.1%        | 38.4%     | 28.9%   | 68.6%       |
| 2023 (Recovery)  | 68.4%        | 26.8%     | 19.2%   | 71.9%       |
| 2024 (ETF Bull)  | 78.2%        | 30.5%     | 22.1%   | 71.7%       |

---

## 🔧 Module Details

### `src/data.py` - Data Module

```python
from src.data import load_all_data

# Load all data for backtesting
df = load_all_data("2022-01-01", "2025-12-03")
# Returns DataFrame with: spot, futures, dvol, net_delta, basis
```

**Data Sources**:
- BTC Spot: `yfinance` (BTC-USD)
- BTC Futures: `yfinance` (BTC=F) or synthetic
- DVOL: Simulated from realized volatility
- Net Delta: Simulated for 10% OTM strangle

### `src/covariance.py` - EWMA Covariance

```python
from src.covariance import EWMACovariance, compute_ewma_covariance_series

# Create EWMA estimator
ewma = EWMACovariance(lambda_=0.94, n_assets=3)

# Compute full time series
cov_series = compute_ewma_covariance_series(returns_df)
```

**From Lecture 4**: EWMA assigns exponentially declining weights:
- λ = 0.94 → half-life ≈ 11 days
- More responsive to recent volatility changes

### `src/optimize.py` - Optimization

```python
from src.optimize import optimize_hedge_portfolio, compute_efficient_frontier_with_delta

# Find optimal hedge weights
weights, diagnostics = optimize_hedge_portfolio(
    cov_matrix=cov,
    net_delta=-0.05,
    expected_returns=mu
)

# Compute efficient frontier
frontier = compute_efficient_frontier_with_delta(cov, mu, net_delta=-0.05)
```

### `src/backtest.py` - Backtesting

```python
from src.backtest import BacktestEngine, run_full_backtest

# Full backtest with one line
engine, summary = run_full_backtest("2022-01-01", "2025-12-03")

# Or step by step
engine = BacktestEngine(start_date, end_date, notional=100000)
engine.load_data()
engine.run_backtest()
metrics = engine.calculate_metrics()
```

### `src/robo_advisor.py` - Robo-Advisor

```python
from src.robo_advisor import robo_advisor_interface, compare_profiles

# Get recommendation by risk profile
result = robo_advisor_interface(risk_profile='moderate')
print(result['recommendation'])

# Or by custom risk aversion
result = robo_advisor_interface(risk_aversion=5.0)

# Compare all profiles
comparison = compare_profiles()
```

**From Lecture 4**: Quadratic utility maximization:
```
U(w) = μ'w - (A/2) * w'Σw
```
Where A is the risk aversion coefficient.

---

## 📈 Visualizations

The following plots are generated in `figures/`:

1. **`cumulative_pnl.png`**: Cumulative P&L comparison
2. **`pnl_distribution.png`**: Daily P&L distributions
3. **`weight_evolution.png`**: Portfolio weights over time
4. **`rolling_volatility.png`**: 30-day rolling volatility
5. **`drawdowns.png`**: Drawdown analysis
6. **`efficient_frontier.png`**: Efficient frontier with current position
7. **`btc_context.png`**: BTC price with key events
8. **`metrics_comparison.png`**: Bar charts of key metrics

---

## 🧮 Mathematical Framework

### Returns Calculation (Lecture 2)

- **Log returns**: `r_t = log(P_t / P_{t-1})`
- **Basis returns**: `r_basis = log(basis_t / basis_{t-1})`
- **DVOL change**: `Δσ = (DVOL_t - DVOL_{t-1}) / DVOL_{t-1}`

### EWMA Covariance (Lecture 4)

Update formula:
```
Σ_t = λ * Σ_{t-1} + (1-λ) * r_{t-1} * r'_{t-1}
```

### Strangle P&L Approximation (Lecture 3)

Taylor expansion:
```
P&L ≈ θ*dt + (1/2)*Γ*(ΔS)² + Δ*ΔS + ν*Δσ
```

Where:
- θ > 0 (time decay profit)
- Γ < 0 (gamma loss)
- Δ ≈ 0 (small for OTM)
- ν < 0 (vega loss when vol rises)

---

## ⚠️ Limitations & Assumptions

1. **Simulated Data**: DVOL and options Greeks are simulated; real Deribit data would improve accuracy
2. **Transaction Costs**: Not included; would reduce returns
3. **Slippage**: Market impact not modeled
4. **Roll Costs**: Quarterly futures roll not explicitly modeled
5. **Margin**: No margin requirements considered
6. **Liquidity**: Assumes infinite liquidity at mid prices

---

## 📚 References

### Course Materials
- Lecture 2: Asset Returns and Risk Measures
- Lecture 3: Option Pricing and Greeks
- Lecture 4: Mean-Variance Analysis (Markowitz)

### Academic References
- Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
- J.P. Morgan (1996). "RiskMetrics Technical Document."
- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities."

### Data Sources
- Yahoo Finance: BTC-USD, BTC=F
- Deribit: DVOL methodology

---

## 👥 Authors

HKUST Financial Engineering Students  
IEDA3330 Introduction to Financial Engineering  
Fall 2025

---

## 📄 License

This project is for educational purposes as part of HKUST IEDA3330.
MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Prof. Wei JIANG for course instruction and project guidance
- Deribit for BTC options market structure reference
- Open-source libraries: pandas, numpy, cvxpy, yfinance, matplotlib

