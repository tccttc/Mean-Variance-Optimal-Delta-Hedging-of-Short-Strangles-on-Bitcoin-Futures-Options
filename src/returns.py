"""
Returns Calculation Module
==========================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module calculates:
- Log returns for spot prices
- Basis returns (futures-spot relationship)
- DVOL percentage changes
- Expected returns via CAPM

References:
- Lecture 2: Log returns vs simple returns
- Lecture 4: Mean-Variance Analysis and CAPM
"""

import pandas as pd
import numpy as np
from typing import Tuple

from .data import RISK_FREE_RATE_ANNUAL, RISK_FREE_RATE_DAILY


# ============================================================================
# LOG RETURNS
# ============================================================================

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from a price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    
    Returns
    -------
    pd.Series
        Log returns: r_t = log(P_t / P_{t-1})
    
    Notes
    -----
    From Lecture 2: Log returns are preferred for:
    1. Time additivity: r_{t,T} = r_{t,s} + r_{s,T}
    2. Better approximation for small returns
    3. Normal distribution assumption more appropriate
    
    For BTC with high volatility, log returns are essential
    to avoid the bias from simple returns.
    """
    return np.log(prices / prices.shift(1))


# ============================================================================
# RETURNS CALCULATION
# ============================================================================

def calculate_all_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all returns needed for covariance estimation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: ['spot', 'futures', 'dvol', 'basis']
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - r_spot: log return of BTC spot
        - r_basis: log change in basis
        - dvol_chg: percentage change in DVOL
    
    Notes
    -----
    The three factors capture different risk dimensions:
    
    1. r_spot: Directional BTC exposure
       - Main driver of P&L for delta-hedged positions
       - High volatility (~60-100% annualized)
    
    2. r_basis: Futures-spot convergence risk
       - Affects futures hedging effectiveness
       - Increases near futures expiry
    
    3. dvol_chg: Volatility regime changes
       - Vega exposure for short strangles
       - Typically negative correlation with spot
    
    From Lecture 4: Factor model approach to risk decomposition
    """
    returns_df = pd.DataFrame(index=df.index)
    
    # 1. Spot log returns
    # r_spot = log(Spot_t / Spot_{t-1})
    returns_df['r_spot'] = calculate_log_returns(df['spot'])
    
    # 2. Basis returns
    # Basis = (Futures - Spot) / Spot
    # r_basis = log(Basis_t / Basis_{t-1})
    # Handle potential negative basis (backwardation)
    basis = df['basis']
    
    # Add small constant to avoid log of negative numbers
    basis_shifted = basis + 0.10  # Shift to ensure positivity
    returns_df['r_basis'] = calculate_log_returns(basis_shifted)
    
    # 3. DVOL percentage change
    # dvol_chg = (DVOL_t - DVOL_{t-1}) / DVOL_{t-1}
    returns_df['dvol_chg'] = df['dvol'].pct_change()
    
    # Drop first row (NaN from returns calculation)
    returns_df = returns_df.iloc[1:]
    
    return returns_df


# ============================================================================
# EXPECTED RETURNS (CAPM)
# ============================================================================

def estimate_expected_returns(returns_df: pd.DataFrame,
                              market_premium: float = 0.10) -> pd.Series:
    """
    Estimate expected returns using simple CAPM.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with return columns
    market_premium : float
        Expected BTC market premium (default 10% annualized)
    
    Returns
    -------
    pd.Series
        Expected daily returns for each factor
    
    Notes
    -----
    From Lecture 4: CAPM relates expected returns to systematic risk
    
    μ_i = r_f + β_i * (μ_market - r_f)
    
    For our factors:
    - r_spot: β = 1 (it IS the market)
    - r_basis: β ≈ 0.1 (low correlation with spot direction)
    - dvol_chg: β ≈ -0.3 (negative correlation - leverage effect)
    
    These betas are estimated from historical correlations.
    """
    # Factor betas (estimated from BTC characteristics)
    betas = {
        'r_spot': 1.0,      # Spot IS the market
        'r_basis': 0.1,     # Basis weakly correlated with direction
        'dvol_chg': -0.3    # Vol tends to spike on down moves
    }
    
    # Daily risk-free and market premium
    rf_daily = RISK_FREE_RATE_DAILY
    market_premium_daily = market_premium / 365
    
    # CAPM expected returns
    expected_returns = {}
    for col in returns_df.columns:
        if col in betas:
            expected_returns[col] = rf_daily + betas[col] * market_premium_daily
        else:
            expected_returns[col] = rf_daily  # Default to risk-free
    
    return pd.Series(expected_returns)


# ============================================================================
# ASSET RETURNS FOR HEDGING
# ============================================================================

def calculate_hedge_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate returns for the three hedging assets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spot and futures prices
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - r_spot_asset: BTC spot returns
        - r_futures_asset: BTC futures returns
        - r_rf: Risk-free (USDT) returns
    
    Notes
    -----
    These returns are used for:
    1. Portfolio construction (hedge positions)
    2. P&L calculation
    3. Mean-variance optimization
    
    The risk-free asset (USDT) provides:
    - Stable store of value for margin
    - 5% annualized yield (DeFi staking proxy)
    """
    asset_returns = pd.DataFrame(index=df.index)
    
    # BTC spot returns
    asset_returns['r_spot_asset'] = calculate_log_returns(df['spot'])
    
    # BTC futures returns
    asset_returns['r_futures_asset'] = calculate_log_returns(df['futures'])
    
    # Risk-free returns (constant daily rate)
    asset_returns['r_rf'] = RISK_FREE_RATE_DAILY
    
    # Drop first row
    asset_returns = asset_returns.iloc[1:]
    
    return asset_returns


# ============================================================================
# STRANGLE P&L COMPONENTS
# ============================================================================

def calculate_strangle_pnl_components(df: pd.DataFrame,
                                       theta_daily: float = 0.002,
                                       gamma: float = 0.0001,
                                       vega: float = 0.01) -> pd.DataFrame:
    """
    Calculate approximate P&L components for short strangle.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with spot and dvol data
    theta_daily : float
        Daily theta decay (positive = profit for short position)
    gamma : float
        Gamma exposure (negative for short options)
    vega : float
        Vega exposure (negative for short options)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with P&L components:
        - theta_pnl: Time decay profit
        - gamma_pnl: Gamma loss from price moves
        - vega_pnl: Vega P&L from vol changes
        - delta_pnl_unhedged: Unhedged delta P&L
    
    Notes
    -----
    Short strangle P&L approximation (Taylor expansion):
    
    P&L ≈ θ*dt + (1/2)*Γ*(ΔS)² + Δ*ΔS + ν*Δσ
    
    Where:
    - θ > 0 for short options (time decay is profit)
    - Γ < 0 for short options (convexity works against us)
    - Δ ≈ small for OTM strangle
    - ν < 0 for short options (vol up = loss)
    
    From Lecture 3: Greek letter approximations for P&L
    
    Typical values for 10% OTM short strangle (normalized):
    - Theta: 0.2% of premium daily
    - Gamma: -0.0001 per $1 move
    - Vega: -1% of premium per 1 vol point
    """
    pnl_df = pd.DataFrame(index=df.index)
    
    # Price changes
    spot_change = df['spot'].diff()
    spot_pct_change = df['spot'].pct_change()
    
    # Vol changes
    dvol_change = df['dvol'].diff()
    
    # 1. Theta P&L (positive for short options)
    # Scale by spot price to normalize
    pnl_df['theta_pnl'] = theta_daily * df['spot']
    
    # 2. Gamma P&L (negative for short options)
    # Gamma loss = -(1/2) * |Gamma| * (ΔS)²
    pnl_df['gamma_pnl'] = -0.5 * gamma * (spot_change ** 2)
    
    # 3. Vega P&L (negative for short options when vol increases)
    # Vega loss = -|Vega| * Δσ
    pnl_df['vega_pnl'] = -vega * dvol_change * df['spot'] / 100
    
    # 4. Delta P&L (unhedged)
    # Using the net_delta from data
    pnl_df['delta_pnl_unhedged'] = df['net_delta'] * spot_change
    
    # Drop first row (NaN from diff)
    pnl_df = pnl_df.iloc[1:]
    
    return pnl_df


# ============================================================================
# ANNUALIZATION HELPERS
# ============================================================================

def annualize_returns(daily_returns: pd.Series) -> float:
    """
    Annualize daily returns.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series
    
    Returns
    -------
    float
        Annualized return
    
    Notes
    -----
    Annualized return = Daily mean * 365
    (Using 365 for crypto which trades every day)
    """
    return daily_returns.mean() * 365


def annualize_volatility(daily_returns: pd.Series) -> float:
    """
    Annualize daily volatility.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series
    
    Returns
    -------
    float
        Annualized volatility
    
    Notes
    -----
    Annualized vol = Daily std * sqrt(365)
    
    From Lecture 2: Volatility scales with square root of time
    under random walk assumption.
    """
    return daily_returns.std() * np.sqrt(365)


def calculate_sharpe_ratio(daily_returns: pd.Series,
                           rf_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series
    rf_annual : float
        Annual risk-free rate
    
    Returns
    -------
    float
        Sharpe ratio
    
    Notes
    -----
    Sharpe = (Annualized Return - Risk-free Rate) / Annualized Volatility
    
    From Lecture 4: Sharpe ratio measures risk-adjusted performance
    """
    ann_return = annualize_returns(daily_returns)
    ann_vol = annualize_volatility(daily_returns)
    
    if ann_vol == 0:
        return 0.0
    
    return (ann_return - rf_annual) / ann_vol


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    from .data import load_all_data
    
    # Load data
    df = load_all_data("2022-01-01", "2025-12-03")
    
    # Calculate returns
    returns_df = calculate_all_returns(df)
    print("\nFactor Returns Summary:")
    print(returns_df.describe())
    
    # Calculate expected returns
    exp_returns = estimate_expected_returns(returns_df)
    print("\nExpected Daily Returns (CAPM):")
    print(exp_returns)
    
    # Calculate hedge asset returns
    asset_returns = calculate_hedge_asset_returns(df)
    print("\nHedge Asset Returns Summary:")
    print(asset_returns.describe())
    
    # Calculate strangle P&L components
    pnl_components = calculate_strangle_pnl_components(df)
    print("\nStrangle P&L Components Summary:")
    print(pnl_components.describe())
    
    # Annualized metrics
    print("\nAnnualized Metrics:")
    print(f"Spot Volatility: {annualize_volatility(returns_df['r_spot']):.2%}")
    print(f"Spot Sharpe: {calculate_sharpe_ratio(returns_df['r_spot']):.2f}")

