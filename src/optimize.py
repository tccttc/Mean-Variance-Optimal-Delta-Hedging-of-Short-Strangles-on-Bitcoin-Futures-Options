"""
Mean-Variance Optimization Module
=================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module implements:
- Mean-variance optimization using cvxpy
- Delta-neutral portfolio construction
- Minimum variance hedge portfolio
- Efficient frontier computation

References:
- Lecture 4: Mean-Variance Analysis (Markowitz)
- Lecture 4: Two-fund separation with risk-free asset
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, List, Optional, Dict

from .data import RISK_FREE_RATE_DAILY, RISK_FREE_RATE_ANNUAL
from .covariance import ensure_positive_definite


# ============================================================================
# MINIMUM VARIANCE OPTIMIZATION
# ============================================================================

def minimize_variance(cov_matrix: np.ndarray,
                      delta_constraint: float = None,
                      long_only: bool = True,
                      asset_deltas: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    Find minimum variance portfolio using cvxpy.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of assets
    delta_constraint : float, optional
        Target portfolio delta (for delta-neutral hedging)
    long_only : bool
        Whether to enforce non-negative weights
    asset_deltas : np.ndarray, optional
        Delta exposure of each asset (default: [1, 1, 0] for spot, futures, cash)
    
    Returns
    -------
    Tuple[np.ndarray, float]
        Optimal weights and portfolio variance
    
    Notes
    -----
    From Lecture 4: Minimum Variance Portfolio
    
    Optimization problem:
    
    min  w'Σw
    s.t. Σw_i = 1          (fully invested)
         w_i >= 0          (long-only, if specified)
         Σw_i*δ_i = target (delta-neutral)
    
    The delta-neutral constraint ensures we hedge the
    strangle's delta exposure with the portfolio.
    
    For our hedging problem:
    - Asset 1 (spot): delta = 1
    - Asset 2 (futures): delta = 1
    - Asset 3 (USDT): delta = 0
    """
    n_assets = cov_matrix.shape[0]
    
    # Ensure positive definite
    cov_matrix = ensure_positive_definite(cov_matrix)
    
    # Default asset deltas
    if asset_deltas is None:
        asset_deltas = np.array([1.0, 1.0, 0.0])
    
    # Decision variable: portfolio weights
    w = cp.Variable(n_assets)
    
    # Objective: minimize portfolio variance
    # variance = w'Σw (quadratic form)
    portfolio_variance = cp.quad_form(w, cov_matrix)
    objective = cp.Minimize(portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1  # Fully invested
    ]
    
    # Long-only constraint
    if long_only:
        constraints.append(w >= 0)
    
    # Delta-neutral constraint
    if delta_constraint is not None:
        # Portfolio delta should equal negative of strangle delta
        # This hedges the position
        constraints.append(asset_deltas @ w == -delta_constraint)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status == 'optimal':
            return w.value, problem.value
        else:
            # Fallback: equal weight
            return np.ones(n_assets) / n_assets, np.nan
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        return np.ones(n_assets) / n_assets, np.nan


def optimize_hedge_portfolio(cov_matrix: np.ndarray,
                             net_delta: float,
                             expected_returns: np.ndarray = None,
                             risk_aversion: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    Optimize hedge portfolio for short strangle.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        3x3 covariance matrix for [spot, futures, cash]
    net_delta : float
        Net delta of short strangle (typically negative, ~-0.05)
    expected_returns : np.ndarray, optional
        Expected returns for mean-variance optimization
    risk_aversion : float
        Risk aversion parameter (0 = pure min variance)
    
    Returns
    -------
    Tuple[np.ndarray, Dict]
        Optimal weights and diagnostics dictionary
    
    Notes
    -----
    The optimization finds weights that:
    1. Hedge the strangle's delta exposure
    2. Minimize portfolio variance
    3. (Optionally) maximize expected return given variance
    
    For delta-neutral hedging:
    - If strangle has net_delta = -0.05
    - We need portfolio delta = +0.05 to offset
    - This means small long exposure to spot/futures
    
    From Lecture 4: With risk-free asset, optimal portfolio
    is combination of tangency portfolio and risk-free.
    """
    n_assets = 3  # spot, futures, cash
    
    # Ensure positive definite
    cov_matrix = ensure_positive_definite(cov_matrix)
    
    # Asset deltas: spot=1, futures=1, cash=0
    asset_deltas = np.array([1.0, 1.0, 0.0])
    
    # Default expected returns (CAPM-based)
    if expected_returns is None:
        expected_returns = np.array([
            RISK_FREE_RATE_DAILY + 0.10/365,  # Spot: rf + 10% premium
            RISK_FREE_RATE_DAILY + 0.08/365,  # Futures: rf + 8% premium
            RISK_FREE_RATE_DAILY              # Cash: rf
        ])
    
    # Decision variable
    w = cp.Variable(n_assets)
    
    # Objective: min variance - λ * expected return (if risk_aversion > 0)
    portfolio_variance = cp.quad_form(w, cov_matrix)
    portfolio_return = expected_returns @ w
    
    if risk_aversion > 0:
        # Mean-variance utility: max(μ - (A/2)*σ²)
        # Equivalent to: min((A/2)*σ² - μ)
        objective = cp.Minimize(risk_aversion/2 * portfolio_variance - portfolio_return)
    else:
        # Pure minimum variance
        objective = cp.Minimize(portfolio_variance)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Fully invested
        w >= -0.5,       # Allow small shorts for flexibility
        w <= 1.5,        # Cap leverage
        asset_deltas @ w == -net_delta  # Delta-neutral
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status == 'optimal':
            weights = w.value
            
            # Diagnostics
            diagnostics = {
                'status': problem.status,
                'variance': portfolio_variance.value,
                'volatility': np.sqrt(portfolio_variance.value),
                'expected_return': portfolio_return.value,
                'portfolio_delta': asset_deltas @ weights,
                'target_delta': -net_delta
            }
            
            return weights, diagnostics
        else:
            # Fallback: simple hedge (all in futures to match delta)
            fallback_weights = np.array([0.0, -net_delta, 1 + net_delta])
            fallback_weights = np.clip(fallback_weights, 0, 1)
            fallback_weights /= fallback_weights.sum()
            
            return fallback_weights, {'status': 'fallback'}
            
    except Exception as e:
        print(f"Optimization error: {e}")
        return np.array([0.0, 0.0, 1.0]), {'status': 'error', 'message': str(e)}


# ============================================================================
# EFFICIENT FRONTIER
# ============================================================================

def compute_efficient_frontier(cov_matrix: np.ndarray,
                               expected_returns: np.ndarray,
                               n_points: int = 50,
                               long_only: bool = True) -> pd.DataFrame:
    """
    Compute the efficient frontier.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    expected_returns : np.ndarray
        Expected returns vector
    n_points : int
        Number of points on frontier
    long_only : bool
        Whether to enforce long-only constraint
    
    Returns
    -------
    pd.DataFrame
        Efficient frontier with columns ['return', 'volatility', 'sharpe']
    
    Notes
    -----
    From Lecture 4: The efficient frontier is the set of portfolios
    that maximize return for each level of risk.
    
    Algorithm:
    1. Find minimum variance portfolio (left endpoint)
    2. Find maximum return portfolio (right endpoint)
    3. Interpolate target returns between endpoints
    4. For each target return, minimize variance
    """
    n_assets = cov_matrix.shape[0]
    
    # Ensure positive definite
    cov_matrix = ensure_positive_definite(cov_matrix)
    
    # Find return range
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    
    # Target returns
    target_returns = np.linspace(min_return * 0.8, max_return * 1.2, n_points)
    
    frontier_data = []
    
    for target in target_returns:
        # Optimize for target return
        w = cp.Variable(n_assets)
        
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,
            expected_returns @ w >= target
        ]
        
        if long_only:
            constraints.append(w >= 0)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == 'optimal':
                ret = expected_returns @ w.value
                vol = np.sqrt(problem.value)
                sharpe = (ret - RISK_FREE_RATE_DAILY) / vol if vol > 0 else 0
                
                frontier_data.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': w.value
                })
        except:
            continue
    
    return pd.DataFrame(frontier_data)


def compute_efficient_frontier_with_delta(cov_matrix: np.ndarray,
                                          expected_returns: np.ndarray,
                                          net_delta: float,
                                          n_points: int = 30) -> pd.DataFrame:
    """
    Compute efficient frontier subject to delta-neutral constraint.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix for hedge assets
    expected_returns : np.ndarray
        Expected returns
    net_delta : float
        Strangle net delta to hedge
    n_points : int
        Number of frontier points
    
    Returns
    -------
    pd.DataFrame
        Constrained efficient frontier
    
    Notes
    -----
    This frontier shows the risk-return tradeoff available
    when we must maintain delta neutrality.
    
    The frontier will be "inside" the unconstrained frontier,
    showing the cost of the hedging constraint.
    """
    n_assets = cov_matrix.shape[0]
    asset_deltas = np.array([1.0, 1.0, 0.0])
    
    cov_matrix = ensure_positive_definite(cov_matrix)
    
    # Risk aversion range
    risk_aversions = np.linspace(0.01, 100, n_points)
    
    frontier_data = []
    
    for A in risk_aversions:
        w = cp.Variable(n_assets)
        
        portfolio_variance = cp.quad_form(w, cov_matrix)
        portfolio_return = expected_returns @ w
        
        # Mean-variance utility
        objective = cp.Maximize(portfolio_return - A/2 * portfolio_variance)
        
        constraints = [
            cp.sum(w) == 1,
            w >= -0.5,
            w <= 1.5,
            asset_deltas @ w == -net_delta
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == 'optimal':
                ret = expected_returns @ w.value
                vol = np.sqrt(portfolio_variance.value)
                
                frontier_data.append({
                    'risk_aversion': A,
                    'return': ret,
                    'volatility': vol,
                    'sharpe': (ret - RISK_FREE_RATE_DAILY) / vol if vol > 0 else 0,
                    'weights': w.value,
                    'w_spot': w.value[0],
                    'w_futures': w.value[1],
                    'w_cash': w.value[2]
                })
        except:
            continue
    
    return pd.DataFrame(frontier_data)


# ============================================================================
# NAIVE HEDGE BENCHMARK
# ============================================================================

def compute_naive_hedge(net_delta: float) -> np.ndarray:
    """
    Compute naive 1:1 futures hedge weights.
    
    Parameters
    ----------
    net_delta : float
        Net delta of short strangle
    
    Returns
    -------
    np.ndarray
        Hedge weights [spot, futures, cash]
    
    Notes
    -----
    Naive hedge: Use futures to exactly offset delta exposure.
    
    If strangle delta = -0.05:
    - Need to go long 0.05 units of futures
    - Rest in cash
    
    This is the simplest delta-neutral strategy,
    but doesn't optimize for variance.
    """
    # Use futures only for delta hedge
    futures_weight = -net_delta  # Long if strangle is short delta
    cash_weight = 1 - futures_weight
    
    # Ensure non-negative weights
    if futures_weight < 0:
        futures_weight = 0
        cash_weight = 1
    elif futures_weight > 1:
        futures_weight = 1
        cash_weight = 0
    
    return np.array([0.0, futures_weight, cash_weight])


# ============================================================================
# OPTIMIZATION UTILITIES
# ============================================================================

def check_feasibility(cov_matrix: np.ndarray,
                      net_delta: float) -> bool:
    """
    Check if delta-neutral optimization is feasible.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    net_delta : float
        Target delta to hedge
    
    Returns
    -------
    bool
        True if problem is feasible
    
    Notes
    -----
    Feasibility requires:
    1. Valid covariance matrix (positive semi-definite)
    2. Delta constraint achievable with available assets
    """
    # Check covariance
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.min(eigenvalues) < -1e-8:
        return False
    
    # Check delta achievability
    # With spot (δ=1) and futures (δ=1), we can achieve any delta in [0, 1]
    # With cash (δ=0), we can extend to [-0.5, 1.5] with constraints
    target_delta = -net_delta
    if target_delta < -0.5 or target_delta > 1.5:
        return False
    
    return True


def compute_hedge_effectiveness(mv_weights: np.ndarray,
                                naive_weights: np.ndarray,
                                cov_matrix: np.ndarray) -> Dict:
    """
    Compare MV optimal hedge to naive hedge.
    
    Parameters
    ----------
    mv_weights : np.ndarray
        Mean-variance optimal weights
    naive_weights : np.ndarray
        Naive 1:1 hedge weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns
    -------
    Dict
        Comparison metrics
    
    Notes
    -----
    Hedge effectiveness measures:
    1. Variance reduction vs unhedged
    2. Variance reduction vs naive hedge
    3. Information ratio of MV vs naive
    """
    # Compute variances
    mv_var = mv_weights @ cov_matrix @ mv_weights
    naive_var = naive_weights @ cov_matrix @ naive_weights
    
    # Assume unhedged is 100% spot
    unhedged_var = cov_matrix[0, 0]
    
    return {
        'mv_variance': mv_var,
        'naive_variance': naive_var,
        'unhedged_variance': unhedged_var,
        'mv_vol': np.sqrt(mv_var),
        'naive_vol': np.sqrt(naive_var),
        'unhedged_vol': np.sqrt(unhedged_var),
        'var_reduction_vs_unhedged': (unhedged_var - mv_var) / unhedged_var * 100,
        'var_reduction_vs_naive': (naive_var - mv_var) / naive_var * 100 if naive_var > 0 else 0
    }


# ============================================================================
# BATCH OPTIMIZATION
# ============================================================================

def optimize_all_periods(cov_series: List[np.ndarray],
                         delta_series: pd.Series,
                         expected_returns: np.ndarray = None) -> pd.DataFrame:
    """
    Run optimization for all time periods.
    
    Parameters
    ----------
    cov_series : List[np.ndarray]
        Time series of covariance matrices
    delta_series : pd.Series
        Time series of strangle deltas
    expected_returns : np.ndarray, optional
        Expected returns (constant or time-varying)
    
    Returns
    -------
    pd.DataFrame
        Optimal weights for each period
    
    Notes
    -----
    This is the core function for backtesting:
    - For each day t, use Σ_t and δ_t to find optimal weights
    - Weights are used for next period's hedge
    """
    n_periods = len(cov_series)
    
    results = []
    
    for t in range(n_periods):
        cov = cov_series[t]
        delta = delta_series.iloc[t] if t < len(delta_series) else -0.05
        
        # MV optimal hedge
        mv_weights, diagnostics = optimize_hedge_portfolio(cov, delta, expected_returns)
        
        # Naive hedge
        naive_weights = compute_naive_hedge(delta)
        
        results.append({
            'date': delta_series.index[t] if t < len(delta_series) else None,
            'net_delta': delta,
            'mv_w_spot': mv_weights[0],
            'mv_w_futures': mv_weights[1],
            'mv_w_cash': mv_weights[2],
            'naive_w_spot': naive_weights[0],
            'naive_w_futures': naive_weights[1],
            'naive_w_cash': naive_weights[2],
            'mv_variance': diagnostics.get('variance', np.nan),
            'status': diagnostics.get('status', 'unknown')
        })
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    from .data import load_all_data
    from .returns import calculate_hedge_asset_returns
    from .covariance import compute_hedge_asset_covariance
    
    # Load data
    df = load_all_data("2022-01-01", "2025-12-03")
    
    # Get returns and covariance
    asset_returns = calculate_hedge_asset_returns(df)
    cov_series = compute_hedge_asset_covariance(asset_returns)
    
    # Use last covariance matrix for test
    cov = cov_series[-1]
    net_delta = -0.05  # Typical short strangle delta
    
    print("=" * 60)
    print("MEAN-VARIANCE OPTIMIZATION TEST")
    print("=" * 60)
    
    # Test minimum variance
    print("\n1. Minimum Variance (no constraints):")
    weights, var = minimize_variance(cov)
    print(f"   Weights: Spot={weights[0]:.3f}, Futures={weights[1]:.3f}, Cash={weights[2]:.3f}")
    print(f"   Variance: {var:.8f}")
    
    # Test delta-neutral optimization
    print("\n2. Delta-Neutral Optimization:")
    mv_weights, diagnostics = optimize_hedge_portfolio(cov, net_delta)
    print(f"   Target delta to hedge: {-net_delta:.3f}")
    print(f"   MV Weights: Spot={mv_weights[0]:.3f}, Futures={mv_weights[1]:.3f}, Cash={mv_weights[2]:.3f}")
    print(f"   Portfolio delta: {diagnostics.get('portfolio_delta', 'N/A'):.4f}")
    print(f"   Status: {diagnostics.get('status', 'N/A')}")
    
    # Compare to naive hedge
    print("\n3. Naive 1:1 Hedge:")
    naive_weights = compute_naive_hedge(net_delta)
    print(f"   Naive Weights: Spot={naive_weights[0]:.3f}, Futures={naive_weights[1]:.3f}, Cash={naive_weights[2]:.3f}")
    
    # Hedge effectiveness
    print("\n4. Hedge Effectiveness:")
    effectiveness = compute_hedge_effectiveness(mv_weights, naive_weights, cov)
    print(f"   MV Volatility: {effectiveness['mv_vol']*np.sqrt(365):.2%} (annualized)")
    print(f"   Naive Volatility: {effectiveness['naive_vol']*np.sqrt(365):.2%} (annualized)")
    print(f"   Variance reduction vs naive: {effectiveness['var_reduction_vs_naive']:.1f}%")
    
    # Efficient frontier
    print("\n5. Efficient Frontier (first 5 points):")
    expected_returns = np.array([0.0003, 0.00025, RISK_FREE_RATE_DAILY])
    frontier = compute_efficient_frontier_with_delta(cov, expected_returns, net_delta, n_points=10)
    print(frontier[['risk_aversion', 'return', 'volatility', 'sharpe']].head())

