"""
METHODOLOGY 3: Mean-Variance Optimal Portfolio Construction
===========================================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Based on Lecture 5: Capital Asset Pricing Model

This module implements:
1. Markowitz Mean-Variance Optimization Framework
   - Minimum variance portfolio
   - Risk-free asset (USDT cash)

2. Delta-Neutrality Constraint
   - Asset deltas @ weights == -net_delta
   - Ensures hedge portfolio neutralizes strangle delta

3. Optimization Problem:
   minimize    w' Σ w           (portfolio variance)
   subject to  Σ w_i = 1        (fully invested)
               δ' w = -net_δ    (delta neutral)
               w_i ≥ 0          (long only, optional)

Uses EWMA covariance from Methodology 2 as input.

References:
- Markowitz, H. (1952) "Portfolio Selection"
- Lecture 5: Capital Asset Pricing Model
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, Dict, Optional, List


# ============================================================================
# RISK-FREE RATE (USDT/Cash)
# ============================================================================

RISK_FREE_RATE_ANNUAL = 0.05  # 5% annual risk-free rate
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 365


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_positive_definite(cov_matrix: np.ndarray,
                              min_eigenvalue: float = 1e-6) -> np.ndarray:
    """
    Ensure covariance matrix is positive definite.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Input covariance matrix (may be near-singular)
    min_eigenvalue : float
        Minimum eigenvalue to enforce
    
    Returns
    -------
    np.ndarray
        Positive definite covariance matrix
    """
    # Check symmetry
    matrix = (cov_matrix + cov_matrix.T) / 2
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Floor small/negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure symmetry
    cov_fixed = (cov_fixed + cov_fixed.T) / 2
    
    return cov_fixed


# ============================================================================
# MEAN-VARIANCE OPTIMIZATION
# ============================================================================

def optimize_minimum_variance_portfolio(
    cov_matrix: np.ndarray,
    net_delta: float,
    asset_deltas: np.ndarray = None,
    long_only: bool = True
) -> Dict[str, any]:
    """
    Find minimum variance portfolio subject to delta-neutrality constraint.
    
    This is the core of Methodology 3: Mean-Variance Optimal Construction.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (n x n) from Methodology 2 (EWMA)
    net_delta : float
        Net delta of strangle position (from Methodology 1)
    asset_deltas : np.ndarray
        Delta of each hedge asset [delta_spot, delta_futures, delta_cash]
        Default: [1.0, 1.0, 0.0]
    long_only : bool
        Whether to enforce long-only constraint
    
    Returns
    -------
    dict
        Dictionary with:
        - weights: Optimal portfolio weights
        - variance: Portfolio variance
        - volatility: Portfolio volatility (std dev)
        - status: Optimization status
    
    Notes
    -----
    Optimization Problem (from Lecture 5):
    
        minimize    w' Σ w           
        subject to  Σ w_i = 1        (budget constraint)
                    δ' w = -net_δ    (delta neutrality)
                    w_i ≥ 0          (long only, if enabled)
    
    The delta-neutrality constraint ensures:
    - Hedge portfolio delta offsets strangle delta
    - Combined position is delta-neutral
    
    Example
    -------
    >>> cov = np.array([[0.04, 0.02, 0], [0.02, 0.03, 0], [0, 0, 0.0001]])
    >>> result = optimize_minimum_variance_portfolio(cov, net_delta=-0.05)
    >>> print(f"Optimal weights: {result['weights']}")
    """
    n_assets = cov_matrix.shape[0]
    
    # Default asset deltas: [spot=1, futures=1, cash=0]
    if asset_deltas is None:
        asset_deltas = np.array([1.0, 1.0, 0.0])
    
    # Ensure positive definite covariance
    cov_matrix = _ensure_positive_definite(cov_matrix)
    
    # =========================================================================
    # CVXPY OPTIMIZATION (Markowitz Framework)
    # =========================================================================
    
    # Decision variable: portfolio weights
    w = cp.Variable(n_assets)
    
    # Objective: minimize portfolio variance
    # Var(portfolio) = w' Σ w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    objective = cp.Minimize(portfolio_variance)
    
    # Constraints
    constraints = [
        # Budget constraint: fully invested
        cp.sum(w) == 1,
        
        # Delta-neutrality: hedge delta = -strangle delta
        # δ' w = -net_δ
        asset_deltas @ w == -net_delta
    ]
    
    # Long-only constraint (optional)
    if long_only:
        constraints.append(w >= 0)

    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status == cp.OPTIMAL:
            weights = w.value
            variance = problem.value
            
            return {
                'weights': weights,
                'variance': variance,
                'volatility': np.sqrt(variance),
                'status': 'optimal',
                'w_spot': weights[0],
                'w_futures': weights[1],
                'w_cash': weights[2]
            }
        else:
            # Fallback to naive hedge
            # return _naive_hedge_fallback(net_delta)
            return _smart_fallback(net_delta, cov_matrix)
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        # return _naive_hedge_fallback(net_delta)
        return _smart_fallback(net_delta, cov_matrix)


# def _naive_hedge_fallback(net_delta: float) -> Dict[str, any]:
#     """Fallback to naive 1:1 futures hedge when optimization fails."""
#     return {
#         'weights': np.array([0.0, -net_delta, 1.0 + net_delta]),
#         'variance': np.nan,
#         'volatility': np.nan,
#         'status': 'fallback',
#         'w_spot': 0.0,
#         'w_futures': -net_delta,
#         'w_cash': 1.0 + net_delta
#     }

def _smart_fallback(net_delta: float, 
                    cov_matrix: np.ndarray) -> Dict:
    """
    Smart fallback that considers both delta and risk.
    """
    # Calculate hedge ratio from minimum variance hedge formula
    # h* = Cov(S,F) / Var(F)
    cov_spot_futures = cov_matrix[0, 1]
    var_futures = cov_matrix[1, 1]
    
    hedge_ratio = cov_spot_futures / (var_futures + 1e-10)
    
    # Combine delta hedge with MV hedge ratio
    alpha = 0.5  # Weight between delta (1.0) and MV hedge
    optimal_futures = alpha * (-net_delta) + (1 - alpha) * hedge_ratio * (-net_delta)
    
    w_futures = np.clip(optimal_futures, -0.5, 1.0)
    w_cash = 1.0 - w_futures
    w_spot = 0.0
    
    weights = np.array([w_spot, w_futures, w_cash])
    
    return {
        'weights': weights,
        'variance': weights @ cov_matrix @ weights,
        'volatility': np.sqrt(weights @ cov_matrix @ weights),
        'status': 'smart_fallback',
        'w_spot': w_spot,
        'w_futures': w_futures,
        'w_cash': w_cash
    }


def _ensure_positive_definite(matrix: np.ndarray, 
                               min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Ensure matrix is positive definite."""
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# ============================================================================
# NAIVE DELTA HEDGE (BASELINE COMPARISON)
# ============================================================================

def optimize_hedge_portfolio(cov_matrix: np.ndarray,
                             net_delta: float,
                             concentration_limit: float = 0.333, # max 33% in single asset
                             min_liquid_weight = 0.1,  # Keep at least 10% in cash for liquidity
                             lambda_reg: float = 1e-4,
                             target_sharpe: float = None, # target sharpe ratio
                             expected_returns: np.ndarray = None,
                             risk_aversion: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    Optimize hedge portfolio for short strangle (Methodology 3).
    
    This is the core function used by the backtest.
    
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
    
    From Lecture 5: With risk-free asset, optimal portfolio
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

    # penalize extreme positions
    regularization = lambda_reg * cp.sum_squares(w)
    
    # Objective: min variance - λ * expected return (if risk_aversion > 0)
    portfolio_variance = cp.quad_form(w, cov_matrix)
    portfolio_return = expected_returns @ w
    
    # if risk_aversion > 0:
    #     # Mean-variance utility: max(μ - (A/2)*σ²)
    #     # Equivalent to: min((A/2)*σ² - μ)
    #     objective = cp.Minimize(risk_aversion/2 * portfolio_variance - portfolio_return)
    # else:
    #     # Pure minimum variance
    #     objective = cp.Minimize(portfolio_variance)
    if risk_aversion > 0:
        # Mean-variance utility with regularization
        # Objective: min((A/2)*σ² - μ + λ_reg * ||w||²)
        objective = cp.Minimize(risk_aversion/2 * portfolio_variance - portfolio_return + regularization)
    else:
        # Pure minimum variance with regularization
        # Objective: min(σ² + λ_reg * ||w||²)
        objective = cp.Minimize(portfolio_variance + regularization)
    

    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Fully invested
        w >= -0.5,       # Allow small shorts for flexibility
        w <= 1.5,        # Cap leverage
        asset_deltas @ w == -net_delta,  # Delta-neutral
        # Additional Constraints: 
        w <= concentration_limit, # concentration limits 
    ]

    # maximum risk constraint (optional)
    if target_sharpe is not None and expected_returns is not None:
        max_volatility = expected_returns / target_sharpe
        max_var = max_volatility**2
        constraints.append(cp.quad_form (w, cov_matrix) <= max_var) # maximum risk constraint
    
    constraints.append(w[2] >= min_liquid_weight)

    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status == 'optimal':
            weights = w.value
            
            # Diagnostics
            diagnostics = {
                'status': 'optimal',
                'variance': float(problem.value),
                'volatility': float(np.sqrt(problem.value)),
                'expected_return': float(expected_returns @ weights),
                'sharpe': float((expected_returns @ weights - RISK_FREE_RATE_DAILY) / 
                                (np.sqrt(problem.value) + 1e-10))
            }
            
            return weights, diagnostics
        else:
            # Fallback: simple hedge (all in futures to match delta)
            fallback_weights = np.array([0.0, -net_delta, 1 + net_delta])
            fallback_weights = np.clip(fallback_weights, 0, 1)
            fallback_weights /= fallback_weights.sum()
            
            return fallback_weights, {'status': 'fallback'}
            
    except Exception as e:
        # Fallback on error
        fallback_weights = np.array([0.0, -net_delta, 1 + net_delta])
        fallback_weights = np.clip(fallback_weights, 0, 1)
        fallback_weights /= fallback_weights.sum()
        
        return fallback_weights, {'status': 'error', 'message': str(e)}

def compute_naive_delta_hedge(net_delta: float) -> Dict[str, float]:
    """
    Compute simple 1:1 delta hedge using futures only.
    
    This is the baseline comparison for the MV optimal hedge.
    
    Parameters
    ----------
    net_delta : float
        Net delta of strangle position
    
    Returns
    -------
    dict
        Hedge weights (all in futures)
    
    Notes
    -----
    Naive hedge: 
    - Hold -net_delta in futures to neutralize delta
    - Remaining capital in cash
    - Ignores covariance structure (unlike MV optimal)
    """
    return {
        'w_spot': 0.0,
        'w_futures': -net_delta,  # Opposite of strangle delta
        'w_cash': 1.0 + net_delta,
        'hedge_ratio': -net_delta
    }


# ============================================================================
# EXPECTED RETURNS (CAPM)
# ============================================================================

def compute_expected_returns(
    market_return: float = 0.10,  # 10% annual market return
    betas: np.ndarray = None,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL
) -> np.ndarray:
    """
    Compute expected returns using CAPM.
    
    Parameters
    ----------
    market_return : float
        Expected market return (annual)
    betas : np.ndarray
        Asset betas [beta_spot, beta_futures, beta_cash]
    risk_free_rate : float
        Risk-free rate
    
    Returns
    -------
    np.ndarray
        Expected returns for each asset (daily)
    
    Notes
    -----
    CAPM (from Lecture 5):
        E[r_i] = r_f + β_i * (E[r_m] - r_f)
    """
    if betas is None:
        # Default betas: BTC has high beta, futures similar, cash zero
        betas = np.array([1.5, 1.5, 0.0])
    
    # Market risk premium
    market_premium = market_return - risk_free_rate
    
    # CAPM expected returns (annual)
    expected_annual = risk_free_rate + betas * market_premium
    
    # Convert to daily
    expected_daily = expected_annual / 365
    
    return expected_daily


# ============================================================================
# EFFICIENT FRONTIER
# ============================================================================

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


def compute_efficient_frontier(
    cov_matrix: np.ndarray,
    expected_returns: np.ndarray = None,
    n_points: int = 50,
    net_delta: float = 0.0
) -> pd.DataFrame:
    """
    Compute efficient frontier with delta-neutrality constraint.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    expected_returns : np.ndarray
        Expected returns for each asset
    n_points : int
        Number of points on frontier
    net_delta : float
        Net delta to hedge
    
    Returns
    -------
    pd.DataFrame
        Efficient frontier with return, volatility, weights
    """
    n_assets = cov_matrix.shape[0]
    
    if expected_returns is None:
        expected_returns = compute_expected_returns()
    
    # Asset deltas
    asset_deltas = np.array([1.0, 1.0, 0.0])
    
    # Ensure positive definite
    cov_matrix = _ensure_positive_definite(cov_matrix)
    
    # Find return range
    min_return = min(expected_returns)
    max_return = max(expected_returns)
    target_returns = np.linspace(min_return, max_return, n_points)
    
    frontier_data = []
    
    for target_ret in target_returns:
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            expected_returns @ w >= target_ret,
            asset_deltas @ w == -net_delta,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                variance = problem.value
                actual_return = expected_returns @ weights
                
                frontier_data.append({
                    'return': actual_return * 365,  # Annualize
                    'volatility': np.sqrt(variance * 252),  # Annualize
                    'w_spot': weights[0],
                    'w_futures': weights[1],
                    'w_cash': weights[2],
                    'sharpe': (actual_return * 365 - RISK_FREE_RATE_ANNUAL) / 
                              (np.sqrt(variance * 252) + 1e-10)
                })
        except:
            continue
    
    return pd.DataFrame(frontier_data)


# ============================================================================
# PORTFOLIO METRICS
# ============================================================================

def compute_portfolio_metrics(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    expected_returns: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute portfolio metrics for given weights.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    expected_returns : np.ndarray
        Expected returns
    
    Returns
    -------
    dict
        Portfolio metrics
    """
    if expected_returns is None:
        expected_returns = compute_expected_returns()
    
    # Portfolio variance and volatility
    variance = weights @ cov_matrix @ weights
    volatility = np.sqrt(variance)
    
    # Portfolio return
    portfolio_return = expected_returns @ weights
    
    # Sharpe ratio
    sharpe = (portfolio_return * 365 - RISK_FREE_RATE_ANNUAL) / (volatility * np.sqrt(252) + 1e-10)
    
    return {
        'expected_return_daily': portfolio_return,
        'expected_return_annual': portfolio_return * 365,
        'variance_daily': variance,
        'volatility_daily': volatility,
        'volatility_annual': volatility * np.sqrt(252),
        'sharpe_ratio': sharpe
    }


# ============================================================================
# HEDGE EFFECTIVENESS
# ============================================================================

def compute_hedge_effectiveness(
    unhedged_variance: float,
    hedged_variance: float
) -> float:
    """
    Compute hedge effectiveness ratio.
    
    Parameters
    ----------
    unhedged_variance : float
        Variance of unhedged position
    hedged_variance : float
        Variance of hedged position
    
    Returns
    -------
    float
        Hedge effectiveness (1 = perfect hedge, 0 = no improvement)
    
    Notes
    -----
    Hedge Effectiveness = 1 - (Var_hedged / Var_unhedged)
    
    Interpretation:
    - 1.0 = Perfect hedge (variance eliminated)
    - 0.9 = 90% variance reduction
    - 0.0 = No variance reduction
    """
    if unhedged_variance <= 0:
        return 0.0
    
    return 1 - (hedged_variance / unhedged_variance)


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("METHODOLOGY 3: Mean-Variance Optimal Portfolio Construction")
    print("="*70)
    print(f"""
    This module implements Markowitz mean-variance optimization from Lecture 5.
    
    OPTIMIZATION PROBLEM:
    
        minimize    w' Σ w           (portfolio variance)
        subject to  Σ w_i = 1        (budget constraint)
                    δ' w = -net_δ    (delta neutrality)
                    w_i ≥ 0          (long only)
    
    INPUTS:
    - Σ: Covariance matrix from Methodology 2 (EWMA)
    - net_δ: Net delta from Methodology 1 (Option Greeks)
    
    HEDGE ASSETS:
    - BTC Spot (δ = 1.0)
    - BTC Futures (δ = 1.0)
    - USDT Cash (δ = 0.0, risk-free)
    
    COMPARISON:
    - MV Optimal: Uses covariance structure for optimal weights
    - Naive Hedge: Simple 1:1 futures hedge (ignores covariance)
    
    Reference: Lecture 5 - Capital Asset Pricing Model
    """)
    
    # Demonstration
    print("\nDEMONSTRATION:")
    print("-" * 50)
    
    # Sample covariance matrix (from EWMA)
    cov_matrix = np.array([
        [0.0009, 0.00085, 0.00001],   # Spot variance, covariances
        [0.00085, 0.0008, 0.00001],   # Futures 
        [0.00001, 0.00001, 0.000001]  # Cash (very low variance)
    ])
    
    # Net delta from strangle
    net_delta = -0.05  # Slightly negative delta
    
    # Compute optimal hedge
    result = optimize_minimum_variance_portfolio(cov_matrix, net_delta)
    
    print(f"Strangle net delta: {net_delta:.4f}")
    print(f"\nMV OPTIMAL HEDGE:")
    print(f"  w_spot:    {result['w_spot']:.4f}")
    print(f"  w_futures: {result['w_futures']:.4f}")
    print(f"  w_cash:    {result['w_cash']:.4f}")
    print(f"  Portfolio volatility: {result['volatility']*np.sqrt(252)*100:.2f}% (annualized)")
    
    # Naive hedge for comparison
    naive = compute_naive_delta_hedge(net_delta)
    print(f"\nNAIVE HEDGE:")
    print(f"  w_spot:    {naive['w_spot']:.4f}")
    print(f"  w_futures: {naive['w_futures']:.4f}")
    print(f"  w_cash:    {naive['w_cash']:.4f}")
    
    # Compute naive hedge variance
    naive_weights = np.array([naive['w_spot'], naive['w_futures'], naive['w_cash']])
    naive_variance = naive_weights @ cov_matrix @ naive_weights
    
    print(f"\nHEDGE EFFECTIVENESS:")
    print(f"  MV Optimal variance: {result['variance']:.8f}")
    print(f"  Naive variance:      {naive_variance:.8f}")
    print(f"  Variance reduction:  {(1 - result['variance']/naive_variance)*100:.1f}%")

