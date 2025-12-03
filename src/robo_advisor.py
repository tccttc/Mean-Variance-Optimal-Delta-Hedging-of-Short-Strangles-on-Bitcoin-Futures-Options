"""
Robo-Advisor Prototype Module
=============================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module implements:
- Robo-advisor function for personalized hedge recommendations
- Risk aversion-based portfolio optimization
- User-friendly output formatting

References:
- Lecture 4: Quadratic utility and risk aversion
- FinTech applications of portfolio optimization
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Tuple, Optional
from datetime import datetime

from .data import RISK_FREE_RATE_DAILY, RISK_FREE_RATE_ANNUAL
from .covariance import ensure_positive_definite


# ============================================================================
# RISK AVERSION PROFILES
# ============================================================================

RISK_PROFILES = {
    'conservative': {
        'risk_aversion': 10.0,
        'description': 'Conservative - Prioritizes stability over returns',
        'max_leverage': 1.0,
        'min_cash': 0.3
    },
    'moderate': {
        'risk_aversion': 5.0,
        'description': 'Moderate - Balanced approach to risk and return',
        'max_leverage': 1.2,
        'min_cash': 0.2
    },
    'aggressive': {
        'risk_aversion': 2.0,
        'description': 'Aggressive - Willing to accept higher volatility for returns',
        'max_leverage': 1.5,
        'min_cash': 0.1
    },
    'very_aggressive': {
        'risk_aversion': 1.0,
        'description': 'Very Aggressive - Maximum return focus',
        'max_leverage': 2.0,
        'min_cash': 0.0
    }
}


# ============================================================================
# ROBO-ADVISOR CORE FUNCTION
# ============================================================================

def get_optimal_hedge(risk_aversion: float,
                      cov_matrix: np.ndarray,
                      expected_returns: np.ndarray,
                      net_delta: float = -0.05,
                      constraints: Dict = None) -> Dict:
    """
    Robo-advisor function: Get optimal hedge given risk aversion.
    
    Parameters
    ----------
    risk_aversion : float
        Risk aversion coefficient (A > 0)
        - A = 1: Aggressive (high risk tolerance)
        - A = 5: Moderate
        - A = 10: Conservative (low risk tolerance)
    cov_matrix : np.ndarray
        3x3 covariance matrix for [spot, futures, cash]
    expected_returns : np.ndarray
        Expected returns vector
    net_delta : float
        Net delta of short strangle to hedge
    constraints : Dict, optional
        Additional constraints (max_leverage, min_cash)
    
    Returns
    -------
    Dict
        Optimal portfolio recommendation with weights and analytics
    
    Notes
    -----
    From Lecture 4: Quadratic Utility Maximization
    
    Investor utility: U(w) = μ'w - (A/2) * w'Σw
    
    Where:
    - μ = expected returns vector
    - Σ = covariance matrix
    - A = risk aversion coefficient
    - w = portfolio weights
    
    Higher A → investor penalizes variance more → safer portfolio
    Lower A → investor tolerates variance → riskier portfolio
    
    The optimal portfolio lies on the efficient frontier at the point
    where indifference curve (based on A) is tangent to frontier.
    
    For delta hedging:
    - We add constraint: portfolio delta = -strangle delta
    - This ensures delta-neutral position
    """
    # Input validation
    if risk_aversion <= 0:
        raise ValueError("Risk aversion must be positive")
    
    n_assets = 3
    asset_names = ['BTC Spot', 'BTC Futures', 'USDT (Cash)']
    asset_deltas = np.array([1.0, 1.0, 0.0])
    
    # Default constraints
    if constraints is None:
        constraints = {'max_leverage': 1.5, 'min_cash': 0.1}
    
    # Ensure covariance is positive definite
    cov_matrix = ensure_positive_definite(cov_matrix)
    
    # Decision variable
    w = cp.Variable(n_assets)
    
    # Objective: maximize utility = μ'w - (A/2)*w'Σw
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    
    utility = portfolio_return - (risk_aversion / 2) * portfolio_variance
    objective = cp.Maximize(utility)
    
    # Constraints
    constraint_list = [
        cp.sum(w) == 1,                            # Fully invested
        asset_deltas @ w == -net_delta,            # Delta-neutral
        w >= -0.5,                                 # Limited shorting
        w <= constraints.get('max_leverage', 1.5), # Leverage limit
        w[2] >= constraints.get('min_cash', 0.1)   # Minimum cash
    ]
    
    # Solve
    problem = cp.Problem(objective, constraint_list)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status == 'optimal':
            weights = w.value
            
            # Calculate metrics
            exp_return = expected_returns @ weights
            variance = weights @ cov_matrix @ weights
            volatility = np.sqrt(variance)
            
            # Annualize
            ann_return = exp_return * 365
            ann_vol = volatility * np.sqrt(365)
            
            # Sharpe ratio
            sharpe = (ann_return - RISK_FREE_RATE_ANNUAL) / ann_vol if ann_vol > 0 else 0
            
            # Portfolio delta (should be -net_delta)
            portfolio_delta = asset_deltas @ weights
            
            result = {
                'status': 'optimal',
                'risk_aversion': risk_aversion,
                'weights': {
                    'spot': round(weights[0] * 100, 2),
                    'futures': round(weights[1] * 100, 2),
                    'cash': round(weights[2] * 100, 2)
                },
                'metrics': {
                    'expected_return': round(ann_return * 100, 2),
                    'volatility': round(ann_vol * 100, 2),
                    'sharpe_ratio': round(sharpe, 3),
                    'portfolio_delta': round(portfolio_delta, 4),
                    'utility': round(utility.value, 6)
                },
                'recommendation': _generate_recommendation(weights, risk_aversion, ann_vol)
            }
            
        else:
            result = {
                'status': 'infeasible',
                'message': f"Optimization failed: {problem.status}",
                'weights': {'spot': 0, 'futures': 0, 'cash': 100},
                'recommendation': "Unable to find optimal hedge. Consider adjusting constraints."
            }
            
    except Exception as e:
        result = {
            'status': 'error',
            'message': str(e),
            'weights': {'spot': 0, 'futures': 0, 'cash': 100},
            'recommendation': f"Error in optimization: {e}"
        }
    
    return result


def _generate_recommendation(weights: np.ndarray, 
                            risk_aversion: float,
                            ann_vol: float) -> str:
    """Generate human-readable recommendation."""
    
    # Determine risk profile
    if risk_aversion >= 8:
        profile = "conservative"
    elif risk_aversion >= 4:
        profile = "moderate"
    elif risk_aversion >= 1.5:
        profile = "aggressive"
    else:
        profile = "very aggressive"
    
    recommendation = f"""
╔══════════════════════════════════════════════════════════════════╗
║              ROBO-ADVISOR HEDGE RECOMMENDATION                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Risk Profile: {profile.upper():^48} ║
║  Risk Aversion (A): {risk_aversion:^43.1f} ║
╠══════════════════════════════════════════════════════════════════╣
║  OPTIMAL HEDGE ALLOCATION:                                       ║
║    • BTC Spot:    {weights[0]*100:>6.1f}%                                       ║
║    • BTC Futures: {weights[1]*100:>6.1f}%                                       ║
║    • USDT (Cash): {weights[2]*100:>6.1f}%                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Expected Annual Volatility: {ann_vol*100:>5.1f}%                             ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return recommendation


# ============================================================================
# USER-FRIENDLY INTERFACE
# ============================================================================

def robo_advisor_interface(risk_aversion: float = None,
                           risk_profile: str = None,
                           cov_matrix: np.ndarray = None,
                           net_delta: float = -0.05) -> Dict:
    """
    User-friendly robo-advisor interface.
    
    Parameters
    ----------
    risk_aversion : float, optional
        Custom risk aversion (1-10 scale)
    risk_profile : str, optional
        Predefined profile: 'conservative', 'moderate', 'aggressive', 'very_aggressive'
    cov_matrix : np.ndarray, optional
        Custom covariance matrix (uses default if not provided)
    net_delta : float
        Strangle delta to hedge
    
    Returns
    -------
    Dict
        Complete recommendation
    
    Examples
    --------
    >>> # Using risk profile
    >>> result = robo_advisor_interface(risk_profile='moderate')
    >>> print(result['recommendation'])
    
    >>> # Using custom risk aversion
    >>> result = robo_advisor_interface(risk_aversion=7.5)
    >>> print(f"Optimal: {result['weights']}")
    """
    # Get risk aversion from profile if not specified
    if risk_aversion is None:
        if risk_profile is None:
            risk_profile = 'moderate'
        
        if risk_profile not in RISK_PROFILES:
            raise ValueError(f"Unknown profile. Choose from: {list(RISK_PROFILES.keys())}")
        
        risk_aversion = RISK_PROFILES[risk_profile]['risk_aversion']
        constraints = {
            'max_leverage': RISK_PROFILES[risk_profile]['max_leverage'],
            'min_cash': RISK_PROFILES[risk_profile]['min_cash']
        }
    else:
        # Use default constraints for custom risk aversion
        constraints = {'max_leverage': 1.5, 'min_cash': 0.1}
    
    # Default covariance if not provided (typical BTC daily covariance)
    if cov_matrix is None:
        # Approximate daily covariances based on historical BTC data
        # Spot vol ~60% annual → ~0.038 daily
        # Futures vol ~58% annual → ~0.037 daily
        # Correlation spot-futures ~0.98
        daily_spot_vol = 0.038
        daily_futures_vol = 0.037
        spot_futures_corr = 0.98
        
        cov_matrix = np.array([
            [daily_spot_vol**2, 
             spot_futures_corr * daily_spot_vol * daily_futures_vol, 
             0],
            [spot_futures_corr * daily_spot_vol * daily_futures_vol, 
             daily_futures_vol**2, 
             0],
            [0, 0, 1e-10]  # Near-zero for cash
        ])
    
    # Expected returns (CAPM-based approximation)
    expected_returns = np.array([
        RISK_FREE_RATE_DAILY + 0.10/365,   # Spot: rf + 10% risk premium
        RISK_FREE_RATE_DAILY + 0.08/365,   # Futures: rf + 8% premium (roll costs)
        RISK_FREE_RATE_DAILY               # Cash: rf
    ])
    
    # Get optimal hedge
    result = get_optimal_hedge(
        risk_aversion=risk_aversion,
        cov_matrix=cov_matrix,
        expected_returns=expected_returns,
        net_delta=net_delta,
        constraints=constraints
    )
    
    return result


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def analyze_risk_aversion_sensitivity(cov_matrix: np.ndarray = None,
                                       net_delta: float = -0.05,
                                       risk_aversions: list = None) -> pd.DataFrame:
    """
    Analyze how optimal allocation changes with risk aversion.
    
    Parameters
    ----------
    cov_matrix : np.ndarray, optional
        Covariance matrix
    net_delta : float
        Strangle delta
    risk_aversions : list, optional
        List of risk aversion values to test
    
    Returns
    -------
    pd.DataFrame
        Sensitivity analysis results
    """
    if risk_aversions is None:
        risk_aversions = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]
    
    results = []
    
    for A in risk_aversions:
        result = robo_advisor_interface(
            risk_aversion=A,
            cov_matrix=cov_matrix,
            net_delta=net_delta
        )
        
        if result['status'] == 'optimal':
            results.append({
                'Risk Aversion': A,
                'Spot %': result['weights']['spot'],
                'Futures %': result['weights']['futures'],
                'Cash %': result['weights']['cash'],
                'Expected Return %': result['metrics']['expected_return'],
                'Volatility %': result['metrics']['volatility'],
                'Sharpe': result['metrics']['sharpe_ratio']
            })
    
    return pd.DataFrame(results)


# ============================================================================
# PORTFOLIO COMPARISON
# ============================================================================

def compare_profiles(cov_matrix: np.ndarray = None,
                     net_delta: float = -0.05) -> pd.DataFrame:
    """
    Compare all predefined risk profiles.
    
    Parameters
    ----------
    cov_matrix : np.ndarray, optional
        Covariance matrix
    net_delta : float
        Strangle delta
    
    Returns
    -------
    pd.DataFrame
        Comparison of all profiles
    """
    results = []
    
    for profile_name in RISK_PROFILES.keys():
        result = robo_advisor_interface(
            risk_profile=profile_name,
            cov_matrix=cov_matrix,
            net_delta=net_delta
        )
        
        if result['status'] == 'optimal':
            results.append({
                'Profile': profile_name.title(),
                'Risk Aversion': result['risk_aversion'],
                'Spot': f"{result['weights']['spot']:.1f}%",
                'Futures': f"{result['weights']['futures']:.1f}%",
                'Cash': f"{result['weights']['cash']:.1f}%",
                'Vol': f"{result['metrics']['volatility']:.1f}%",
                'Sharpe': f"{result['metrics']['sharpe_ratio']:.2f}"
            })
    
    return pd.DataFrame(results)


# ============================================================================
# INTERACTIVE CLI ADVISOR
# ============================================================================

def run_interactive_advisor():
    """
    Run interactive command-line robo-advisor.
    
    This function provides a simple CLI interface for users
    to get personalized hedge recommendations.
    """
    print("\n" + "="*70)
    print("       CRYPTO STRANGLE HEDGING ROBO-ADVISOR")
    print("       HKUST IEDA3330 - Financial Engineering")
    print("="*70)
    
    print("\nWelcome! I'll help you find the optimal hedge for your short strangle.")
    print("\nRisk Profiles Available:")
    for name, profile in RISK_PROFILES.items():
        print(f"  • {name}: {profile['description']}")
    
    # Get user input
    print("\n" + "-"*70)
    choice = input("Enter profile name (or 'custom' for custom risk aversion): ").strip().lower()
    
    if choice == 'custom':
        try:
            risk_aversion = float(input("Enter risk aversion (0.5-20, higher=more conservative): "))
            result = robo_advisor_interface(risk_aversion=risk_aversion)
        except ValueError:
            print("Invalid input. Using moderate profile.")
            result = robo_advisor_interface(risk_profile='moderate')
    elif choice in RISK_PROFILES:
        result = robo_advisor_interface(risk_profile=choice)
    else:
        print(f"Unknown profile '{choice}'. Using moderate.")
        result = robo_advisor_interface(risk_profile='moderate')
    
    # Display recommendation
    print(result['recommendation'])
    
    if result['status'] == 'optimal':
        print("\nDetailed Metrics:")
        for key, value in result['metrics'].items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Show profile comparison
    print("\n" + "-"*70)
    print("COMPARISON OF ALL PROFILES:")
    print("-"*70)
    comparison = compare_profiles()
    print(comparison.to_string(index=False))
    
    print("\n" + "="*70)
    print("Thank you for using the Robo-Advisor!")
    print("="*70 + "\n")


# ============================================================================
# MAIN (for testing and demonstration)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROBO-ADVISOR MODULE TEST")
    print("="*70)
    
    # Test 1: Default moderate profile
    print("\n1. Testing Moderate Profile:")
    result = robo_advisor_interface(risk_profile='moderate')
    print(result['recommendation'])
    
    # Test 2: Custom risk aversion
    print("\n2. Testing Custom Risk Aversion (A=7.5):")
    result = robo_advisor_interface(risk_aversion=7.5)
    print(f"   Weights: {result['weights']}")
    print(f"   Volatility: {result['metrics']['volatility']:.1f}%")
    
    # Test 3: Compare all profiles
    print("\n3. Profile Comparison:")
    print("-"*70)
    comparison = compare_profiles()
    print(comparison.to_string(index=False))
    
    # Test 4: Sensitivity analysis
    print("\n4. Risk Aversion Sensitivity:")
    print("-"*70)
    sensitivity = analyze_risk_aversion_sensitivity()
    print(sensitivity.to_string(index=False))
    
    # Test 5: Interactive mode (commented out for automated testing)
    # print("\n5. Interactive Mode:")
    # run_interactive_advisor()

