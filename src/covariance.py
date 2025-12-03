"""
EWMA Covariance Estimation Module
=================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module implements:
- EWMA (Exponentially Weighted Moving Average) covariance estimation
- RiskMetrics methodology with λ = 0.94
- 3x3 covariance matrix for [r_spot, r_basis, dvol_chg]

References:
- Lecture 4: Covariance estimation for Mean-Variance Analysis
- RiskMetrics Technical Document (J.P. Morgan, 1996)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from .data import EWMA_LAMBDA


# ============================================================================
# EWMA COVARIANCE ESTIMATION
# ============================================================================

class EWMACovariance:
    """
    EWMA Covariance Matrix Estimator.
    
    Implements the RiskMetrics methodology for time-varying
    covariance estimation.
    
    Attributes
    ----------
    lambda_ : float
        Decay factor (default 0.94 for daily data)
    n_assets : int
        Number of assets/factors
    current_cov : np.ndarray
        Current covariance matrix estimate
    
    Notes
    -----
    From Lecture 4 and RiskMetrics:
    
    EWMA assigns exponentially declining weights to past observations:
    - Recent observations get higher weight
    - λ = 0.94 implies half-life ≈ 11 days
    
    Update formula for variance:
    σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
    
    Update formula for covariance:
    σ_ij,t = λ * σ_ij,{t-1} + (1-λ) * r_i,{t-1} * r_j,{t-1}
    
    Benefits over simple rolling covariance:
    1. More responsive to recent changes
    2. No "ghosting" effect when old data drops out
    3. Adapts to volatility regimes
    """
    
    def __init__(self, lambda_: float = EWMA_LAMBDA, n_assets: int = 3):
        """
        Initialize EWMA Covariance estimator.
        
        Parameters
        ----------
        lambda_ : float
            Decay factor (0 < λ < 1)
            Higher λ = slower decay = smoother estimates
            Lower λ = faster decay = more reactive
        n_assets : int
            Number of assets/factors in covariance matrix
        """
        self.lambda_ = lambda_
        self.n_assets = n_assets
        self.current_cov = None
        self.initialized = False
        
    def initialize(self, returns: np.ndarray, min_periods: int = 20) -> np.ndarray:
        """
        Initialize covariance matrix with sample covariance.
        
        Parameters
        ----------
        returns : np.ndarray
            Initial returns matrix (n_obs x n_assets)
        min_periods : int
            Minimum observations for initialization
        
        Returns
        -------
        np.ndarray
            Initial covariance matrix
        
        Notes
        -----
        Use simple sample covariance for initialization:
        Σ_0 = (1/T) * Σ_t (r_t - r̄)(r_t - r̄)ᵀ
        
        This provides a stable starting point before
        EWMA begins its recursive updates.
        """
        if len(returns) < min_periods:
            raise ValueError(f"Need at least {min_periods} observations for initialization")
        
        # Use first min_periods observations for sample covariance
        init_returns = returns[:min_periods]
        
        # Demean returns
        demeaned = init_returns - init_returns.mean(axis=0)
        
        # Sample covariance
        self.current_cov = (demeaned.T @ demeaned) / (min_periods - 1)
        self.initialized = True
        
        return self.current_cov.copy()
    
    def update(self, return_vector: np.ndarray) -> np.ndarray:
        """
        Update covariance matrix with new return observation.
        
        Parameters
        ----------
        return_vector : np.ndarray
            New return observation (1 x n_assets)
        
        Returns
        -------
        np.ndarray
            Updated covariance matrix
        
        Notes
        -----
        EWMA update formula (vectorized):
        Σ_t = λ * Σ_{t-1} + (1-λ) * r_{t-1} * r_{t-1}ᵀ
        
        This is equivalent to:
        σ²_i,t = λ * σ²_i,{t-1} + (1-λ) * r²_i,{t-1}
        σ_ij,t = λ * σ_ij,{t-1} + (1-λ) * r_i,{t-1} * r_j,{t-1}
        """
        if not self.initialized:
            raise RuntimeError("Must initialize before updating")
        
        r = return_vector.reshape(-1, 1)
        outer_product = r @ r.T
        
        self.current_cov = self.lambda_ * self.current_cov + \
                          (1 - self.lambda_) * outer_product
        
        return self.current_cov.copy()
    
    def get_correlation(self) -> np.ndarray:
        """
        Convert current covariance to correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix
        
        Notes
        -----
        ρ_ij = σ_ij / (σ_i * σ_j)
        
        Correlation is often more interpretable than covariance
        for comparing across assets with different volatilities.
        """
        if not self.initialized:
            raise RuntimeError("Must initialize first")
        
        std = np.sqrt(np.diag(self.current_cov))
        correlation = self.current_cov / np.outer(std, std)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(correlation, 1.0)
        
        return correlation
    
    def get_volatilities(self) -> np.ndarray:
        """
        Extract volatilities (standard deviations) from covariance.
        
        Returns
        -------
        np.ndarray
            Volatility vector
        """
        if not self.initialized:
            raise RuntimeError("Must initialize first")
        
        return np.sqrt(np.diag(self.current_cov))


# ============================================================================
# BATCH EWMA COMPUTATION
# ============================================================================

def compute_ewma_covariance_series(returns_df: pd.DataFrame,
                                    lambda_: float = EWMA_LAMBDA,
                                    init_periods: int = 20) -> List[np.ndarray]:
    """
    Compute full time series of EWMA covariance matrices.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns DataFrame with columns for each factor
    lambda_ : float
        EWMA decay factor
    init_periods : int
        Number of periods for initialization
    
    Returns
    -------
    List[np.ndarray]
        List of covariance matrices, one per time period
    
    Notes
    -----
    This function computes the EWMA covariance at each point in time,
    allowing us to use the "look-ahead-free" covariance for optimization.
    
    Each Σ_t uses only information available up to time t-1.
    """
    n_obs, n_assets = returns_df.shape
    returns = returns_df.values
    
    # Initialize EWMA estimator
    ewma = EWMACovariance(lambda_=lambda_, n_assets=n_assets)
    
    # Store covariance series
    cov_series = []
    
    # Initialize with first init_periods
    init_cov = ewma.initialize(returns, min_periods=init_periods)
    
    # For first init_periods, use initialization covariance
    for t in range(init_periods):
        cov_series.append(init_cov.copy())
    
    # Update and store for remaining periods
    for t in range(init_periods, n_obs):
        # Update with yesterday's return
        cov = ewma.update(returns[t-1])
        cov_series.append(cov.copy())
    
    return cov_series


# ============================================================================
# HEDGE ASSET COVARIANCE
# ============================================================================

def compute_hedge_asset_covariance(asset_returns: pd.DataFrame,
                                    lambda_: float = EWMA_LAMBDA,
                                    init_periods: int = 20) -> List[np.ndarray]:
    """
    Compute EWMA covariance for hedging assets (spot, futures, risk-free).
    
    Parameters
    ----------
    asset_returns : pd.DataFrame
        DataFrame with columns: ['r_spot_asset', 'r_futures_asset', 'r_rf']
    lambda_ : float
        EWMA decay factor
    init_periods : int
        Initialization period
    
    Returns
    -------
    List[np.ndarray]
        Covariance matrices for each time period
    
    Notes
    -----
    The 3x3 covariance matrix structure:
    
           | Spot   Futures  RF   |
    Spot   | σ²_s   σ_sf     0    |
    Futures| σ_sf   σ²_f     0    |
    RF     | 0      0        ~0   |
    
    Risk-free has near-zero variance and zero correlation.
    """
    # For risk-free, variance is effectively zero
    # Compute 2x2 for spot/futures, then expand
    
    risky_returns = asset_returns[['r_spot_asset', 'r_futures_asset']].copy()
    
    n_obs = len(asset_returns)
    cov_2x2_series = compute_ewma_covariance_series(risky_returns, lambda_, init_periods)
    
    # Expand to 3x3 with risk-free
    cov_series = []
    rf_var = 1e-10  # Near-zero variance for risk-free
    
    for cov_2x2 in cov_2x2_series:
        cov_3x3 = np.zeros((3, 3))
        cov_3x3[:2, :2] = cov_2x2
        cov_3x3[2, 2] = rf_var
        cov_series.append(cov_3x3)
    
    return cov_series


# ============================================================================
# COVARIANCE ANALYSIS UTILITIES
# ============================================================================

def decompose_covariance(cov_matrix: np.ndarray,
                         asset_names: List[str] = None) -> pd.DataFrame:
    """
    Decompose covariance matrix into interpretable components.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix
    asset_names : List[str], optional
        Names for assets/factors
    
    Returns
    -------
    pd.DataFrame
        DataFrame with volatilities, correlations, and betas
    
    Notes
    -----
    Useful for understanding risk structure:
    - Volatilities: Individual asset risk
    - Correlations: Diversification potential
    - Betas: Systematic risk contribution
    """
    n = cov_matrix.shape[0]
    
    if asset_names is None:
        asset_names = [f'Asset_{i}' for i in range(n)]
    
    # Volatilities (annualized for daily data)
    vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(365)
    
    # Correlation matrix
    std = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Asset': asset_names,
        'Ann_Volatility': vols,
    })
    
    # Add correlations with first asset (spot)
    for i, name in enumerate(asset_names):
        summary[f'Corr_with_{asset_names[0]}'] = corr[:, 0]
    
    return summary


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
    
    Notes
    -----
    Numerical issues can cause EWMA covariance to become
    near-singular. This function:
    1. Eigendecomposes the matrix
    2. Floors small eigenvalues
    3. Reconstructs the matrix
    
    Essential for cvxpy optimization which requires PSD matrices.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Floor small eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct
    cov_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure symmetry
    cov_fixed = (cov_fixed + cov_fixed.T) / 2
    
    return cov_fixed


def calculate_portfolio_variance(weights: np.ndarray,
                                  cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio variance given weights and covariance.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns
    -------
    float
        Portfolio variance
    
    Notes
    -----
    From Lecture 4: Portfolio variance
    σ²_p = w'Σw
    
    This is the objective function we minimize in
    mean-variance optimization.
    """
    return weights @ cov_matrix @ weights


def calculate_portfolio_volatility(weights: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    annualize: bool = True) -> float:
    """
    Calculate portfolio volatility (standard deviation).
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    annualize : bool
        Whether to annualize the volatility
    
    Returns
    -------
    float
        Portfolio volatility
    """
    variance = calculate_portfolio_variance(weights, cov_matrix)
    vol = np.sqrt(variance)
    
    if annualize:
        vol *= np.sqrt(365)
    
    return vol


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    from .data import load_all_data
    from .returns import calculate_all_returns, calculate_hedge_asset_returns
    
    # Load data
    df = load_all_data("2022-01-01", "2025-12-03")
    
    # Calculate factor returns
    returns_df = calculate_all_returns(df)
    print("\nFactor Returns:")
    print(returns_df.head())
    
    # Compute EWMA covariance series
    cov_series = compute_ewma_covariance_series(returns_df)
    print(f"\nComputed {len(cov_series)} covariance matrices")
    
    # Show last covariance matrix
    print("\nLast EWMA Covariance Matrix:")
    last_cov = cov_series[-1]
    print(pd.DataFrame(last_cov, 
                       index=returns_df.columns, 
                       columns=returns_df.columns))
    
    # Show correlation
    ewma = EWMACovariance(n_assets=3)
    ewma.current_cov = last_cov
    ewma.initialized = True
    print("\nCorrelation Matrix:")
    print(pd.DataFrame(ewma.get_correlation(),
                       index=returns_df.columns,
                       columns=returns_df.columns))
    
    # Annualized volatilities
    print("\nAnnualized Volatilities:")
    vols = ewma.get_volatilities() * np.sqrt(365)
    for col, vol in zip(returns_df.columns, vols):
        print(f"  {col}: {vol:.2%}")
    
    # Hedge asset covariance
    asset_returns = calculate_hedge_asset_returns(df)
    hedge_cov_series = compute_hedge_asset_covariance(asset_returns)
    print(f"\nComputed {len(hedge_cov_series)} hedge asset covariance matrices")
    
    print("\nLast Hedge Asset Covariance:")
    print(hedge_cov_series[-1])

