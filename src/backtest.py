"""
Backtesting Engine Module
=========================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module backtests THREE METHODOLOGIES separately:

METHODOLOGY 1 (M1): Option Strategy and Delta-Hedging (Lecture 6)
  - Simple 1:1 delta hedge using BTC futures
  - Hedge ratio = -net_delta (basic delta-neutral)
  - Does NOT use EWMA or optimization
  
METHODOLOGY 2 (M2): Dynamic Covariance Estimation via EWMA (Lecture 7)
  - Uses EWMA volatility to adjust hedge ratio
  - Hedge ratio = -net_delta * (σ_spot / σ_futures)
  - Accounts for time-varying volatility
  - Does NOT use full optimization

METHODOLOGY 3 (M3): Mean-Variance Optimal Portfolio Construction (Lecture 5)
  - Full Markowitz optimization using EWMA covariance
  - Minimizes portfolio variance subject to delta-neutrality
  - Uses cvxpy for quadratic optimization

Backtest compares M1, M2, M3 head-to-head to show:
- M1: Basic delta-hedging concept
- M2: Improvement from EWMA covariance estimation
- M3: Improvement from mean-variance optimization

Performance Metrics:
- Sharpe Ratio, Max Drawdown, VaR, Win Rate
- Sub-period analysis (e.g., FTX crash Nov 2022)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .data import load_all_data, RISK_FREE_RATE_ANNUAL, RISK_FREE_RATE_DAILY
from .returns import (
    calculate_all_returns, 
    calculate_hedge_asset_returns,
    calculate_strangle_pnl_components,
    annualize_returns,
    annualize_volatility,
    calculate_sharpe_ratio
)
# Import from methodology files (M1, M2, M3)
from . import _1_Option_Delta as m1_module  # M1: Option Strategy and Delta-Hedging
from . import _2_Covariance_Estimation as m2_module  # M2: EWMA Covariance Estimation
from . import _3_MV_Optimization as m3_module  # M3: Mean-Variance Optimization

# Import functions from methodology modules
from ._1_Option_Delta import calculate_strangle_pnl, compute_naive_hedge
from ._2_Covariance_Estimation import compute_hedge_asset_covariance, compute_ewma_hedge_weights
from ._3_MV_Optimization import optimize_hedge_portfolio


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_max_drawdown(cumulative_pnl: pd.Series, 
                           initial_capital: float = 100000) -> float:
    """
    Calculate maximum drawdown from cumulative P&L series.
    
    Parameters
    ----------
    cumulative_pnl : pd.Series
        Cumulative P&L series
    initial_capital : float
        Starting capital (default $100,000)
    
    Returns
    -------
    float
        Maximum drawdown (as positive percentage, max 1.0 = 100%)
    
    Notes
    -----
    Max Drawdown = max(peak - trough) / peak
    
    Measures worst peak-to-trough decline.
    Important for risk management and investor psychology.
    
    We calculate based on portfolio value (capital + P&L) to avoid
    division issues when cumulative P&L is negative.
    """
    # Convert cumulative P&L to portfolio value
    portfolio_value = initial_capital + cumulative_pnl
    
    # Running maximum of portfolio value
    running_max = portfolio_value.cummax()
    
    # Drawdown series (as percentage of peak)
    drawdown = (running_max - portfolio_value) / running_max
    
    # Cap at 100% (can't lose more than 100% of portfolio)
    return min(drawdown.max(), 1.0)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at specified confidence level.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence : float
        Confidence level (default 95%)
    
    Returns
    -------
    float
        VaR (as positive number representing potential loss)
    
    Notes
    -----
    From Lecture 2: VaR is the loss level that will not be exceeded
    with probability (1 - α).
    
    VaR_α = -q_α(returns) where q_α is the α-quantile
    
    For 95% VaR: We expect to lose more than VaR on 5% of days.
    """
    alpha = 1 - confidence
    return -np.percentile(returns, alpha * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence : float
        Confidence level
    
    Returns
    -------
    float
        CVaR (expected loss given VaR breach)
    
    Notes
    -----
    CVaR = E[Loss | Loss > VaR]
    
    More coherent risk measure than VaR as it accounts
    for tail severity, not just frequency.
    """
    var = calculate_var(returns, confidence)
    # Average of returns worse than VaR
    return -returns[returns <= -var].mean()


def calculate_performance_metrics(daily_returns: pd.Series,
                                  cumulative_pnl: pd.Series = None) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series
    cumulative_pnl : pd.Series, optional
        Cumulative P&L for drawdown calculation
    
    Returns
    -------
    Dict
        Dictionary of performance metrics
    """
    metrics = {
        'total_return': (1 + daily_returns).prod() - 1,
        'annualized_return': annualize_returns(daily_returns),
        'annualized_volatility': annualize_volatility(daily_returns),
        'sharpe_ratio': calculate_sharpe_ratio(daily_returns),
        'var_95': calculate_var(daily_returns, 0.95),
        'var_99': calculate_var(daily_returns, 0.99),
        'skewness': daily_returns.skew(),
        'kurtosis': daily_returns.kurtosis(),
        'win_rate': (daily_returns > 0).mean(),
        'avg_win': daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0,
        'avg_loss': daily_returns[daily_returns < 0].mean() if (daily_returns < 0).any() else 0,
    }
    
    if cumulative_pnl is not None:
        metrics['max_drawdown'] = calculate_max_drawdown(cumulative_pnl)
    
    return metrics


# ============================================================================
# P&L CALCULATION
# ============================================================================

def calculate_hedge_pnl(df: pd.DataFrame,
                        weights: pd.DataFrame,
                        notional: float = 100000) -> pd.DataFrame:
    """
    Calculate P&L from hedge positions (legacy function).
    """
    return calculate_hedge_pnl_three_methods(df, weights, notional)


def calculate_hedge_pnl_three_methods(df: pd.DataFrame,
                                       weights: pd.DataFrame,
                                       notional: float = 100000) -> pd.DataFrame:
    """
    Calculate P&L for all three methodologies (M1, M2, M3).
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with spot, futures
    weights : pd.DataFrame
        Hedge weights with m1_w_*, m2_w_*, m3_w_* columns
    notional : float
        Notional value for hedge
    
    Returns
    -------
    pd.DataFrame
        Hedge P&L for M1, M2, M3
    
    Notes
    -----
    M1: Delta Hedge (Lecture 6) - Simple 1:1 futures hedge
    M2: EWMA Hedge (Lecture 7) - Volatility-adjusted hedge ratio
    M3: MV Optimal (Lecture 5) - Full Markowitz optimization
    """
    hedge_pnl = pd.DataFrame(index=df.index)
    
    # Asset returns
    r_spot = df['spot'].pct_change()
    r_futures = df['futures'].pct_change()
    r_rf = RISK_FREE_RATE_DAILY
    
    # ================================================================
    # M1: Delta Hedge P&L (Lecture 6)
    # ================================================================
    hedge_pnl['m1_spot_pnl'] = weights['m1_w_spot'] * r_spot * notional
    hedge_pnl['m1_futures_pnl'] = weights['m1_w_futures'] * r_futures * notional
    hedge_pnl['m1_cash_pnl'] = weights['m1_w_cash'] * r_rf * notional
    hedge_pnl['total_hedge_m1'] = hedge_pnl[['m1_spot_pnl', 'm1_futures_pnl', 'm1_cash_pnl']].sum(axis=1)
    
    # ================================================================
    # M2: EWMA Hedge P&L (Lecture 7)
    # ================================================================
    hedge_pnl['m2_spot_pnl'] = weights['m2_w_spot'] * r_spot * notional
    hedge_pnl['m2_futures_pnl'] = weights['m2_w_futures'] * r_futures * notional
    hedge_pnl['m2_cash_pnl'] = weights['m2_w_cash'] * r_rf * notional
    hedge_pnl['total_hedge_m2'] = hedge_pnl[['m2_spot_pnl', 'm2_futures_pnl', 'm2_cash_pnl']].sum(axis=1)
    
    # ================================================================
    # M3: MV Optimal P&L (Lecture 5)
    # ================================================================
    hedge_pnl['m3_spot_pnl'] = weights['m3_w_spot'] * r_spot * notional
    hedge_pnl['m3_futures_pnl'] = weights['m3_w_futures'] * r_futures * notional
    hedge_pnl['m3_cash_pnl'] = weights['m3_w_cash'] * r_rf * notional
    hedge_pnl['total_hedge_m3'] = hedge_pnl[['m3_spot_pnl', 'm3_futures_pnl', 'm3_cash_pnl']].sum(axis=1)
    
    # Legacy compatibility
    hedge_pnl['total_hedge_naive'] = hedge_pnl['total_hedge_m1']
    hedge_pnl['total_hedge_mv'] = hedge_pnl['total_hedge_m3']
    
    # Drop first row
    hedge_pnl = hedge_pnl.iloc[1:]
    
    return hedge_pnl


# ============================================================================
# MAIN BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """
    Main backtesting engine for delta-hedged short strangles.
    
    Implements daily rebalancing with:
    - EWMA covariance estimation
    - Mean-variance optimal hedging
    - Naive benchmark comparison
    - Comprehensive performance analytics
    
    Attributes
    ----------
    df : pd.DataFrame
        Market data
    notional : float
        Notional value of strangle
    results : Dict
        Backtest results
    """
    
    def __init__(self, start_date: str = "2022-01-01",
                 end_date: str = None,
                 notional: float = 100000):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        start_date : str
            Backtest start date
        end_date : str
            Backtest end date
        notional : float
            Notional value
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.notional = notional
        
        self.df = None
        self.results = {}
        self.weights_history = None
        self.pnl_history = None
        
    def load_data(self):
        """Load all required data."""
        print("Loading data...")
        self.df = load_all_data(self.start_date, self.end_date)
        return self
    
    def run_backtest(self, init_periods: int = 20):
        """
        Run full backtest.
        
        Parameters
        ----------
        init_periods : int
            Number of periods for EWMA initialization
        
        Returns
        -------
        self
            For method chaining
        
        Notes
        -----
        Backtest procedure:
        1. Calculate asset returns
        2. Compute EWMA covariance series
        3. For each day:
           a. Get current covariance estimate
           b. Optimize hedge weights (MV and naive)
           c. Calculate strategy P&L
        4. Aggregate results and compute metrics
        """
        print("Running backtest...")
        
        # Calculate returns
        asset_returns = calculate_hedge_asset_returns(self.df)
        
        # Compute EWMA covariance
        cov_series = compute_hedge_asset_covariance(asset_returns, init_periods=init_periods)
        
        # Strangle P&L components
        strangle_pnl = calculate_strangle_pnl(self.df, self.notional)
        
        # Align indices
        common_index = strangle_pnl.index.intersection(asset_returns.index)
        strangle_pnl = strangle_pnl.loc[common_index]
        
        # Initialize results storage
        n_periods = len(common_index)
        weights_list = []
        
        print(f"Optimizing {n_periods} periods...")
        
        # Run optimization for each period
        for t, date in enumerate(common_index):
            # Get covariance (use t + init_periods to account for offset)
            cov_idx = min(t + init_periods, len(cov_series) - 1)
            cov = cov_series[cov_idx]
            
            # Get net delta
            delta_idx = self.df.index.get_loc(date)
            net_delta = self.df['net_delta'].iloc[delta_idx]
            
            # ================================================================
            # M1: Simple Delta Hedge (Lecture 6)
            # ================================================================
            # Simple 1:1 futures hedge: hedge_ratio = -net_delta
            m1_weights = compute_naive_hedge(net_delta)
            
            # ================================================================
            # M2: EWMA Volatility-Adjusted Hedge (Lecture 7)
            # ================================================================
            # Uses EWMA covariance to compute minimum-variance hedge ratio
            m2_weights = compute_ewma_hedge_weights(cov, net_delta)
            
            # ================================================================
            # M3: Mean-Variance Optimal (Lecture 5)
            # ================================================================
            # Full Markowitz optimization with delta-neutrality constraint
            m3_weights, diagnostics = optimize_hedge_portfolio(cov, net_delta)
            
            weights_list.append({
                'date': date,
                'net_delta': net_delta,
                # M1: Simple Delta Hedge
                'm1_w_spot': m1_weights[0],
                'm1_w_futures': m1_weights[1],
                'm1_w_cash': m1_weights[2],
                # M2: EWMA Volatility-Adjusted
                'm2_w_spot': m2_weights[0],
                'm2_w_futures': m2_weights[1],
                'm2_w_cash': m2_weights[2],
                # M3: MV Optimal
                'm3_w_spot': m3_weights[0],
                'm3_w_futures': m3_weights[1],
                'm3_w_cash': m3_weights[2],
                'opt_status': diagnostics.get('status', 'unknown')
            })
            
            if (t + 1) % 200 == 0:
                print(f"  Processed {t + 1}/{n_periods} periods")
        
        # Create weights DataFrame
        self.weights_history = pd.DataFrame(weights_list)
        self.weights_history.set_index('date', inplace=True)
        
        # Calculate hedge P&L for each methodology
        hedge_pnl = calculate_hedge_pnl_three_methods(
            self.df.loc[common_index], 
            self.weights_history, 
            self.notional
        )
        
        # Combine P&L
        self.pnl_history = strangle_pnl.copy()
        self.pnl_history = self.pnl_history.join(hedge_pnl, how='inner')
        
        # Net P&L for each methodology (M1, M2, M3)
        self.pnl_history['pnl_m1'] = self.pnl_history['total_unhedged'] + self.pnl_history['total_hedge_m1']
        self.pnl_history['pnl_m2'] = self.pnl_history['total_unhedged'] + self.pnl_history['total_hedge_m2']
        self.pnl_history['pnl_m3'] = self.pnl_history['total_unhedged'] + self.pnl_history['total_hedge_m3']
        
        # Keep legacy names for backward compatibility
        self.pnl_history['pnl_naive_hedged'] = self.pnl_history['pnl_m1']
        self.pnl_history['pnl_mv_hedged'] = self.pnl_history['pnl_m3']
        self.pnl_history['pnl_unhedged'] = self.pnl_history['total_unhedged']
        
        # Cumulative P&L
        self.pnl_history['cum_m1'] = self.pnl_history['pnl_m1'].cumsum()
        self.pnl_history['cum_m2'] = self.pnl_history['pnl_m2'].cumsum()
        self.pnl_history['cum_m3'] = self.pnl_history['pnl_m3'].cumsum()
        
        # Legacy cumulative names
        self.pnl_history['cum_naive_hedged'] = self.pnl_history['cum_m1']
        self.pnl_history['cum_mv_hedged'] = self.pnl_history['cum_m3']
        self.pnl_history['cum_unhedged'] = self.pnl_history['total_unhedged'].cumsum()
        
        # Daily returns (percentage of notional)
        self.pnl_history['ret_m1'] = self.pnl_history['pnl_m1'] / self.notional
        self.pnl_history['ret_m2'] = self.pnl_history['pnl_m2'] / self.notional
        self.pnl_history['ret_m3'] = self.pnl_history['pnl_m3'] / self.notional
        
        # Legacy return names
        self.pnl_history['ret_naive_hedged'] = self.pnl_history['ret_m1']
        self.pnl_history['ret_mv_hedged'] = self.pnl_history['ret_m3']
        self.pnl_history['ret_unhedged'] = self.pnl_history['total_unhedged'] / self.notional
        
        print("Backtest complete!")
        return self
    
    def calculate_metrics(self, period_name: str = "Full Period") -> Dict:
        """
        Calculate performance metrics for current period.
        
        Parameters
        ----------
        period_name : str
            Name for this analysis period
        
        Returns
        -------
        Dict
            Performance metrics for M1, M2, M3
        """
        metrics = {'period': period_name}
        
        # Calculate metrics for M1, M2, M3
        for method in ['m1', 'm2', 'm3']:
            ret_col = f'ret_{method}'
            cum_col = f'cum_{method}'
            
            if ret_col in self.pnl_history.columns:
                m = calculate_performance_metrics(
                    self.pnl_history[ret_col],
                    self.pnl_history[cum_col]
                )
                for k, v in m.items():
                    metrics[f'{method}_{k}'] = v
        
        # Legacy compatibility
        for old, new in [('naive_hedged', 'm1'), ('mv_hedged', 'm3'), ('unhedged', 'm1')]:
            for k in ['annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown', 'var_95', 'win_rate']:
                if f'{new}_{k}' in metrics:
                    metrics[f'{old}_{k}'] = metrics[f'{new}_{k}']
        
        return metrics
    
    def analyze_subperiods(self) -> pd.DataFrame:
        """
        Analyze performance across key sub-periods.
        
        Returns
        -------
        pd.DataFrame
            Metrics for each sub-period
        
        Notes
        -----
        Key periods for BTC:
        - 2022 Bear Market (Terra/Luna, FTX)
        - 2023 Recovery
        - 2024 Bull Run (ETF approvals)
        - 2025 Current
        """
        subperiods = [
            ('Full Period', self.start_date, self.end_date),
            ('2022 (Bear/FTX)', '2022-01-01', '2022-12-31'),
            ('FTX Crash', '2022-11-01', '2022-11-30'),
            ('2023 (Recovery)', '2023-01-01', '2023-12-31'),
            ('2024 (ETF Bull)', '2024-01-01', '2024-12-31'),
            ('2025 YTD', '2025-01-01', self.end_date),
        ]
        
        results = []
        
        for name, start, end in subperiods:
            try:
                mask = (self.pnl_history.index >= start) & (self.pnl_history.index <= end)
                if mask.sum() < 10:  # Skip if too few observations
                    continue
                
                # Temporarily subset pnl_history
                original_pnl = self.pnl_history.copy()
                self.pnl_history = original_pnl.loc[mask]
                
                metrics = self.calculate_metrics(name)
                results.append(metrics)
                
                self.pnl_history = original_pnl
            except Exception as e:
                print(f"Skipping {name}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Generate summary comparison table.
        
        Returns
        -------
        pd.DataFrame
            Summary table comparing strategies
        """
        metrics = self.calculate_metrics()
        
        # Strategy names mapped to THREE METHODOLOGIES:
        # - Strangle Only: Uses Methodology 1 (Option Greeks) without hedging
        # - Delta Hedge: Uses Methodology 1 (Option Strategy + Delta-Hedging)
        # - MV Optimal: Uses Methodology 1 + 2 (EWMA) + 3 (Markowitz)
        summary = pd.DataFrame({
            'Metric': [
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                '95% VaR',
                'Win Rate'
            ],
            'M1: Delta Hedge': [
                f"{metrics.get('m1_annualized_return', 0):.2%}",
                f"{metrics.get('m1_annualized_volatility', 0):.2%}",
                f"{metrics.get('m1_sharpe_ratio', 0):.2f}",
                f"{metrics.get('m1_max_drawdown', 0):.2%}",
                f"{metrics.get('m1_var_95', 0):.4f}",
                f"{metrics.get('m1_win_rate', 0):.2%}"
            ],
            'M2: EWMA Hedge': [
                f"{metrics.get('m2_annualized_return', 0):.2%}",
                f"{metrics.get('m2_annualized_volatility', 0):.2%}",
                f"{metrics.get('m2_sharpe_ratio', 0):.2f}",
                f"{metrics.get('m2_max_drawdown', 0):.2%}",
                f"{metrics.get('m2_var_95', 0):.4f}",
                f"{metrics.get('m2_win_rate', 0):.2%}"
            ],
            'M3: MV Optimal': [
                f"{metrics.get('m3_annualized_return', 0):.2%}",
                f"{metrics.get('m3_annualized_volatility', 0):.2%}",
                f"{metrics.get('m3_sharpe_ratio', 0):.2f}",
                f"{metrics.get('m3_max_drawdown', 0):.2%}",
                f"{metrics.get('m3_var_95', 0):.4f}",
                f"{metrics.get('m3_win_rate', 0):.2%}"
            ]
        })
        
        return summary
    
    def get_volatility_comparison(self) -> pd.DataFrame:
        """
        Generate volatility comparison table (as requested in project spec).
        
        Returns
        -------
        pd.DataFrame
            Table with volatility comparison across periods
        """
        subperiods_df = self.analyze_subperiods()
        
        if subperiods_df.empty:
            return pd.DataFrame()
        
        comparison = pd.DataFrame({
            'Period': subperiods_df['period'],
            'M1: Delta': subperiods_df.get('m1_annualized_volatility', 0),
            'M2: EWMA': subperiods_df.get('m2_annualized_volatility', 0),
            'M3: MV Opt': subperiods_df.get('m3_annualized_volatility', 0),
        })
        
        # Calculate improvement percentage
        comparison['M3 vs M1'] = (
            (comparison['M1: Delta'] - comparison['M3: MV Opt']) / 
            comparison['M1: Delta'] * 100
        )
        comparison['M3 vs M2'] = (
            (comparison['M2: EWMA'] - comparison['M3: MV Opt']) / 
            comparison['M2: EWMA'] * 100
        )
        
        # Format percentages
        for col in ['M1: Delta', 'M2: EWMA', 'M3: MV Opt']:
            comparison[col] = comparison[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        
        for col in ['M3 vs M1', 'M3 vs M2']:
            comparison[col] = comparison[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        
        return comparison


# ============================================================================
# TRAIN/VALIDATION/TEST ENGINE
# ============================================================================

class TrainValTestEngine:
    """
    Rigorous backtesting engine with proper train/validation/test split.
    
    Split Structure:
    ----------------
    - Training Period (2022-01-01 to 2023-12-31): ~2 years
      - Calibrate base models
      - Estimate initial EWMA parameters
      
    - Validation Period (2024-01-01 to 2024-06-30): ~6 months
      - Tune hyperparameters (lambda, risk aversion)
      - Select best configuration based on Sharpe ratio
      
    - Test Period (2024-07-01 to 2025-12-03): ~1.5 years
      - Final out-of-sample evaluation
      - NO parameter tuning allowed
      - Report final metrics
    
    This avoids overfitting and provides honest performance estimates.
    """
    
    def __init__(self,
                 train_start: str = "2022-01-01",
                 train_end: str = "2023-12-31",
                 val_start: str = "2024-01-01",
                 val_end: str = "2024-06-30",
                 test_start: str = "2024-07-01",
                 test_end: str = "2025-12-03",
                 notional: float = 100000):
        """
        Initialize train/validation/test engine.
        
        Parameters
        ----------
        train_start, train_end : str
            Training period dates
        val_start, val_end : str
            Validation period dates
        test_start, test_end : str
            Test period dates
        notional : float
            Notional value
        """
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.notional = notional
        
        # Data
        self.full_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Best hyperparameters (tuned on validation)
        self.best_lambda = 0.94
        self.best_risk_aversion = 0.0
        
        # Results
        self.train_results = {}
        self.val_results = {}
        self.test_results = {}
        self.tuning_results = []
        
        # Final test engine
        self.test_engine = None
    
    def load_data(self):
        """Load data for all periods."""
        print("=" * 60)
        print("LOADING DATA FOR TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 60)
        
        # Load full dataset
        self.full_df = load_all_data(self.train_start, self.test_end)
        
        # Split into periods
        self.train_df = self.full_df[
            (self.full_df.index >= self.train_start) & 
            (self.full_df.index <= self.train_end)
        ].copy()
        
        self.val_df = self.full_df[
            (self.full_df.index >= self.val_start) & 
            (self.full_df.index <= self.val_end)
        ].copy()
        
        self.test_df = self.full_df[
            (self.full_df.index >= self.test_start) & 
            (self.full_df.index <= self.test_end)
        ].copy()
        
        print(f"\nData Split Summary:")
        print(f"  Training:   {len(self.train_df):4d} days ({self.train_start} to {self.train_end})")
        print(f"  Validation: {len(self.val_df):4d} days ({self.val_start} to {self.val_end})")
        print(f"  Test:       {len(self.test_df):4d} days ({self.test_start} to {self.test_end})")
        print(f"  Total:      {len(self.full_df):4d} days")
        
        return self
    
    def _run_backtest_on_period(self, df: pd.DataFrame, 
                                 ewma_lambda: float = 0.94,
                                 risk_aversion: float = 0.0,
                                 init_periods: int = 20) -> Dict:
        """
        Run backtest on a specific period with given hyperparameters.
        
        Returns dict with performance metrics.
        """
        from ._2_Covariance_Estimation import EWMACovarianceEstimator
        
        if len(df) < init_periods + 10:
            return {'sharpe_m1': np.nan, 'sharpe_m2': np.nan, 'sharpe_m3': np.nan}
        
        # Calculate returns
        asset_returns = calculate_hedge_asset_returns(df)
        
        # Custom EWMA covariance with specified lambda
        risky_returns = asset_returns[['r_spot_asset', 'r_futures_asset']].copy()
        estimator = EWMACovarianceEstimator(lambda_=ewma_lambda, n_assets=2)
        init_data = risky_returns.iloc[:init_periods].values
        estimator.initialize(init_data)
        
        cov_2x2_series = [estimator.current_cov.copy()]
        for i in range(init_periods, len(risky_returns)):
            returns_t = risky_returns.iloc[i].values
            estimator.update(returns_t)
            cov_2x2_series.append(estimator.current_cov.copy())
        
        # Expand to 3x3
        cov_series = []
        rf_var = 1e-10
        for cov_2x2 in cov_2x2_series:
            cov_3x3 = np.zeros((3, 3))
            cov_3x3[:2, :2] = cov_2x2
            cov_3x3[2, 2] = rf_var
            cov_series.append(cov_3x3)
        
        # Strangle P&L
        strangle_pnl = calculate_strangle_pnl(df, self.notional)
        
        # Align indices
        common_index = strangle_pnl.index.intersection(asset_returns.index)
        if len(common_index) < 20:
            return {'sharpe_m1': np.nan, 'sharpe_m2': np.nan, 'sharpe_m3': np.nan}
        
        strangle_pnl = strangle_pnl.loc[common_index]
        
        # Run optimization
        weights_list = []
        for t, date in enumerate(common_index):
            cov_idx = min(t + init_periods, len(cov_series) - 1)
            cov = cov_series[cov_idx]
            
            delta_idx = df.index.get_loc(date)
            net_delta = df['net_delta'].iloc[delta_idx]
            
            # M1: Simple delta hedge
            m1_weights = compute_naive_hedge(net_delta)
            
            # M2: EWMA hedge
            m2_weights = compute_ewma_hedge_weights(cov, net_delta)
            
            # M3: MV Optimal (with custom risk aversion)
            from ._3_MV_Optimization import optimize_hedge_portfolio
            m3_weights, _ = optimize_hedge_portfolio(cov, net_delta, risk_aversion=risk_aversion)
            
            weights_list.append({
                'date': date,
                'm1_w_spot': m1_weights[0], 'm1_w_futures': m1_weights[1], 'm1_w_cash': m1_weights[2],
                'm2_w_spot': m2_weights[0], 'm2_w_futures': m2_weights[1], 'm2_w_cash': m2_weights[2],
                'm3_w_spot': m3_weights[0], 'm3_w_futures': m3_weights[1], 'm3_w_cash': m3_weights[2],
            })
        
        weights_df = pd.DataFrame(weights_list).set_index('date')
        
        # Calculate P&L for each method
        hedge_pnl = calculate_hedge_pnl_three_methods(df.loc[common_index], weights_df, self.notional)
        
        # Combine with strangle P&L
        pnl = pd.DataFrame(index=common_index)
        pnl['m1_pnl'] = strangle_pnl['theta'] + strangle_pnl['gamma'] + strangle_pnl['vega'] + hedge_pnl['total_hedge_m1']
        pnl['m2_pnl'] = strangle_pnl['theta'] + strangle_pnl['gamma'] + strangle_pnl['vega'] + hedge_pnl['total_hedge_m2']
        pnl['m3_pnl'] = strangle_pnl['theta'] + strangle_pnl['gamma'] + strangle_pnl['vega'] + hedge_pnl['total_hedge_m3']
        
        # Calculate metrics
        def calc_sharpe(daily_pnl):
            if len(daily_pnl.dropna()) < 20:
                return np.nan
            mean_ret = daily_pnl.mean() / self.notional
            std_ret = daily_pnl.std() / self.notional
            if std_ret < 1e-10:
                return 0.0
            sharpe = (mean_ret - RISK_FREE_RATE_DAILY) / std_ret * np.sqrt(252)
            return sharpe
        
        def calc_vol(daily_pnl):
            if len(daily_pnl.dropna()) < 20:
                return np.nan
            return (daily_pnl.std() / self.notional) * np.sqrt(252)
        
        def calc_return(daily_pnl):
            if len(daily_pnl.dropna()) < 20:
                return np.nan
            return (daily_pnl.mean() / self.notional) * 252
        
        def calc_max_dd(daily_pnl):
            cum_pnl = daily_pnl.cumsum()
            running_max = (self.notional + cum_pnl).cummax()
            drawdown = (running_max - (self.notional + cum_pnl)) / running_max
            return drawdown.max()
        
        return {
            'sharpe_m1': calc_sharpe(pnl['m1_pnl']),
            'sharpe_m2': calc_sharpe(pnl['m2_pnl']),
            'sharpe_m3': calc_sharpe(pnl['m3_pnl']),
            'vol_m1': calc_vol(pnl['m1_pnl']),
            'vol_m2': calc_vol(pnl['m2_pnl']),
            'vol_m3': calc_vol(pnl['m3_pnl']),
            'return_m1': calc_return(pnl['m1_pnl']),
            'return_m2': calc_return(pnl['m2_pnl']),
            'return_m3': calc_return(pnl['m3_pnl']),
            'max_dd_m1': calc_max_dd(pnl['m1_pnl']),
            'max_dd_m2': calc_max_dd(pnl['m2_pnl']),
            'max_dd_m3': calc_max_dd(pnl['m3_pnl']),
            'pnl_df': pnl,
            'weights_df': weights_df
        }
    
    def train(self, init_periods: int = 20):
        """
        Phase 1: Training
        
        Calibrate models on training data.
        Establishes baseline performance.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: TRAINING (2022-01-01 to 2023-12-31)")
        print("=" * 60)
        print("Calibrating models on training data...")
        
        self.train_results = self._run_backtest_on_period(
            self.train_df, 
            ewma_lambda=0.94,  # Initial lambda
            risk_aversion=0.0,
            init_periods=init_periods
        )
        
        print(f"\nTraining Period Performance:")
        print(f"  M1 (Delta Hedge): Sharpe = {self.train_results['sharpe_m1']:.2f}, Vol = {self.train_results['vol_m1']*100:.2f}%")
        print(f"  M2 (EWMA Hedge):  Sharpe = {self.train_results['sharpe_m2']:.2f}, Vol = {self.train_results['vol_m2']*100:.2f}%")
        print(f"  M3 (MV Optimal):  Sharpe = {self.train_results['sharpe_m3']:.2f}, Vol = {self.train_results['vol_m3']*100:.2f}%")
        
        return self
    
    def validate(self, 
                 lambda_candidates: List[float] = [0.90, 0.92, 0.94, 0.96, 0.97],
                 risk_aversion_candidates: List[float] = [0.0, 1.0, 2.0, 5.0, 10.0],
                 init_periods: int = 20):
        """
        Phase 2: Validation
        
        Tune hyperparameters on validation data.
        Select best lambda and risk_aversion based on M3 Sharpe ratio.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: VALIDATION (2024-01-01 to 2024-06-30)")
        print("=" * 60)
        print("Tuning hyperparameters on validation data...")
        print(f"  Lambda candidates: {lambda_candidates}")
        print(f"  Risk aversion candidates: {risk_aversion_candidates}")
        
        best_sharpe = -np.inf
        self.tuning_results = []
        
        total_combinations = len(lambda_candidates) * len(risk_aversion_candidates)
        current = 0
        
        for lam in lambda_candidates:
            for ra in risk_aversion_candidates:
                current += 1
                
                results = self._run_backtest_on_period(
                    self.val_df,
                    ewma_lambda=lam,
                    risk_aversion=ra,
                    init_periods=init_periods
                )
                
                self.tuning_results.append({
                    'lambda': lam,
                    'risk_aversion': ra,
                    'sharpe_m1': results['sharpe_m1'],
                    'sharpe_m2': results['sharpe_m2'],
                    'sharpe_m3': results['sharpe_m3'],
                    'vol_m3': results['vol_m3']
                })
                
                # Select best based on M3 Sharpe (our main strategy)
                if not np.isnan(results['sharpe_m3']) and results['sharpe_m3'] > best_sharpe:
                    best_sharpe = results['sharpe_m3']
                    self.best_lambda = lam
                    self.best_risk_aversion = ra
                
                if current % 5 == 0:
                    print(f"  Progress: {current}/{total_combinations} combinations tested...")
        
        print(f"\n✓ Hyperparameter Tuning Complete!")
        print(f"  Best Lambda: {self.best_lambda}")
        print(f"  Best Risk Aversion: {self.best_risk_aversion}")
        print(f"  Validation M3 Sharpe: {best_sharpe:.2f}")
        
        # Store validation results with best params
        self.val_results = self._run_backtest_on_period(
            self.val_df,
            ewma_lambda=self.best_lambda,
            risk_aversion=self.best_risk_aversion,
            init_periods=init_periods
        )
        
        print(f"\nValidation Period Performance (with tuned params):")
        print(f"  M1 (Delta Hedge): Sharpe = {self.val_results['sharpe_m1']:.2f}")
        print(f"  M2 (EWMA Hedge):  Sharpe = {self.val_results['sharpe_m2']:.2f}")
        print(f"  M3 (MV Optimal):  Sharpe = {self.val_results['sharpe_m3']:.2f}")
        
        return self
    
    def test(self, init_periods: int = 20):
        """
        Phase 3: Testing
        
        Final out-of-sample evaluation on test data.
        Uses hyperparameters tuned during validation.
        NO PARAMETER CHANGES ALLOWED IN THIS PHASE.
        """
        print("\n" + "=" * 60)
        print("PHASE 3: TESTING (2024-07-01 to 2025-12-03)")
        print("=" * 60)
        print("Running final out-of-sample evaluation...")
        print(f"  Using tuned Lambda: {self.best_lambda}")
        print(f"  Using tuned Risk Aversion: {self.best_risk_aversion}")
        
        self.test_results = self._run_backtest_on_period(
            self.test_df,
            ewma_lambda=self.best_lambda,
            risk_aversion=self.best_risk_aversion,
            init_periods=init_periods
        )
        
        print(f"\n" + "=" * 60)
        print("FINAL TEST RESULTS (Out-of-Sample)")
        print("=" * 60)
        
        # Create summary table
        print(f"\n{'Metric':<25} {'M1: Delta Hedge':>15} {'M2: EWMA Hedge':>15} {'M3: MV Optimal':>15}")
        print("-" * 75)
        print(f"{'Annualized Return':<25} {self.test_results['return_m1']*100:>14.2f}% {self.test_results['return_m2']*100:>14.2f}% {self.test_results['return_m3']*100:>14.2f}%")
        print(f"{'Annualized Volatility':<25} {self.test_results['vol_m1']*100:>14.2f}% {self.test_results['vol_m2']*100:>14.2f}% {self.test_results['vol_m3']*100:>14.2f}%")
        print(f"{'Sharpe Ratio':<25} {self.test_results['sharpe_m1']:>15.2f} {self.test_results['sharpe_m2']:>15.2f} {self.test_results['sharpe_m3']:>15.2f}")
        print(f"{'Max Drawdown':<25} {self.test_results['max_dd_m1']*100:>14.2f}% {self.test_results['max_dd_m2']*100:>14.2f}% {self.test_results['max_dd_m3']*100:>14.2f}%")
        
        # Store for plotting
        self.test_pnl = self.test_results.get('pnl_df')
        self.test_weights = self.test_results.get('weights_df')
        
        return self
    
    def run_full_pipeline(self,
                          lambda_candidates: List[float] = None,
                          risk_aversion_candidates: List[float] = None,
                          init_periods: int = 20):
        """
        Run the complete train/validation/test pipeline.
        """
        if lambda_candidates is None:
            lambda_candidates = [0.90, 0.92, 0.94, 0.96, 0.97]
        if risk_aversion_candidates is None:
            risk_aversion_candidates = [0.0, 1.0, 2.0, 5.0, 10.0]
        
        self.load_data()
        self.train(init_periods)
        self.validate(lambda_candidates, risk_aversion_candidates, init_periods)
        self.test(init_periods)
        
        return self
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table comparing all periods."""
        data = []
        
        if self.train_results:
            data.append({
                'Period': 'Training (2022-2023)',
                'M1 Sharpe': self.train_results.get('sharpe_m1', np.nan),
                'M2 Sharpe': self.train_results.get('sharpe_m2', np.nan),
                'M3 Sharpe': self.train_results.get('sharpe_m3', np.nan),
                'M3 Vol': self.train_results.get('vol_m3', np.nan),
                'M3 Return': self.train_results.get('return_m3', np.nan),
            })
        
        if self.val_results:
            data.append({
                'Period': 'Validation (2024 H1)',
                'M1 Sharpe': self.val_results.get('sharpe_m1', np.nan),
                'M2 Sharpe': self.val_results.get('sharpe_m2', np.nan),
                'M3 Sharpe': self.val_results.get('sharpe_m3', np.nan),
                'M3 Vol': self.val_results.get('vol_m3', np.nan),
                'M3 Return': self.val_results.get('return_m3', np.nan),
            })
        
        if self.test_results:
            data.append({
                'Period': 'Test (2024 H2 - 2025)',
                'M1 Sharpe': self.test_results.get('sharpe_m1', np.nan),
                'M2 Sharpe': self.test_results.get('sharpe_m2', np.nan),
                'M3 Sharpe': self.test_results.get('sharpe_m3', np.nan),
                'M3 Vol': self.test_results.get('vol_m3', np.nan),
                'M3 Return': self.test_results.get('return_m3', np.nan),
            })
        
        return pd.DataFrame(data)
    
    def get_test_metrics_table(self) -> pd.DataFrame:
        """
        Get metrics table in format expected by plot_metrics_comparison.
        
        Returns test period metrics formatted like BacktestEngine.get_summary_table().
        """
        if not self.test_results:
            return pd.DataFrame()
        
        tr = self.test_results
        
        # Calculate additional metrics from P&L if available
        def calc_win_rate(pnl_series):
            if pnl_series is None or len(pnl_series.dropna()) == 0:
                return 0.0
            return (pnl_series.dropna() > 0).mean()
        
        def calc_var(pnl_series, confidence=0.95):
            if pnl_series is None or len(pnl_series.dropna()) == 0:
                return 0.0
            return -np.percentile(pnl_series.dropna(), (1 - confidence) * 100) / self.notional
        
        pnl_df = tr.get('pnl_df')
        m1_pnl = pnl_df['m1_pnl'] if pnl_df is not None else None
        m2_pnl = pnl_df['m2_pnl'] if pnl_df is not None else None
        m3_pnl = pnl_df['m3_pnl'] if pnl_df is not None else None
        
        summary = pd.DataFrame({
            'Metric': [
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                '95% VaR',
                'Win Rate'
            ],
            'M1: Delta Hedge': [
                f"{tr.get('return_m1', 0):.2%}",
                f"{tr.get('vol_m1', 0):.2%}",
                f"{tr.get('sharpe_m1', 0):.2f}",
                f"{tr.get('max_dd_m1', 0):.2%}",
                f"{calc_var(m1_pnl, 0.95):.4f}",
                f"{calc_win_rate(m1_pnl):.2%}"
            ],
            'M2: EWMA Hedge': [
                f"{tr.get('return_m2', 0):.2%}",
                f"{tr.get('vol_m2', 0):.2%}",
                f"{tr.get('sharpe_m2', 0):.2f}",
                f"{tr.get('max_dd_m2', 0):.2%}",
                f"{calc_var(m2_pnl, 0.95):.4f}",
                f"{calc_win_rate(m2_pnl):.2%}"
            ],
            'M3: MV Optimal': [
                f"{tr.get('return_m3', 0):.2%}",
                f"{tr.get('vol_m3', 0):.2%}",
                f"{tr.get('sharpe_m3', 0):.2f}",
                f"{tr.get('max_dd_m3', 0):.2%}",
                f"{calc_var(m3_pnl, 0.95):.4f}",
                f"{calc_win_rate(m3_pnl):.2%}"
            ]
        })
        
        return summary
    
    def get_tuning_results(self) -> pd.DataFrame:
        """Get hyperparameter tuning results."""
        return pd.DataFrame(self.tuning_results)


# ============================================================================
# STANDALONE BACKTEST FUNCTION
# ============================================================================

def run_full_backtest(start_date: str = "2022-01-01",
                      end_date: str = None,
                      notional: float = 100000) -> Tuple[BacktestEngine, pd.DataFrame]:
    """
    Convenience function to run complete backtest.
    
    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date
    notional : float
        Notional value
    
    Returns
    -------
    Tuple[BacktestEngine, pd.DataFrame]
        Backtest engine and summary table
    """
    engine = BacktestEngine(start_date, end_date, notional)
    engine.load_data()
    engine.run_backtest()
    
    summary = engine.get_summary_table()
    
    return engine, summary


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BACKTEST: Mean-Variance Optimal Delta-Hedging of Short Strangles")
    print("=" * 70)
    
    # Run backtest
    engine, summary = run_full_backtest("2022-01-01", "2025-12-03", notional=100000)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("VOLATILITY COMPARISON BY PERIOD")
    print("=" * 70)
    vol_comparison = engine.get_volatility_comparison()
    print(vol_comparison.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("P&L STATISTICS")
    print("=" * 70)
    print(f"Total Unhedged P&L: ${engine.pnl_history['cum_unhedged'].iloc[-1]:,.2f}")
    print(f"Total Naive Hedged P&L: ${engine.pnl_history['cum_naive_hedged'].iloc[-1]:,.2f}")
    print(f"Total MV Hedged P&L: ${engine.pnl_history['cum_mv_hedged'].iloc[-1]:,.2f}")
    
    print("\n" + "=" * 70)
    print("SAMPLE WEIGHTS (Last 5 Days)")
    print("=" * 70)
    print(engine.weights_history[['mv_w_spot', 'mv_w_futures', 'mv_w_cash', 
                                   'naive_w_spot', 'naive_w_futures', 'naive_w_cash']].tail())

