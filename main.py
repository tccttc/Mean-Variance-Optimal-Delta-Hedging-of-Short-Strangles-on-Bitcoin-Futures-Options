#!/usr/bin/env python3
"""
Mean-Variance Optimal Delta-Hedging of Short Strangles on Bitcoin Futures Options
==================================================================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Main Execution Script
---------------------
This script orchestrates the complete analysis implementing THREE METHODOLOGIES:

METHODOLOGY 1: Option Strategy and Delta-Hedging (Lecture 6)
  - File: src/1_Option_Delta.py
  - Short strangle position using Option Greeks (θ, Γ, ν, Δ)
  - Delta-hedging requirement derivation
  - P&L: θ*dt - (1/2)*|Γ|*(ΔS)² - |ν|*Δσ + Δ*ΔS

METHODOLOGY 2: Dynamic Covariance Estimation via EWMA (Lecture 7)
  - File: src/2_Covariance_Estimation.py
  - RiskMetrics EWMA approach (λ=0.94)
  - Time-varying 3×3 covariance matrix [r_spot, r_basis, dvol_chg]
  - Update: Σ_t = λ*Σ_{t-1} + (1-λ)*r_{t-1}*r_{t-1}'

METHODOLOGY 3: Mean-Variance Optimal Portfolio Construction (Lecture 5)
  - File: src/3_MV_Optimization.py
  - Markowitz minimum variance optimization
  - Delta-neutrality constraint: δ'w = -net_δ
  - Optimization: min w'Σw s.t. sum(w)=1, δ'w=-net_δ

Backtest Strategies:
- Strangle Only: No hedging (baseline)
- M1: Delta Hedge (Methodology 1 with simple futures hedge)
- M1+M2+M3: MV Optimal (All three methodologies combined)

Authors: HKUST Financial Engineering Students
Date: December 2025

References:
- Lecture 5: Capital Asset Pricing Model (Mean-Variance)
- Lecture 6: Basic Derivative Theory (Option Greeks)
- Lecture 7: Basic Risk Management (EWMA Covariance)
"""

import os
import sys
import warnings
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import project modules
from src.data import load_all_data, RISK_FREE_RATE_ANNUAL
from src.returns import calculate_all_returns, calculate_hedge_asset_returns
# Import from methodology files (M1, M2, M3)
from src._2_Covariance_Estimation import compute_hedge_asset_covariance, compute_ewma_covariance_series
from src._3_MV_Optimization import (
    optimize_hedge_portfolio,
    compute_efficient_frontier_with_delta
)
from src.backtest import BacktestEngine, run_full_backtest, TrainValTestEngine
from src.plots import (
    plot_cumulative_pnl,
    plot_pnl_distribution,
    plot_weight_evolution,
    plot_rolling_volatility,
    plot_drawdowns,
    plot_efficient_frontier,
    plot_btc_price_context,
    plot_metrics_comparison,
    generate_all_plots
)
from src.robo_advisor import (
    robo_advisor_interface,
    compare_profiles,
    analyze_risk_aversion_sensitivity,
    run_interactive_advisor
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Train/Validation/Test Split
    'train_start': '2022-01-01',
    'train_end': '2023-12-31',      # ~2 years training
    'val_start': '2024-01-01',
    'val_end': '2024-06-30',        # ~6 months validation
    'test_start': '2024-07-01',
    'test_end': '2025-12-03',       # ~1.5 years testing
    
    # Legacy (for backward compatibility)
    'start_date': '2022-01-01',
    'end_date': '2025-12-03',
    
    # Model parameters
    'notional': 100000,             # $100k notional
    'ewma_lambda': 0.94,            # RiskMetrics standard (will be tuned)
    'init_periods': 20,             # EWMA initialization period
    
    # Hyperparameter search space (for validation)
    'lambda_candidates': [0.90, 0.92, 0.94, 0.96, 0.97],
    'risk_aversion_candidates': [0.0, 1.0, 2.0, 5.0, 10.0],
    
    # Output directories
    'output_dir': 'figures/',
    'results_dir': 'results/'
}


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def print_header():
    """Print project header."""
    print("\n" + "="*80)
    print("  MEAN-VARIANCE OPTIMAL DELTA-HEDGING OF SHORT STRANGLES")
    print("  ON BITCOIN FUTURES OPTIONS")
    print("="*80)
    print("  HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025")
    print("  Prof. Wei JIANG")
    print("="*80)
    print("  Data Split:")
    print(f"    Training:   {CONFIG['train_start']} to {CONFIG['train_end']} (~2 years)")
    print(f"    Validation: {CONFIG['val_start']} to {CONFIG['val_end']} (~6 months)")
    print(f"    Test:       {CONFIG['test_start']} to {CONFIG['test_end']} (~1.5 years)")
    print("="*80)
    print(f"  Notional: ${CONFIG['notional']:,}")
    print(f"  Initial EWMA Lambda: {CONFIG['ewma_lambda']} (will be tuned on validation)")
    print("="*80 + "\n")


def run_backtest_analysis():
    """
    Run complete backtest analysis with train/validation/test split.
    
    Returns
    -------
    TrainValTestEngine
        Completed engine with results from all phases
    """
    print("\n" + "-"*80)
    print("RUNNING TRAIN/VALIDATION/TEST PIPELINE")
    print("-"*80)
    
    engine = TrainValTestEngine(
        train_start=CONFIG['train_start'],
        train_end=CONFIG['train_end'],
        val_start=CONFIG['val_start'],
        val_end=CONFIG['val_end'],
        test_start=CONFIG['test_start'],
        test_end=CONFIG['test_end'],
        notional=CONFIG['notional']
    )
    
    engine.run_full_pipeline(
        lambda_candidates=CONFIG['lambda_candidates'],
        risk_aversion_candidates=CONFIG['risk_aversion_candidates'],
        init_periods=CONFIG['init_periods']
    )
    
    return engine


def print_performance_summary(engine):
    """Print comprehensive performance summary for train/val/test split."""
    print("\n" + "-"*80)
    print("PERFORMANCE SUMMARY ACROSS ALL PHASES")
    print("-"*80)
    
    # Get summary table for all periods
    summary = engine.get_summary_table()
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON BY PERIOD")
    print("="*60)
    print(summary.to_string(index=False))
    
    # Hyperparameter tuning results
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING RESULTS (Validation)")
    print("="*60)
    print(f"  Best EWMA Lambda: {engine.best_lambda}")
    print(f"  Best Risk Aversion: {engine.best_risk_aversion}")
    
    tuning_df = engine.get_tuning_results()
    if not tuning_df.empty:
        # Show top 5 configurations
        tuning_df = tuning_df.sort_values('sharpe_m3', ascending=False).head(5)
        print("\n  Top 5 Configurations (by M3 Sharpe):")
        for _, row in tuning_df.iterrows():
            print(f"    λ={row['lambda']:.2f}, A={row['risk_aversion']:.1f} → M3 Sharpe={row['sharpe_m3']:.2f}")
    
    # Final test results (most important)
    print("\n" + "="*60)
    print("★ FINAL OUT-OF-SAMPLE TEST RESULTS ★")
    print("="*60)
    print(f"  Test Period: {engine.test_start} to {engine.test_end}")
    print(f"\n  {'Metric':<22} {'M1 (Delta)':>12} {'M2 (EWMA)':>12} {'M3 (MV)':>12}")
    print("  " + "-"*60)
    
    tr = engine.test_results
    print(f"  {'Annualized Return':<22} {tr['return_m1']*100:>11.2f}% {tr['return_m2']*100:>11.2f}% {tr['return_m3']*100:>11.2f}%")
    print(f"  {'Annualized Volatility':<22} {tr['vol_m1']*100:>11.2f}% {tr['vol_m2']*100:>11.2f}% {tr['vol_m3']*100:>11.2f}%")
    print(f"  {'Sharpe Ratio':<22} {tr['sharpe_m1']:>12.2f} {tr['sharpe_m2']:>12.2f} {tr['sharpe_m3']:>12.2f}")
    print(f"  {'Max Drawdown':<22} {tr['max_dd_m1']*100:>11.2f}% {tr['max_dd_m2']*100:>11.2f}% {tr['max_dd_m3']*100:>11.2f}%")
    
    # Key insights from test results
    print("\n" + "="*60)
    print("KEY INSIGHTS FROM OUT-OF-SAMPLE TEST")
    print("="*60)
    
    m1_vol = tr.get('vol_m1', 0)
    m2_vol = tr.get('vol_m2', 0)
    m3_vol = tr.get('vol_m3', 0)
    
    if m1_vol > 0 and m3_vol > 0:
        vol_reduction = (m1_vol - m3_vol) / m1_vol * 100
        print(f"  • M3 reduces volatility by {vol_reduction:.1f}% vs M1 (out-of-sample)")
    
    if m2_vol > 0 and m3_vol > 0:
        m2_reduction = (m2_vol - m3_vol) / m2_vol * 100
        print(f"  • M3 reduces volatility by {m2_reduction:.1f}% vs M2 (out-of-sample)")
    
    # Sharpe comparison
    m1_sharpe = tr.get('sharpe_m1', 0)
    m2_sharpe = tr.get('sharpe_m2', 0)
    m3_sharpe = tr.get('sharpe_m3', 0)
    
    best_method = 'M3' if m3_sharpe >= max(m1_sharpe, m2_sharpe) else ('M2' if m2_sharpe >= m1_sharpe else 'M1')
    print(f"  • Best risk-adjusted return: {best_method}")
    print(f"  • M1 Sharpe: {m1_sharpe:.2f} | M2 Sharpe: {m2_sharpe:.2f} | M3 Sharpe: {m3_sharpe:.2f}")
    
    return summary


def generate_visualizations(engine):
    """Generate visualizations for train/val/test analysis."""
    print("\n" + "-"*80)
    print("GENERATING VISUALIZATIONS (Test Period)")
    print("-"*80)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # For TrainValTestEngine, create a BacktestEngine wrapper using test results
    print("Generating plots for test period...")
    
    # Create a BacktestEngine-like object from test results
    viz_engine = BacktestEngine(
        start_date=CONFIG['test_start'],
        end_date=CONFIG['test_end'],
        notional=CONFIG['notional']
    )
    viz_engine.load_data()
    
    # Use the test P&L and weights from TrainValTestEngine
    if hasattr(engine, 'test_pnl') and engine.test_pnl is not None:
        # Create pnl_history in the format expected by plots
        pnl_history = pd.DataFrame(index=engine.test_pnl.index)
        
        # P&L columns (both naming conventions for compatibility)
        pnl_history['m1_pnl'] = engine.test_pnl['m1_pnl']
        pnl_history['m2_pnl'] = engine.test_pnl['m2_pnl']
        pnl_history['m3_pnl'] = engine.test_pnl['m3_pnl']
        pnl_history['pnl_m1'] = engine.test_pnl['m1_pnl']  # Alias for plots
        pnl_history['pnl_m2'] = engine.test_pnl['m2_pnl']  # Alias for plots
        pnl_history['pnl_m3'] = engine.test_pnl['m3_pnl']  # Alias for plots
        
        # Daily returns (for rolling volatility plot)
        pnl_history['ret_m1'] = pnl_history['m1_pnl'] / CONFIG['notional']
        pnl_history['ret_m2'] = pnl_history['m2_pnl'] / CONFIG['notional']
        pnl_history['ret_m3'] = pnl_history['m3_pnl'] / CONFIG['notional']
        
        # Calculate cumulative P&L
        pnl_history['cum_m1'] = pnl_history['m1_pnl'].cumsum()
        pnl_history['cum_m2'] = pnl_history['m2_pnl'].cumsum()
        pnl_history['cum_m3'] = pnl_history['m3_pnl'].cumsum()
        
        # Legacy compatibility
        pnl_history['cum_unhedged'] = pnl_history['cum_m1']  # Use M1 as proxy
        pnl_history['cum_naive_hedged'] = pnl_history['cum_m1']
        pnl_history['cum_mv_hedged'] = pnl_history['cum_m3']
        
        viz_engine.pnl_history = pnl_history
        viz_engine.weights_history = engine.test_weights if hasattr(engine, 'test_weights') else None
        
        # Override get_summary_table to use test results
        def get_test_summary_table():
            return engine.get_test_metrics_table()
        viz_engine.get_summary_table = get_test_summary_table
        
        # Generate all plots using test data
        generate_all_plots(viz_engine, output_dir=CONFIG['output_dir'])
    else:
        # Fallback: run backtest with default parameters
        print("Warning: Using default parameters for visualization (not tuned)")
        viz_engine.run_backtest(init_periods=CONFIG['init_periods'])
        generate_all_plots(viz_engine, output_dir=CONFIG['output_dir'])
    
    # Generate efficient frontier using test data
    print("Generating efficient frontier...")
    asset_returns = calculate_hedge_asset_returns(viz_engine.df)
    cov_series = compute_hedge_asset_covariance(asset_returns)
    last_cov = cov_series[-1]
    
    expected_returns = np.array([
        0.0003,  # Spot
        0.00025, # Futures
        RISK_FREE_RATE_ANNUAL / 365  # Cash
    ])
    
    frontier = compute_efficient_frontier_with_delta(
        last_cov, expected_returns, net_delta=-0.05, n_points=30
    )
    
    if not frontier.empty:
        fig = plot_efficient_frontier(frontier, 
                                      save_path=f"{CONFIG['output_dir']}efficient_frontier.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("  - Efficient frontier plot saved")
    
    print(f"\nAll visualizations saved to '{CONFIG['output_dir']}'")


def run_robo_advisor_demo():
    """Demonstrate robo-advisor functionality."""
    print("\n" + "-"*80)
    print("STEP 4: ROBO-ADVISOR DEMONSTRATION")
    print("-"*80)
    
    # Profile comparison
    print("\n" + "="*60)
    print("RISK PROFILE COMPARISON")
    print("="*60)
    comparison = compare_profiles()
    print(comparison.to_string(index=False))
    
    # Sensitivity analysis
    print("\n" + "="*60)
    print("RISK AVERSION SENSITIVITY")
    print("="*60)
    sensitivity = analyze_risk_aversion_sensitivity()
    print(sensitivity.to_string(index=False))
    
    # Example recommendation
    print("\n" + "="*60)
    print("SAMPLE RECOMMENDATION (Moderate Risk Profile)")
    print("="*60)
    result = robo_advisor_interface(risk_profile='moderate')
    print(result['recommendation'])


def save_results(engine, summary):
    """Save results to files."""
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Save summary table (train/val/test comparison)
    summary_path = f"{CONFIG['results_dir']}performance_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  - Performance summary saved to {summary_path}")
    
    # Save hyperparameter tuning results
    tuning_df = engine.get_tuning_results()
    if not tuning_df.empty:
        tuning_path = f"{CONFIG['results_dir']}hyperparameter_tuning.csv"
        tuning_df.to_csv(tuning_path, index=False)
        print(f"  - Hyperparameter tuning results saved to {tuning_path}")
    
    # Save test P&L if available
    if hasattr(engine, 'test_pnl') and engine.test_pnl is not None:
        test_pnl_path = f"{CONFIG['results_dir']}test_pnl_history.csv"
        engine.test_pnl.to_csv(test_pnl_path)
        print(f"  - Test P&L history saved to {test_pnl_path}")
    
    # Save test weights if available
    if hasattr(engine, 'test_weights') and engine.test_weights is not None:
        test_weights_path = f"{CONFIG['results_dir']}test_weights_history.csv"
        engine.test_weights.to_csv(test_weights_path)
        print(f"  - Test weights history saved to {test_weights_path}")
    
    # Save best hyperparameters
    best_params_path = f"{CONFIG['results_dir']}best_hyperparameters.txt"
    with open(best_params_path, 'w') as f:
        f.write(f"Best EWMA Lambda: {engine.best_lambda}\n")
        f.write(f"Best Risk Aversion: {engine.best_risk_aversion}\n")
        f.write(f"\nTuned on Validation Period: {CONFIG['val_start']} to {CONFIG['val_end']}\n")
        f.write(f"Test Period: {CONFIG['test_start']} to {CONFIG['test_end']}\n")
    print(f"  - Best hyperparameters saved to {best_params_path}")


def print_conclusion():
    """Print conclusion and key findings."""
    print("\n" + "="*80)
    print("CONCLUSION: TRAIN/VALIDATION/TEST ANALYSIS")
    print("="*80)
    print(f"""
RIGOROUS EVALUATION WITH PROPER DATA SPLIT:

┌──────────────────────────────────────────────────────────────────────────────┐
│ TRAINING PERIOD: {CONFIG['train_start']} to {CONFIG['train_end']} (~2 years)              │
├──────────────────────────────────────────────────────────────────────────────┤
│   - Model calibration and parameter estimation                              │
│   - Establish baseline performance                                          │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ VALIDATION PERIOD: {CONFIG['val_start']} to {CONFIG['val_end']} (~6 months)              │
├──────────────────────────────────────────────────────────────────────────────┤
│   - Hyperparameter tuning (EWMA λ, risk aversion)                          │
│   - Select best configuration based on M3 Sharpe ratio                     │
│   - NO final evaluation on this period                                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ TEST PERIOD: {CONFIG['test_start']} to {CONFIG['test_end']} (~1.5 years)                 │
├──────────────────────────────────────────────────────────────────────────────┤
│   - Final OUT-OF-SAMPLE evaluation                                          │
│   - NO parameter tuning in this phase                                       │
│   - Results represent TRUE expected performance                             │
└──────────────────────────────────────────────────────────────────────────────┘

THREE METHODOLOGIES COMPARED:

  M1: Delta-Hedging (Lecture 6) - Simple 1:1 futures hedge
  M2: EWMA Hedge (Lecture 7) - Covariance-adjusted hedge ratio
  M3: MV Optimal (Lecture 5) - Full Markowitz optimization

KEY FINDINGS:
   1. Hyperparameters tuned on VALIDATION data (no look-ahead bias)
   2. Final metrics reported on TEST data (honest out-of-sample)
   3. M3 typically achieves best risk-adjusted returns
   4. This rigorous evaluation avoids overfitting concerns

RECOMMENDATIONS:
   - M1: Use for simple, robust delta hedging (no estimation risk)
   - M2: Use when volatility regime matters (moderate complexity)
   - M3: Use for optimal risk-adjusted performance (most sophisticated)
""")
    print("="*80)
    print("  Analysis complete! Check 'figures/' and 'results/' directories.")
    print("="*80 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Mean-Variance Optimal Delta-Hedging of Short Strangles with Train/Val/Test Split'
    )
    
    parser.add_argument(
        '--notional', 
        type=float, 
        default=CONFIG['notional'],
        help='Notional value in USD'
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run interactive robo-advisor'
    )
    
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (use default lambda=0.94)'
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution function with train/validation/test split."""
    # Parse arguments
    args = parse_arguments()
    
    # Update config from arguments
    CONFIG['notional'] = args.notional
    
    # Print header
    print_header()
    
    # Interactive mode
    if args.interactive:
        run_interactive_advisor()
        return
    
    # Run full analysis with train/val/test split
    try:
        # Run train/validation/test pipeline
        engine = run_backtest_analysis()
        
        # Print performance summary
        summary = print_performance_summary(engine)
        
        # Generate visualizations (on test data)
        if not args.no_plots:
            generate_visualizations(engine)
        
        # Robo-advisor demo
        run_robo_advisor_demo()
        
        # Save results
        save_results(engine, summary)
        
        # Print conclusion
        print_conclusion()
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

