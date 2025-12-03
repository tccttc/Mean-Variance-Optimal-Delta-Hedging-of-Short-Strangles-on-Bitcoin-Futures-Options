"""
Visualization Module
====================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module creates:
- Cumulative P&L curves
- Efficient frontier plots
- Performance comparison charts
- Risk metrics visualization

References:
- Lecture 4: Efficient frontier visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color scheme
COLORS = {
    'unhedged': '#e74c3c',      # Red
    'naive': '#3498db',          # Blue
    'mv_optimal': '#2ecc71',     # Green
    'spot': '#f39c12',           # Orange
    'futures': '#9b59b6',        # Purple
    'cash': '#1abc9c',           # Teal
    'frontier': '#34495e',       # Dark gray
    'highlight': '#e67e22'       # Bright orange
}


# ============================================================================
# CUMULATIVE P&L PLOTS
# ============================================================================

def plot_cumulative_pnl(pnl_history: pd.DataFrame,
                        title: str = "Cumulative P&L: Hedging Strategy Comparison",
                        figsize: tuple = (14, 8),
                        save_path: str = None) -> plt.Figure:
    """
    Plot cumulative P&L for all strategies.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with cumulative P&L columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Notes
    -----
    This is the main visualization for comparing hedging effectiveness.
    Lower variance in cumulative P&L indicates better hedging.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each strategy
    ax.plot(pnl_history.index, pnl_history['cum_unhedged'], 
            color=COLORS['unhedged'], linewidth=2, label='Unhedged', alpha=0.8)
    ax.plot(pnl_history.index, pnl_history['cum_naive_hedged'], 
            color=COLORS['naive'], linewidth=2, label='Naive Hedge (1:1 Futures)', alpha=0.8)
    ax.plot(pnl_history.index, pnl_history['cum_mv_hedged'], 
            color=COLORS['mv_optimal'], linewidth=2.5, label='MV Optimal Hedge', alpha=0.9)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_pnl_distribution(pnl_history: pd.DataFrame,
                          figsize: tuple = (14, 5),
                          save_path: str = None) -> plt.Figure:
    """
    Plot distribution of daily P&L for each strategy.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with daily P&L columns
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    strategies = [
        ('pnl_unhedged', 'Unhedged', COLORS['unhedged']),
        ('pnl_naive_hedged', 'Naive Hedge', COLORS['naive']),
        ('pnl_mv_hedged', 'MV Optimal', COLORS['mv_optimal'])
    ]
    
    for ax, (col, name, color) in zip(axes, strategies):
        data = pnl_history[col].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=50, density=True, alpha=0.6, color=color)
        data.plot.kde(ax=ax, color=color, linewidth=2)
        
        # Add VaR lines
        var_95 = np.percentile(data, 5)
        ax.axvline(var_95, color='red', linestyle='--', alpha=0.7, 
                   label=f'95% VaR: ${var_95:.0f}')
        
        # Stats
        mean = data.mean()
        std = data.std()
        ax.axvline(mean, color='black', linestyle='-', alpha=0.5, label=f'Mean: ${mean:.0f}')
        
        ax.set_title(f'{name}\nStd: ${std:.0f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Daily P&L ($)', fontsize=10)
        ax.legend(fontsize=8)
    
    plt.suptitle('Daily P&L Distribution by Strategy', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# EFFICIENT FRONTIER
# ============================================================================

def plot_efficient_frontier(frontier_df: pd.DataFrame,
                            current_portfolio: tuple = None,
                            figsize: tuple = (10, 8),
                            save_path: str = None) -> plt.Figure:
    """
    Plot the efficient frontier.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data with 'return' and 'volatility' columns
    current_portfolio : tuple, optional
        (volatility, return) of current portfolio to highlight
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Notes
    -----
    From Lecture 4: The efficient frontier shows the best possible
    risk-return tradeoff available to investors.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Annualize for display
    frontier_df = frontier_df.copy()
    frontier_df['ann_return'] = frontier_df['return'] * 365
    frontier_df['ann_vol'] = frontier_df['volatility'] * np.sqrt(365)
    
    # Plot frontier
    ax.plot(frontier_df['ann_vol'] * 100, frontier_df['ann_return'] * 100,
            color=COLORS['frontier'], linewidth=2.5, marker='o', markersize=4,
            label='Efficient Frontier (Delta-Neutral)')
    
    # Color by Sharpe ratio
    scatter = ax.scatter(frontier_df['ann_vol'] * 100, frontier_df['ann_return'] * 100,
                        c=frontier_df['sharpe'], cmap='RdYlGn', s=50, zorder=5)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Highlight current/optimal portfolio
    if current_portfolio:
        vol, ret = current_portfolio
        ax.scatter([vol * 100], [ret * 100], s=200, c=COLORS['highlight'],
                   marker='*', edgecolor='black', linewidth=1.5,
                   label='Current MV Optimal', zorder=10)
    
    # Formatting
    ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax.set_ylabel('Annualized Return (%)', fontsize=12)
    ax.set_title('Efficient Frontier (Subject to Delta-Neutral Constraint)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# WEIGHT EVOLUTION
# ============================================================================

def plot_weight_evolution(weights_history: pd.DataFrame,
                          figsize: tuple = (14, 6),
                          save_path: str = None) -> plt.Figure:
    """
    Plot evolution of portfolio weights over time.
    
    Parameters
    ----------
    weights_history : pd.DataFrame
        DataFrame with weight columns over time
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # MV Optimal weights
    ax1 = axes[0]
    ax1.stackplot(weights_history.index,
                  weights_history['mv_w_spot'],
                  weights_history['mv_w_futures'],
                  weights_history['mv_w_cash'],
                  labels=['Spot', 'Futures', 'Cash'],
                  colors=[COLORS['spot'], COLORS['futures'], COLORS['cash']],
                  alpha=0.8)
    ax1.set_ylabel('Weight', fontsize=11)
    ax1.set_title('MV Optimal Hedge Weights', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 1)
    
    # Naive weights
    ax2 = axes[1]
    ax2.stackplot(weights_history.index,
                  weights_history['naive_w_spot'],
                  weights_history['naive_w_futures'],
                  weights_history['naive_w_cash'],
                  labels=['Spot', 'Futures', 'Cash'],
                  colors=[COLORS['spot'], COLORS['futures'], COLORS['cash']],
                  alpha=0.8)
    ax2.set_ylabel('Weight', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title('Naive Hedge Weights', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# ROLLING VOLATILITY
# ============================================================================

def plot_rolling_volatility(pnl_history: pd.DataFrame,
                            window: int = 30,
                            figsize: tuple = (14, 6),
                            save_path: str = None) -> plt.Figure:
    """
    Plot rolling volatility comparison.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with P&L columns
    window : int
        Rolling window size
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate rolling volatility (annualized)
    roll_vol_unhedged = pnl_history['ret_unhedged'].rolling(window).std() * np.sqrt(365) * 100
    roll_vol_naive = pnl_history['ret_naive_hedged'].rolling(window).std() * np.sqrt(365) * 100
    roll_vol_mv = pnl_history['ret_mv_hedged'].rolling(window).std() * np.sqrt(365) * 100
    
    ax.plot(pnl_history.index, roll_vol_unhedged, 
            color=COLORS['unhedged'], linewidth=1.5, label='Unhedged', alpha=0.8)
    ax.plot(pnl_history.index, roll_vol_naive, 
            color=COLORS['naive'], linewidth=1.5, label='Naive Hedge', alpha=0.8)
    ax.plot(pnl_history.index, roll_vol_mv, 
            color=COLORS['mv_optimal'], linewidth=2, label='MV Optimal', alpha=0.9)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{window}-Day Rolling Volatility (% annualized)', fontsize=12)
    ax.set_title(f'Rolling Volatility Comparison ({window}-Day Window)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# DRAWDOWN ANALYSIS
# ============================================================================

def plot_drawdowns(pnl_history: pd.DataFrame,
                   figsize: tuple = (14, 6),
                   save_path: str = None) -> plt.Figure:
    """
    Plot drawdown analysis for each strategy.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with cumulative P&L columns
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col, name, color in [
        ('cum_unhedged', 'Unhedged', COLORS['unhedged']),
        ('cum_naive_hedged', 'Naive Hedge', COLORS['naive']),
        ('cum_mv_hedged', 'MV Optimal', COLORS['mv_optimal'])
    ]:
        cum_pnl = pnl_history[col]
        running_max = cum_pnl.cummax()
        drawdown = (running_max - cum_pnl)
        
        ax.fill_between(pnl_history.index, 0, -drawdown, 
                       color=color, alpha=0.4, label=name)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown ($)', fontsize=12)
    ax.set_title('Drawdown Analysis by Strategy', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# PERFORMANCE COMPARISON BAR CHART
# ============================================================================

def plot_metrics_comparison(metrics_df: pd.DataFrame,
                            figsize: tuple = (12, 8),
                            save_path: str = None) -> plt.Figure:
    """
    Create bar chart comparing key metrics across strategies.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics for each strategy
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    strategies = ['Unhedged', 'Naive Hedge', 'MV Optimal']
    colors = [COLORS['unhedged'], COLORS['naive'], COLORS['mv_optimal']]
    
    # Convert string percentages to floats if needed
    def parse_pct(val):
        if isinstance(val, str):
            return float(val.strip('%')) / 100
        return val
    
    # Sharpe Ratio
    ax1 = axes[0, 0]
    sharpe_vals = [
        float(metrics_df[metrics_df['Metric'] == 'Sharpe Ratio']['Unhedged'].values[0]),
        float(metrics_df[metrics_df['Metric'] == 'Sharpe Ratio']['Naive Hedge'].values[0]),
        float(metrics_df[metrics_df['Metric'] == 'Sharpe Ratio']['MV Optimal'].values[0])
    ]
    ax1.bar(strategies, sharpe_vals, color=colors)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Volatility
    ax2 = axes[0, 1]
    vol_vals = [
        parse_pct(metrics_df[metrics_df['Metric'] == 'Annualized Volatility']['Unhedged'].values[0]),
        parse_pct(metrics_df[metrics_df['Metric'] == 'Annualized Volatility']['Naive Hedge'].values[0]),
        parse_pct(metrics_df[metrics_df['Metric'] == 'Annualized Volatility']['MV Optimal'].values[0])
    ]
    ax2.bar(strategies, [v * 100 for v in vol_vals], color=colors)
    ax2.set_ylabel('Volatility (%)', fontsize=11)
    ax2.set_title('Annualized Volatility', fontsize=12, fontweight='bold')
    
    # Max Drawdown
    ax3 = axes[1, 0]
    dd_vals = [
        parse_pct(metrics_df[metrics_df['Metric'] == 'Max Drawdown']['Unhedged'].values[0]),
        parse_pct(metrics_df[metrics_df['Metric'] == 'Max Drawdown']['Naive Hedge'].values[0]),
        parse_pct(metrics_df[metrics_df['Metric'] == 'Max Drawdown']['MV Optimal'].values[0])
    ]
    ax3.bar(strategies, [d * 100 for d in dd_vals], color=colors)
    ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax3.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    
    # Win Rate
    ax4 = axes[1, 1]
    wr_vals = [
        parse_pct(metrics_df[metrics_df['Metric'] == 'Win Rate']['Unhedged'].values[0]),
        parse_pct(metrics_df[metrics_df['Metric'] == 'Win Rate']['Naive Hedge'].values[0]),
        parse_pct(metrics_df[metrics_df['Metric'] == 'Win Rate']['MV Optimal'].values[0])
    ]
    ax4.bar(strategies, [w * 100 for w in wr_vals], color=colors)
    ax4.set_ylabel('Win Rate (%)', fontsize=11)
    ax4.set_title('Daily Win Rate', fontsize=12, fontweight='bold')
    
    plt.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# BTC PRICE WITH EVENTS
# ============================================================================

def plot_btc_price_context(df: pd.DataFrame,
                           pnl_history: pd.DataFrame = None,
                           figsize: tuple = (14, 10),
                           save_path: str = None) -> plt.Figure:
    """
    Plot BTC price with key events and optionally overlay P&L.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with spot prices
    pnl_history : pd.DataFrame, optional
        P&L history to overlay
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                            gridspec_kw={'height_ratios': [2, 1]})
    
    # BTC Price
    ax1 = axes[0]
    ax1.plot(df.index, df['spot'], color=COLORS['spot'], linewidth=1.5)
    ax1.set_ylabel('BTC Price (USD)', fontsize=12)
    ax1.set_title('BTC Spot Price', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add key events
    events = [
        ('2022-05-09', 'Terra/Luna\nCollapse'),
        ('2022-11-08', 'FTX\nBankruptcy'),
        ('2024-01-10', 'BTC ETF\nApproved'),
    ]
    
    for date, label in events:
        try:
            if date in df.index.astype(str).tolist() or pd.Timestamp(date) in df.index:
                ax1.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.5)
                ax1.annotate(label, xy=(pd.Timestamp(date), df['spot'].max() * 0.9),
                           fontsize=8, ha='center', color='red')
        except:
            pass
    
    # DVOL
    ax2 = axes[1]
    ax2.plot(df.index, df['dvol'], color=COLORS['futures'], linewidth=1.5)
    ax2.set_ylabel('DVOL (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('BTC Implied Volatility (DVOL)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# GENERATE ALL PLOTS
# ============================================================================

def generate_all_plots(engine, output_dir: str = "figures/"):
    """
    Generate all standard plots from backtest results.
    
    Parameters
    ----------
    engine : BacktestEngine
        Completed backtest engine
    output_dir : str
        Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # 1. Cumulative P&L
    fig1 = plot_cumulative_pnl(engine.pnl_history, 
                               save_path=f"{output_dir}cumulative_pnl.png")
    plt.close(fig1)
    print("  - Cumulative P&L plot saved")
    
    # 2. P&L Distribution
    fig2 = plot_pnl_distribution(engine.pnl_history,
                                 save_path=f"{output_dir}pnl_distribution.png")
    plt.close(fig2)
    print("  - P&L distribution plot saved")
    
    # 3. Weight Evolution
    fig3 = plot_weight_evolution(engine.weights_history,
                                save_path=f"{output_dir}weight_evolution.png")
    plt.close(fig3)
    print("  - Weight evolution plot saved")
    
    # 4. Rolling Volatility
    fig4 = plot_rolling_volatility(engine.pnl_history,
                                   save_path=f"{output_dir}rolling_volatility.png")
    plt.close(fig4)
    print("  - Rolling volatility plot saved")
    
    # 5. Drawdowns
    fig5 = plot_drawdowns(engine.pnl_history,
                         save_path=f"{output_dir}drawdowns.png")
    plt.close(fig5)
    print("  - Drawdown plot saved")
    
    # 6. BTC Context
    fig6 = plot_btc_price_context(engine.df,
                                  save_path=f"{output_dir}btc_context.png")
    plt.close(fig6)
    print("  - BTC context plot saved")
    
    # 7. Metrics Comparison
    try:
        summary = engine.get_summary_table()
        fig7 = plot_metrics_comparison(summary,
                                       save_path=f"{output_dir}metrics_comparison.png")
        plt.close(fig7)
        print("  - Metrics comparison plot saved")
    except Exception as e:
        print(f"  - Skipped metrics comparison: {e}")
    
    print(f"All plots saved to {output_dir}")


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    from .backtest import run_full_backtest
    
    # Run backtest
    print("Running backtest for plot testing...")
    engine, summary = run_full_backtest("2022-01-01", "2025-12-03")
    
    # Generate all plots
    generate_all_plots(engine, output_dir="figures/")
    
    print("\nAll plots generated successfully!")

