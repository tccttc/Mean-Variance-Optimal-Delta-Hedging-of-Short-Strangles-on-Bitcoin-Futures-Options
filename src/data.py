"""
Data Fetching and Cleaning Module
=================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module handles:
- BTC spot price fetching from yfinance
- BTC quarterly futures fetching (or simulation from spot + basis)
- DVOL (Deribit Volatility Index) loading
- Options Greeks simulation (net delta for OTM strangle)
- Data cleaning: NaN handling, winsorization at 5%

References:
- Lecture 2: Asset returns and risk measures
- Lecture 4: Mean-Variance Analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import mstats
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS
# ============================================================================

# Risk-free rate: 5% annualized (USDT staking proxy)
RISK_FREE_RATE_ANNUAL = 0.05
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 365

# EWMA decay factor (RiskMetrics standard)
EWMA_LAMBDA = 0.94

# Winsorization percentile
WINSOR_PERCENTILE = 0.05

# Default net delta for 10% OTM short strangle
# Short call delta ≈ +0.20, Short put delta ≈ -0.25 → Net ≈ -0.05
DEFAULT_NET_DELTA = -0.05


# ============================================================================
# BTC SPOT DATA
# ============================================================================

def fetch_btc_spot(start_date: str = "2022-01-01", 
                   end_date: str = None) -> pd.DataFrame:
    """
    Fetch BTC-USD daily close prices from yfinance.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' index and 'spot' column
    
    Notes
    -----
    From Lecture 2: We use daily closing prices as our price series
    for computing log returns.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching BTC-USD spot data from {start_date} to {end_date}...")
    
    btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    spot_df = pd.DataFrame({
        'spot': btc['Close'].values.flatten()
    }, index=btc.index)
    
    spot_df.index.name = 'Date'
    
    print(f"  -> Retrieved {len(spot_df)} daily observations")
    return spot_df


# ============================================================================
# BTC FUTURES DATA
# ============================================================================

def fetch_btc_futures(start_date: str = "2022-01-01",
                      end_date: str = None) -> pd.DataFrame:
    """
    Fetch BTC futures data from yfinance (BTC=F - CME Bitcoin Futures).
    
    If data is unavailable or sparse, simulate futures from spot + basis.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' index and 'futures' column
    
    Notes
    -----
    Deribit quarterly futures typically trade at a premium to spot
    (contango) due to funding rates and demand for leverage.
    Historical basis for BTC: ~5-15% annualized in normal markets.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching BTC futures data from {start_date} to {end_date}...")
    
    try:
        futures = yf.download("BTC=F", start=start_date, end=end_date, progress=False)
        
        if isinstance(futures.columns, pd.MultiIndex):
            futures.columns = futures.columns.get_level_values(0)
        
        if len(futures) > 100:  # Sufficient data available
            futures_df = pd.DataFrame({
                'futures': futures['Close'].values.flatten()
            }, index=futures.index)
            futures_df.index.name = 'Date'
            print(f"  -> Retrieved {len(futures_df)} futures observations from yfinance")
            return futures_df
    except Exception as e:
        print(f"  -> yfinance futures fetch failed: {e}")
    
    # Fallback: Simulate futures from spot + synthetic basis
    print("  -> Using synthetic futures (spot + simulated basis)")
    return None  # Will be handled in combine_data


def simulate_futures_from_spot(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate quarterly futures prices from spot + synthetic basis.
    
    The basis is modeled as:
    - Base annualized premium: 8% (historical average for BTC)
    - Mean-reverting noise around the base
    
    Parameters
    ----------
    spot_df : pd.DataFrame
        DataFrame with 'spot' column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'futures' column
    
    Notes
    -----
    Basis = (F - S) / S ≈ r * T (cost of carry model)
    For BTC, basis includes: risk-free rate + funding premium - convenience yield
    """
    np.random.seed(42)  # Reproducibility
    
    # Parameters for synthetic basis
    base_annual_premium = 0.08  # 8% annualized
    mean_reversion_speed = 0.1
    volatility = 0.02
    
    n_days = len(spot_df)
    
    # Simulate mean-reverting basis
    basis = np.zeros(n_days)
    basis[0] = base_annual_premium / 365 * 90  # Approx 90 days to expiry
    
    for t in range(1, n_days):
        # Ornstein-Uhlenbeck process
        target = base_annual_premium / 365 * 90
        basis[t] = basis[t-1] + mean_reversion_speed * (target - basis[t-1]) + \
                   volatility * np.random.randn() * np.sqrt(1/365)
        basis[t] = max(basis[t], -0.05)  # Floor at -5% (backwardation limit)
        basis[t] = min(basis[t], 0.20)   # Cap at 20%
    
    futures_prices = spot_df['spot'].values * (1 + basis)
    
    futures_df = pd.DataFrame({
        'futures': futures_prices
    }, index=spot_df.index)
    
    print(f"  -> Simulated {len(futures_df)} futures prices")
    return futures_df


# ============================================================================
# DVOL (DERIBIT VOLATILITY INDEX)
# ============================================================================

def fetch_dvol(start_date: str = "2022-01-01",
               end_date: str = None) -> pd.DataFrame:
    """
    Load DVOL (Deribit BTC Volatility Index) data.
    
    Attempts to load from local CSV first, then simulates if unavailable.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' index and 'dvol' column
    
    Notes
    -----
    DVOL is Deribit's implied volatility index for BTC options,
    similar to VIX for S&P 500. It represents 30-day expected volatility.
    Historical range: ~40% to ~150%+ during stress periods.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Loading DVOL data from {start_date} to {end_date}...")
    
    # Try loading from local file
    try:
        dvol_df = pd.read_csv("Deribit_BTC_DVOL_daily.csv", parse_dates=['Date'])
        dvol_df.set_index('Date', inplace=True)
        dvol_df = dvol_df.loc[start_date:end_date]
        if len(dvol_df) > 100:
            print(f"  -> Loaded {len(dvol_df)} DVOL observations from CSV")
            return dvol_df[['dvol']]
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  -> CSV load failed: {e}")
    
    # Simulate DVOL if not available
    print("  -> Simulating DVOL (no local data found)")
    return None  # Will be handled in combine_data


def simulate_dvol(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate DVOL from spot price dynamics.
    
    Uses realized volatility + mean-reversion to long-term average.
    
    Parameters
    ----------
    spot_df : pd.DataFrame
        DataFrame with 'spot' column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'dvol' column (in percentage points, e.g., 80 = 80%)
    
    Notes
    -----
    DVOL characteristics:
    - Long-term mean: ~65-75%
    - Spikes during market stress (FTX crash: >120%)
    - Positively correlated with spot price drops (leverage effect)
    """
    np.random.seed(43)  # Reproducibility
    
    # Calculate 20-day realized volatility as base
    log_returns = np.log(spot_df['spot'] / spot_df['spot'].shift(1))
    realized_vol = log_returns.rolling(20).std() * np.sqrt(365) * 100
    
    # Parameters
    long_term_mean = 70  # 70% implied volatility
    mean_reversion = 0.05
    vol_of_vol = 5
    
    n_days = len(spot_df)
    dvol = np.zeros(n_days)
    dvol[0] = 70  # Start at long-term mean
    
    for t in range(1, n_days):
        # Mean-reverting process with realized vol influence
        rv = realized_vol.iloc[t] if not np.isnan(realized_vol.iloc[t]) else long_term_mean
        target = 0.7 * long_term_mean + 0.3 * rv
        
        # Leverage effect: vol increases when price drops
        spot_return = log_returns.iloc[t] if not np.isnan(log_returns.iloc[t]) else 0
        leverage_effect = -50 * spot_return  # Negative correlation
        
        dvol[t] = dvol[t-1] + mean_reversion * (target - dvol[t-1]) + \
                  leverage_effect + vol_of_vol * np.random.randn()
        
        dvol[t] = max(dvol[t], 30)   # Floor at 30%
        dvol[t] = min(dvol[t], 200)  # Cap at 200%
    
    dvol_df = pd.DataFrame({
        'dvol': dvol
    }, index=spot_df.index)
    
    print(f"  -> Simulated {len(dvol_df)} DVOL observations")
    return dvol_df


# ============================================================================
# OPTIONS GREEKS SIMULATION
# ============================================================================

def simulate_strangle_delta(spot_df: pd.DataFrame,
                           strike_otm_pct: float = 0.10) -> pd.DataFrame:
    """
    Simulate net delta for a 10% OTM short strangle.
    
    Short strangle = Short OTM call + Short OTM put
    
    Parameters
    ----------
    spot_df : pd.DataFrame
        DataFrame with 'spot' column
    strike_otm_pct : float
        OTM percentage for strikes (default 10%)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'net_delta' column
    
    Notes
    -----
    For a short strangle:
    - Short call delta: negative exposure (we're short a positive delta)
    - Short put delta: positive exposure (we're short a negative delta)
    - Net delta for 10% OTM: typically small, around -0.05 to 0.05
    
    The delta changes as spot moves relative to strikes:
    - Spot up → call delta increases (more negative exposure)
    - Spot down → put delta increases (more positive exposure)
    
    From Lecture 3: Greek letters measure option sensitivities
    """
    np.random.seed(44)
    
    n_days = len(spot_df)
    net_delta = np.zeros(n_days)
    
    # Calculate log returns for delta dynamics
    log_returns = np.log(spot_df['spot'] / spot_df['spot'].shift(1)).fillna(0)
    
    # Initialize with typical OTM strangle delta
    net_delta[0] = DEFAULT_NET_DELTA
    
    for t in range(1, n_days):
        # Delta changes based on spot movement
        # Gamma effect: delta becomes more negative when spot rises
        spot_return = log_returns.iloc[t]
        
        # Gamma for OTM options is lower, but still affects delta
        gamma_effect = -0.3 * spot_return  # Simplified gamma exposure
        
        # Mean reversion to base delta (roll effect / new positions)
        mean_reversion = 0.1 * (DEFAULT_NET_DELTA - net_delta[t-1])
        
        # Random noise (model uncertainty)
        noise = 0.01 * np.random.randn()
        
        net_delta[t] = net_delta[t-1] + gamma_effect + mean_reversion + noise
        
        # Bound delta to reasonable range for OTM strangle
        net_delta[t] = np.clip(net_delta[t], -0.30, 0.20)
    
    delta_df = pd.DataFrame({
        'net_delta': net_delta
    }, index=spot_df.index)
    
    print(f"  -> Simulated {len(delta_df)} strangle delta observations")
    return delta_df


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_data(df: pd.DataFrame, winsor_pct: float = WINSOR_PERCENTILE) -> pd.DataFrame:
    """
    Clean financial data: handle NaNs, winsorize outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data DataFrame
    winsor_pct : float
        Winsorization percentile (default 5%)
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    
    Notes
    -----
    From Lecture 2: Data quality is crucial for reliable risk estimates.
    Outliers can severely distort covariance estimates.
    
    Winsorization caps extreme values at specified percentiles,
    preserving data while reducing outlier influence.
    """
    df_clean = df.copy()
    
    # Forward fill NaNs (use previous day's value)
    df_clean = df_clean.ffill()
    
    # Backward fill any remaining NaNs at start
    df_clean = df_clean.bfill()
    
    # Winsorize each column
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            df_clean[col] = mstats.winsorize(df_clean[col], 
                                              limits=[winsor_pct, winsor_pct])
    
    return df_clean


# ============================================================================
# COMBINED DATA PIPELINE
# ============================================================================

def load_all_data(start_date: str = "2022-01-01",
                  end_date: str = None) -> pd.DataFrame:
    """
    Load and combine all required data for the backtesting engine.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        ['spot', 'futures', 'dvol', 'net_delta']
    
    Notes
    -----
    This function orchestrates all data loading:
    1. Fetch BTC spot from yfinance
    2. Fetch/simulate BTC futures
    3. Fetch/simulate DVOL
    4. Simulate strangle delta
    5. Clean and align all series
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 60)
    print("LOADING ALL DATA")
    print("=" * 60)
    
    # 1. BTC Spot
    spot_df = fetch_btc_spot(start_date, end_date)
    
    # 2. BTC Futures
    futures_df = fetch_btc_futures(start_date, end_date)
    if futures_df is None or len(futures_df) < len(spot_df) * 0.5:
        futures_df = simulate_futures_from_spot(spot_df)
    
    # 3. DVOL
    dvol_df = fetch_dvol(start_date, end_date)
    if dvol_df is None:
        dvol_df = simulate_dvol(spot_df)
    
    # 4. Strangle Delta
    delta_df = simulate_strangle_delta(spot_df)
    
    # 5. Combine all data
    combined = spot_df.copy()
    combined = combined.join(futures_df, how='left')
    combined = combined.join(dvol_df, how='left')
    combined = combined.join(delta_df, how='left')
    
    # 6. Clean data
    combined = clean_data(combined)
    
    # 7. Calculate basis (for later use)
    combined['basis'] = (combined['futures'] - combined['spot']) / combined['spot']
    
    print("=" * 60)
    print(f"FINAL DATASET: {len(combined)} observations")
    print(f"Date range: {combined.index[0]} to {combined.index[-1]}")
    print(f"Columns: {list(combined.columns)}")
    print("=" * 60)
    
    return combined


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test data loading
    df = load_all_data("2022-01-01", "2025-12-03")
    print("\nData Summary:")
    print(df.describe())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

