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
import requests
import os

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
                   end_date: str = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch BTC-USD daily close prices from yfinance.
    
    Checks for cached CSV file first, downloads if not found or outdated.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    use_cache : bool
        Whether to use cached CSV file if available (default: True)
    
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
    
    # Use data folder for CSV storage
    os.makedirs("data", exist_ok=True)
    csv_path = "data/BTC_USD_spot.csv"
    
    print(f"Loading BTC-USD spot data from {start_date} to {end_date}...")
    
    # Try loading from cached CSV
    if use_cache and os.path.exists(csv_path):
        try:
            spot_df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            # Check if cached data covers the requested range (with some tolerance)
            cached_start = pd.Timestamp(spot_df.index.min())
            cached_end = pd.Timestamp(spot_df.index.max())
            request_start = pd.Timestamp(start_date)
            request_end = pd.Timestamp(end_date)
            
            # Use cached data if it covers at least 90% of the requested range
            date_range_days = (request_end - request_start).days
            
            if cached_start <= request_start and cached_end >= request_end:
                # Perfect coverage - use cached data
                spot_df = spot_df.loc[start_date:end_date]
                print(f"  -> Loaded {len(spot_df)} observations from cached CSV")
                return spot_df
            elif cached_start <= request_start and cached_end >= request_start + pd.Timedelta(days=date_range_days * 0.9):
                # Covers at least 90% - use cached data
                spot_df = spot_df.loc[start_date:min(end_date, cached_end.strftime('%Y-%m-%d'))]
                print(f"  -> Loaded {len(spot_df)} observations from cached CSV (covers {len(spot_df)/date_range_days*100:.1f}% of requested range)")
                return spot_df
            else:
                print(f"  -> Cached data incomplete (cached: {cached_start.date()} to {cached_end.date()}), fetching from yfinance...")
        except Exception as e:
            print(f"  -> CSV load failed: {e}, fetching from yfinance...")
    
    # Fetch from yfinance
    print(f"  -> Fetching from yfinance...")
    btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    spot_df = pd.DataFrame({
        'spot': btc['Close'].values.flatten()
    }, index=btc.index)
    
    spot_df.index.name = 'Date'
    
    # Save to CSV for future use
    if use_cache:
        try:
            spot_df.to_csv(csv_path)
            print(f"  -> Saved {len(spot_df)} observations to {csv_path}")
        except Exception as e:
            print(f"  -> Warning: Failed to save CSV: {e}")
    
    print(f"  -> Retrieved {len(spot_df)} daily observations")
    return spot_df


# ============================================================================
# BTC FUTURES DATA
# ============================================================================

def fetch_btc_futures(start_date: str = "2022-01-01",
                      end_date: str = None,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch BTC futures data from yfinance (BTC=F - CME Bitcoin Futures).
    
    Checks for cached CSV file first, downloads if not found or outdated.
    If data is unavailable or sparse, simulate futures from spot + basis.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        Whether to use cached CSV file if available (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' index and 'futures' column, or None if simulation needed
    
    Notes
    -----
    Deribit quarterly futures typically trade at a premium to spot
    (contango) due to funding rates and demand for leverage.
    Historical basis for BTC: ~5-15% annualized in normal markets.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Use data folder for CSV storage
    os.makedirs("data", exist_ok=True)
    csv_path = "data/BTC_F_futures.csv"
    
    print(f"Loading BTC futures data from {start_date} to {end_date}...")
    
    # Try loading from cached CSV
    if use_cache and os.path.exists(csv_path):
        try:
            futures_df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            # Check if cached data covers the requested range
            cached_start = futures_df.index.min()
            cached_end = futures_df.index.max()
            request_start = pd.Timestamp(start_date)
            request_end = pd.Timestamp(end_date)
            
            # Use cached data if it covers at least 90% of the requested range
            cached_start = pd.Timestamp(futures_df.index.min())
            cached_end = pd.Timestamp(futures_df.index.max())
            request_start = pd.Timestamp(start_date)
            request_end = pd.Timestamp(end_date)
            date_range_days = (request_end - request_start).days
            
            if cached_start <= request_start and cached_end >= request_end:
                # Perfect coverage - use cached data
                futures_df = futures_df.loc[start_date:end_date]
                if len(futures_df) > 100:  # Sufficient data
                    print(f"  -> Loaded {len(futures_df)} observations from cached CSV")
                    return futures_df
                else:
                    print(f"  -> Cached data insufficient ({len(futures_df)} obs), fetching from yfinance...")
            elif cached_start <= request_start and cached_end >= request_start + pd.Timedelta(days=date_range_days * 0.9):
                # Covers at least 90% - use cached data
                futures_df = futures_df.loc[start_date:min(end_date, cached_end.strftime('%Y-%m-%d'))]
                if len(futures_df) > 100:  # Sufficient data
                    print(f"  -> Loaded {len(futures_df)} observations from cached CSV (covers {len(futures_df)/date_range_days*100:.1f}% of requested range)")
                    return futures_df
                else:
                    print(f"  -> Cached data insufficient ({len(futures_df)} obs), fetching from yfinance...")
            else:
                print(f"  -> Cached data incomplete (cached: {cached_start.date()} to {cached_end.date()}), fetching from yfinance...")
        except Exception as e:
            print(f"  -> CSV load failed: {e}, fetching from yfinance...")
    
    # Fetch from yfinance
    try:
        print(f"  -> Fetching from yfinance...")
        futures = yf.download("BTC=F", start=start_date, end=end_date, progress=False)
        
        if isinstance(futures.columns, pd.MultiIndex):
            futures.columns = futures.columns.get_level_values(0)
        
        if len(futures) > 100:  # Sufficient data available
            futures_df = pd.DataFrame({
                'futures': futures['Close'].values.flatten()
            }, index=futures.index)
            futures_df.index.name = 'Date'
            
            # Save to CSV for future use
            if use_cache:
                try:
                    futures_df.to_csv(csv_path)
                    print(f"  -> Saved {len(futures_df)} observations to {csv_path}")
                except Exception as e:
                    print(f"  -> Warning: Failed to save CSV: {e}")
            
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

def download_dvol_csv(save_path: str = None) -> bool:
    """
    Download DVOL data from online source.
    
    Parameters
    ----------
    save_path : str, optional
        Path to save the CSV file. Defaults to 'data/Deribit_BTC_DVOL_daily.csv'
    
    Returns
    -------
    bool
        True if download successful, False otherwise
    
    Notes
    -----
    Sources tried:
    1. satochi.co/csv - Provides BTC volatility data
    2. Alternative: Manual download from cryptodatadownload.com
    
    The CSV should have columns: Date, dvol (or similar)
    """
    if save_path is None:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        save_path = "data/Deribit_BTC_DVOL_daily.csv"
    
    print(f"Attempting to download DVOL data...")
    
    # Try satochi.co API
    try:
        url = "https://satochi.co/csv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Save raw CSV
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"  -> Downloaded DVOL data from satochi.co")
            
            # Try to parse and reformat if needed
            try:
                df = pd.read_csv(save_path)
                # Check format and reformat if needed
                # satochi.co format: date, price, logprice, return, volatility, volatility60
                if 'volatility' in df.columns.str.lower().values:
                    # Reformat to our expected format
                    df_formatted = pd.DataFrame()
                    # Use 'date' column, convert to Date
                    if 'date' in df.columns.str.lower().values:
                        date_col = [c for c in df.columns if c.lower() == 'date'][0]
                        df_formatted['Date'] = pd.to_datetime(df[date_col])
                    else:
                        # Try first column as date
                        df_formatted['Date'] = pd.to_datetime(df.iloc[:, 0])
                    
                    # Use 'volatility' column, multiply by 100 to convert to percentage
                    vol_col = [c for c in df.columns if 'volatility' in c.lower()][0]
                    df_formatted['dvol'] = df[vol_col] * 100  # Convert to percentage
                    
                    # Save reformatted version
                    df_formatted.to_csv(save_path, index=False)
                    print(f"  -> Reformatted CSV (volatility -> dvol, converted to percentage)")
                    return True
                elif 'Date' in df.columns and 'dvol' in df.columns:
                    # Already in correct format
                    return True
                else:
                    print(f"  -> Warning: CSV format may need manual adjustment")
                    print(f"  -> Columns found: {list(df.columns)}")
                    return True
            except Exception as e:
                print(f"  -> Warning: Downloaded but parsing failed: {e}")
                return False
    except Exception as e:
        print(f"  -> Download failed: {e}")
        return False
    
    return False


def fetch_dvol(start_date: str = "2022-01-01",
               end_date: str = None,
               auto_download: bool = True) -> pd.DataFrame:
    """
    Load DVOL (Deribit BTC Volatility Index) data.
    
    Attempts to load from local CSV first, downloads if auto_download=True,
    then simulates if unavailable.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    auto_download : bool
        Whether to attempt automatic download if CSV not found
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' index and 'dvol' column
    
    Notes
    -----
    DVOL is Deribit's implied volatility index for BTC options,
    similar to VIX for S&P 500. It represents 30-day expected volatility.
    Historical range: ~40% to ~150%+ during stress periods.
    
    CSV Format Expected:
    - Column 'Date' (or 'date') with dates
    - Column 'dvol' (or 'DVOL', 'volatility') with volatility values
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Loading DVOL data from {start_date} to {end_date}...")
    
    # Use data folder for CSV storage
    os.makedirs("data", exist_ok=True)
    csv_path = "data/Deribit_BTC_DVOL_daily.csv"
    
    # Try loading from local file
    if os.path.exists(csv_path):
        try:
            dvol_df = pd.read_csv(csv_path, parse_dates=['Date'])
            dvol_df.set_index('Date', inplace=True)
            dvol_df = dvol_df.loc[start_date:end_date]
            if len(dvol_df) > 100:
                print(f"  -> Loaded {len(dvol_df)} DVOL observations from CSV")
                return dvol_df[['dvol']]
        except KeyError:
            # Try alternative column names
            try:
                dvol_df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
                # Look for dvol column (case insensitive)
                dvol_col = None
                for col in dvol_df.columns:
                    if 'dvol' in col.lower() or 'volatility' in col.lower():
                        dvol_col = col
                        break
                if dvol_col:
                    dvol_df = dvol_df.rename(columns={dvol_col: 'dvol'})
                    dvol_df = dvol_df.loc[start_date:end_date]
                    if len(dvol_df) > 100:
                        print(f"  -> Loaded {len(dvol_df)} DVOL observations from CSV")
                        return dvol_df[['dvol']]
            except Exception as e:
                print(f"  -> CSV format error: {e}")
        except Exception as e:
            print(f"  -> CSV load failed: {e}")
    
    # Try automatic download if enabled and file doesn't exist
    if auto_download and not os.path.exists(csv_path):
        print("  -> CSV file not found, attempting automatic download...")
        if download_dvol_csv(csv_path):
            # Retry loading after download
            try:
                dvol_df = pd.read_csv(csv_path, parse_dates=['Date'])
                dvol_df.set_index('Date', inplace=True)
                dvol_df = dvol_df.loc[start_date:end_date]
                if len(dvol_df) > 100:
                    print(f"  -> Loaded {len(dvol_df)} DVOL observations from downloaded CSV")
                    return dvol_df[['dvol']]
            except Exception as e:
                print(f"  -> Failed to parse downloaded CSV: {e}")
    
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
                  end_date: str = None,
                  auto_download_dvol: bool = True) -> pd.DataFrame:
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
        print(f"  -> Insufficient futures data ({len(futures_df) if futures_df is not None else 0} < {len(spot_df) * 0.5:.0f}), using simulation")
        futures_df = simulate_futures_from_spot(spot_df)
    else:
        print(f"  -> Using yfinance futures data ({len(futures_df)} observations, {len(futures_df)/len(spot_df)*100:.1f}% coverage)")
    
    # 3. DVOL
    dvol_df = fetch_dvol(start_date, end_date, auto_download=auto_download_dvol)
    if dvol_df is None:
        print("  -> No DVOL CSV found, using simulation")
        dvol_df = simulate_dvol(spot_df)
    else:
        print(f"  -> Using DVOL data from CSV ({len(dvol_df)} observations)")
    
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
    df.to_csv("data/Full_data.csv")

