#!/usr/bin/env python3
"""
usgs_collector.py

Standalone USGS NWIS data collector for atmospheric and snow measurements.
Focused on Senator Beck and Ptarmigan stations for snowpack modeling.

Usage:
    from usgs_collector import fetch_usgs_iv, STATIONS
    
    # Fetch data for Senator Beck
    df = fetch_usgs_iv(STATIONS["senator_beck"], "2025-01-01", "2025-01-10")
    print(df.head())
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
from typing import Optional, List, Dict, Union

# =============================================================================
# CONFIGURATION
# =============================================================================

USGS_IV_URL = "https://nwis.waterservices.usgs.gov/nwis/iv/"

# Target stations for snowpack analysis
STATIONS = {
    "senator_beck": "375429107433201",   # Senator Beck Basin Study Area
    "ptarmigan": "392954106162501",       # Ptarmigan Site
}

# =============================================================================
# PARAMETER CODES (verified from USGS API header 2026-02-05)
# =============================================================================
# These are the CONFIRMED parameter codes from Senator Beck / Ptarmigan stations

PARAM_CODES = {
    # Atmospheric forcing
    "air_temp_c": "00020",           # Temperature, air, degrees Celsius
    "rh_pct": "00052",               # Relative humidity, percent
    "wind_speed_mph": "00035",       # Wind speed, miles per hour
    "wind_dir_deg": "00036",         # Wind direction, degrees clockwise from true north
    "wind_gust_mph": "61728",        # Wind gust speed, air, miles per hour
    "pressure_mmhg": "00025",        # Barometric pressure, millimeters of mercury
    
    # Radiation
    "sw_down_wm2": "72186",          # Shortwave radiation, downward intensity, W/m²
    "sw_up_wm2": "72185",            # Shortwave radiation, upward intensity, W/m²
    "lw_down_wm2": "72175",          # Longwave radiation, downward intensity, W/m² (INCOMING)
    "lw_up_wm2": "72174",            # Longwave radiation, upward intensity, W/m² (OUTGOING)
    
    # Snow properties
    "snow_depth_m": "72189",         # Snow depth, meters
    "surface_temp_c": "72405",       # Surface temperature, non-contact, degrees Celsius (PRIMARY VALIDATION)
    "lwc_pct": "72393",              # Liquid water content, snowpack, percent of total volume
    "drifting_snow": "72394",        # Mass flux density of drifting snow particles, g/m²/s
    
    # Soil (bottom boundary)
    "soil_temp_c": "72253",          # Soil temperature, degrees Celsius (5cm and 20cm depths)
}

# Full parameter set for snowpack energy balance model
MODEL_PARAMS = [
    "00020",  # Air temp (forcing)
    "00052",  # RH (forcing)
    "00035",  # Wind speed (forcing)
    "00036",  # Wind direction (forcing)
    "00025",  # Pressure (forcing)
    "72186",  # SW down (forcing)
    "72185",  # SW up (forcing/validation)
    "72175",  # LW down (forcing)
    "72174",  # LW up (forcing/validation)
    "72189",  # Snow depth (state)
    "72405",  # Surface temp (PRIMARY VALIDATION)
    "72253",  # Soil temp (bottom boundary)
    "72393",  # LWC (filtering wet snow)
    "72394",  # Drifting snow (filtering)
    "61728",  # Wind gust (optional)
]

# Minimal set for quick testing
DEFAULT_PARAMS = MODEL_PARAMS

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def fetch_usgs_iv(
    site: str,
    start_date: str,
    end_date: str,
    param_codes: Optional[List[str]] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch instantaneous values (IV) from USGS NWIS Water Services.
    
    Parameters
    ----------
    site : str
        USGS site number (e.g., "375429107433201" for Senator Beck)
    start_date : str
        Start date in ISO format (YYYY-MM-DD)
    end_date : str
        End date in ISO format (YYYY-MM-DD)
    param_codes : list of str, optional
        Parameter codes to fetch. If None, uses DEFAULT_PARAMS.
    timeout : int
        Request timeout in seconds
        
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime (timezone-naive) with parameter columns.
        Column names are the raw parameter codes (e.g., "00020_00011").
        
    Raises
    ------
    requests.HTTPError
        If the API request fails
    ValueError
        If no valid data is returned
        
    Examples
    --------
    >>> df = fetch_usgs_iv("375429107433201", "2025-01-01", "2025-01-05")
    >>> print(df.columns.tolist())
    ['72405_00011']  # Surface temp in °C
    """
    if param_codes is None:
        param_codes = DEFAULT_PARAMS
    
    params = {
        "format": "rdb",
        "sites": site,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": ",".join(param_codes),
        "siteStatus": "all",
    }
    
    print(f"[USGS] Fetching site {site} from {start_date} to {end_date}...")
    print(f"[USGS] Parameters: {', '.join(param_codes)}")
    
    # Retry logic with exponential backoff
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(USGS_IV_URL, params=params, timeout=timeout)
            response.raise_for_status()
            break  # Success, exit retry loop
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                print(f"[USGS] Connection error, retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"[USGS] Failed after {max_retries} attempts")
                raise
    
    # Parse the RDB format response
    df = _parse_rdb_response(response.text)
    
    if df.empty:
        print(f"[USGS] Warning: No data returned for site {site}")
        return df
    
    print(f"[USGS] Retrieved {len(df)} records from {df.index.min()} to {df.index.max()}")
    return df


def _parse_rdb_response(text: str) -> pd.DataFrame:
    """
    Parse USGS RDB (tab-delimited) format response into a DataFrame.
    
    Handles:
    - Comment lines starting with #
    - Format specifier row (agency_cd == '5s')
    - DateTime parsing and timezone stripping
    - Numeric conversion with error coercion
    """
    # Read as tab-separated, skip comment lines
    df = pd.read_table(
        StringIO(text),
        sep="\t",
        comment="#",
        dtype=str,
        on_bad_lines="skip",
    )
    
    if df.empty:
        return df
    
    # Drop the RDB format specifier row (where agency_cd contains format info like '5s')
    if "agency_cd" in df.columns:
        # The format row typically has '5s' or similar in agency_cd
        mask = ~df["agency_cd"].str.match(r"^\d+s$", na=False)
        df = df[mask].copy()
    
    if df.empty:
        return df
    
    # Convert datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime").sort_index()
    
    # Identify non-numeric columns to preserve
    non_numeric_cols = {"agency_cd", "site_no", "tz_cd"}
    
    # Convert all other columns to numeric
    for col in df.columns:
        if col not in non_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def simplify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify USGS column names by extracting just the parameter code.
    
    USGS returns columns like '307890_00020' (internal_id + param_code).
    This function renames them to just the param code (e.g., '00020').
    Also creates human-readable column aliases.
    
    Handles duplicate parameter codes (e.g., soil temp at multiple depths)
    by appending an index suffix.
    
    Returns a copy of the DataFrame with simplified column names.
    """
    df = df.copy()
    
    # Columns to skip
    skip_cols = {"agency_cd", "site_no", "tz_cd"}
    
    # Reverse mapping for human-readable names
    code_to_name = {v: k for k, v in PARAM_CODES.items()}
    
    # Track how many times we've seen each param code
    param_count = {}
    
    new_columns = {}
    for col in df.columns:
        if col in skip_cols:
            continue
        
        # Extract the 5-digit param code from compound column name
        # Pattern: NNNNNN_PPPPP or NNNNNN_PPPPP_cd
        parts = col.split("_")
        if len(parts) >= 2:
            internal_id = parts[0]
            param_code = parts[1]
            
            # Skip quality code columns (end in '_cd')
            if parts[-1] == "cd":
                new_columns[col] = f"{param_code}_qc"
            else:
                # Use human name if available
                human_name = code_to_name.get(param_code, param_code)
                
                # Handle duplicates by adding suffix
                if human_name in param_count:
                    param_count[human_name] += 1
                    # Add suffix for duplicates (e.g., soil_temp_c_2)
                    human_name = f"{human_name}_{param_count[human_name]}"
                else:
                    param_count[human_name] = 1
                
                new_columns[col] = human_name
    
    # Rename columns
    df = df.rename(columns=new_columns)
    
    # Drop quality code columns and metadata columns
    cols_to_drop = [c for c in df.columns if c.endswith("_qc") or c in skip_cols]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    return df


def get_station_name(site: str) -> str:
    """Get human-readable name for a site ID."""
    reverse_lookup = {v: k for k, v in STATIONS.items()}
    return reverse_lookup.get(site, f"USGS {site}")


def fetch_all_stations(
    start_date: str,
    end_date: str,
    param_codes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data from all configured stations.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    param_codes : list, optional
        Parameter codes to fetch
        
    Returns
    -------
    dict
        Dictionary mapping station names to DataFrames
    """
    results = {}
    for name, site_id in STATIONS.items():
        try:
            df = fetch_usgs_iv(site_id, start_date, end_date, param_codes)
            results[name] = df
            print(f"[USGS] ✓ {name}: {len(df)} records")
        except Exception as e:
            print(f"[USGS] ✗ {name}: {e}")
            results[name] = pd.DataFrame()
    return results


def get_available_params(site: str, start_date: str, end_date: str) -> List[str]:
    """
    Discover which parameters are available for a given station and time range.
    
    Returns list of parameter code columns that have non-null data.
    """
    # Request all known params and see which ones return data
    df = fetch_usgs_iv(site, start_date, end_date, list(PARAM_CODES.values()))
    
    available = []
    for col in df.columns:
        if col not in ["agency_cd", "site_no", "tz_cd"]:
            if df[col].notna().any():
                available.append(col)
    
    return available


# =============================================================================
# UNIT CONVERSION UTILITIES
# =============================================================================

def convert_temp_f_to_c(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert Fahrenheit column to Celsius."""
    return (df[col] - 32) * 5 / 9


def convert_in_to_cm(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert inches column to centimeters."""
    return df[col] * 2.54


# =============================================================================
# MAIN - Example usage when run as script
# =============================================================================

if __name__ == "__main__":
    # Example: Fetch last 10 days of data from Senator Beck
    from datetime import datetime, timedelta
    
    end = datetime.now()
    start = end - timedelta(days=10)
    
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    
    print("=" * 70)
    print("USGS Data Collector - Test Run")
    print(f"Date range: {start_str} to {end_str}")
    print("=" * 70)
    
    # Test Senator Beck
    print("\n" + "-" * 70)
    print("SENATOR BECK (375429107433201)")
    print("-" * 70)
    df_beck_raw = fetch_usgs_iv(STATIONS["senator_beck"], start_str, end_str)
    if not df_beck_raw.empty:
        df_beck = simplify_columns(df_beck_raw)
        print(f"\nShape: {df_beck.shape}")
        print(f"Columns: {df_beck.columns.tolist()}")
        print(f"\nSample data (last 10 rows):")
        print(df_beck.tail(10).to_string())
    
    # Test Ptarmigan
    print("\n" + "-" * 70)
    print("PTARMIGAN (392954106162501)")
    print("-" * 70)
    df_ptar_raw = fetch_usgs_iv(STATIONS["ptarmigan"], start_str, end_str)
    if not df_ptar_raw.empty:
        df_ptar = simplify_columns(df_ptar_raw)
        print(f"\nShape: {df_ptar.shape}")
        print(f"Columns: {df_ptar.columns.tolist()}")
        print(f"\nSample data (last 10 rows):")
        print(df_ptar.tail(10).to_string())
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
