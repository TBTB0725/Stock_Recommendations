# data.py
from __future__ import annotations  # Improve performance

from typing import Iterable, Literal, Tuple # Literal for restraint the value. Tuple for restraint data type and length
import datetime as dt

import numpy as np
import pandas as pd

import yfinance as yf


# -----------------------------
# Public Constant
# -----------------------------
PriceFrame = pd.DataFrame
ReturnFrame = pd.DataFrame

ReturnMethod = Literal["simple", "log"]


# -----------------------------
# Get Data
# -----------------------------
def get_prices(
    tickers: Iterable[str],
    lookback_days: int = 252,
    *,  #keyword-only arguments separator
    end: dt.date | None = None,
    pad_ratio: float = 2.0,
    auto_business_align: bool = True,
    use_adjusted_close: bool = True,
) -> PriceFrame:
    """
    Download and return aligned closing prices (columns = tickers, rows = dates).

    Parameters
    ----------
    tickers : Iterable[str]
        List of stock symbols (e.g., ["AAPL", "MSFT"]).
    lookback_days : int, default 252
        Number of trading days to look back. The final DataFrame is trimmed
        to the most recent `lookback_days` rows.
    end : date | None
        End date for the data (defaults to today if None).
    pad_ratio : float, default 2.0
        Extra factor to fetch more calendar days to account for holidays or suspensions.
        For example, 252 * 1.3 ≈ 328 days will be requested.
    auto_business_align : bool, default True
        Whether to reindex to business-day frequency and forward-fill missing values.
    use_adjusted_close : bool, default True
        If True, use the "Adj Close" field; otherwise use the raw "Close" price.

    Returns
    -------
    prices : pd.DataFrame
        A DataFrame indexed by ascending DatetimeIndex with tickers as columns.
        Missing values are filled (if enabled) and only the most recent
        `lookback_days` rows are kept.
    """
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and str(t).strip()])) # Uppercase, remove spaces, remove duplicates
    
    if len(tickers) == 0:
        raise ValueError("tickers cannot be None")

    if lookback_days <= 5:
        raise ValueError("lookback_days too small")

    if end is None:
        end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback_days * pad_ratio)) # To ensure enough rows, grab more natural days forward

    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )

    col_key = "Adj Close" if use_adjusted_close else "Close"

    if isinstance(raw.columns, pd.MultiIndex): # Multi layer index or not
        if col_key not in raw.columns.levels[0]:
            raise KeyError(f"Did not find column '{col_key}'. Try use_adjusted_close=False。")
        prices = raw[col_key].copy()
    else:
        if col_key not in raw.columns:
            fallback = "Close"
            if fallback not in raw.columns:
                raise KeyError(f"Did not find column '{col_key}' or '{fallback}'.")
            col_key = fallback
        prices = raw[[col_key]].copy()
        prices.columns = tickers  # rename to ticker

    # Only keep what we want
    prices = prices.loc[:, [c for c in prices.columns if c in tickers]]

    # Throw away empty columns
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("No valid price data for the provided tickers.")

    # Align to the weekday frequency and fill forward (use the value before suspension/holiday)
    if auto_business_align:
        prices = (
            prices.sort_index()
                  .asfreq("B")    # business day
                  .ffill()
        )
    else:
        prices = prices.sort_index().copy()

    # Only keep the most recent lookback_days rows (if not enough, an error will be thrown)
    if prices.shape[0] < lookback_days:
        raise ValueError(
            f"Data does not reach {lookback_days} rows, only {prices.shape[0]} rows."
            "Increase pad_ratio or decrease lookback_days。"
        )
    prices = prices.iloc[-lookback_days:].copy()

    # Ensure again
    prices = prices.dropna(axis=0, how="any")
    if prices.shape[0] < lookback_days // 2:
        raise ValueError("Data is not enough. Please decrease Lookback Days.")

    return prices


# -----------------------------
# Yield Calculation
# -----------------------------
def to_returns(
    prices: PriceFrame,
    method: ReturnMethod = "log",
    *,
    dropna: bool = True,
) -> ReturnFrame:
    """
    Generate returns from price data (rows = dates, columns = tickers).

    Parameters
    ----------
    prices : pd.DataFrame
        Price table (columns = tickers, rows = dates, index as ascending DatetimeIndex).
    method : {'simple','log'}, default 'log'
        simple: r_t = P_t / P_{t-1} - 1
        log:    r_t = ln(P_t) - ln(P_{t-1})
    dropna : bool, default True
        Whether to drop the first row with NaN values.

    Returns
    -------
    returns : pd.DataFrame
        Return matrix.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index should be DatetimeIndex。")

    if method == "simple":
        rets = prices.pct_change()
    elif method == "log":
        if (prices <= 0).any().any():
            raise ValueError("There is non positive number, use method='simple'.")
        rets = np.log(prices).diff()
    else:
        raise ValueError("method only support 'simple' or 'log'。")

    return rets.dropna(how="any") if dropna else rets