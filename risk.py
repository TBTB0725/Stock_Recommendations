# risk.py

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

def cov_matrix(returns: pd.DataFrame, annualize: bool = True, trading_days_per_year: int = 252) -> pd.DataFrame:
    """
    Compute the assets' return covariance matrix Σ and optionally annualize it.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns with rows=dates and columns=tickers (log returns recommended).
    annualize : bool, default True
        If True, scale daily covariance by `trading_days_per_year`.
    trading_days_per_year : int, default 252
        Annualization factor for markets using ~252 trading days.

    Returns
    -------
    pd.DataFrame
        Covariance matrix Σ (annualized if requested), indexed by tickers.
    """
    Sigma = returns.cov()
    return Sigma * trading_days_per_year if annualize else Sigma


def portfolio_volatility(weights: np.ndarray, Sigma: pd.DataFrame) -> float:
    """
    Compute portfolio volatility sqrt(wᵀ Σ w).

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights vector (length n, aligned to Σ's order).
    Sigma : pd.DataFrame
        Covariance matrix Σ (annualized → output is annualized volatility).

    Returns
    -------
    float
        Portfolio volatility (same frequency scale as Σ).
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ Sigma.values @ w))


def portfolio_returns_series(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
    """
    Collapse a multi-asset return matrix into a single portfolio return series via weights.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights vector (length n, aligned to `returns` columns).
    returns : pd.DataFrame
        Daily returns matrix (index=dates, columns=tickers); typically log returns.

    Returns
    -------
    pd.Series
        Daily portfolio return series indexed by date (name='portfolio_ret').
    """
    w = np.asarray(weights, dtype=float)
    r = returns.values @ w
    return pd.Series(r, index=returns.index, name="portfolio_ret")


def var_historical(port_ret_series: pd.Series, alpha: float = 0.05, horizon_days: int = 1, log_returns: bool = True) -> float:
    """
    Historical (non-parametric) VaR at tail probability `alpha`; returns a positive loss fraction.

    Parameters
    ----------
    port_ret_series : pd.Series
        Daily portfolio returns (log or simple, per `log_returns`).
    alpha : float, default 0.05
        Tail probability (0.05 → 95% VaR).
    horizon_days : int, default 1
        Holding period in days; if >1, aggregate returns over rolling windows.
    log_returns : bool, default True
        If True, aggregate multi-day returns by summing logs and converting with expm1;
        else aggregate simple returns via rolling product.

    Returns
    -------
    float
        VaR as a positive loss proportion (e.g., 0.03 for 3% loss).

    Notes
    -----
    - For `horizon_days=1`, VaR is `max(0, -quantile(r, alpha))`.
    - For multi-period, the series is aggregated first, then the same quantile logic is applied.
    """
    r = port_ret_series.dropna().astype(float)

    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    if len(r) < horizon_days:
        raise ValueError(f"horizon_days={horizon_days} exceeds available return length={len(r)}.")

    if horizon_days <= 1:
        q = np.quantile(r, alpha)
        return float(max(0.0, -q))

    if log_returns:
        roll_sum = r.rolling(horizon_days).sum().dropna()
        agg = np.expm1(roll_sum)
    else:
        agg = (1.0 + r).rolling(horizon_days).apply(np.prod, raw=True).dropna() - 1.0

    q = np.quantile(agg, alpha)
    return float(max(0.0, -q))
