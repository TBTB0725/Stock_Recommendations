# risk.py

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

def cov_matrix(returns: pd.DataFrame, annualize: bool = True, trading_days_per_year: int = 252) -> pd.DataFrame:
    """
    Compute the covariance matrix Σ. Annualized by default.
    returns: rows = dates, columns = tickers (daily returns; log returns recommended)
    """
    Sigma = returns.cov()
    return Sigma * trading_days_per_year if annualize else Sigma


def portfolio_volatility(weights: np.ndarray, Sigma: pd.DataFrame) -> float:
    """
    Portfolio volatility σ(w) = sqrt(wᵀ Σ w)
    - If Σ is an annualized covariance matrix, the output is annualized volatility
    """
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ Sigma.values @ w))


def portfolio_returns_series(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
    """
    Combine a multi-asset returns matrix into a single portfolio daily-returns series using weights.
    `returns` is typically daily log returns (to match forecast.py)
    """
    w = np.asarray(weights, dtype=float)
    r = returns.values @ w
    return pd.Series(r, index=returns.index, name="portfolio_ret")


def var_historical(port_ret_series: pd.Series, alpha: float = 0.05, horizon_days: int = 1, log_returns: bool = True) -> float:
    """
    Historical VaR; returns a positive loss proportion.
    - alpha=0.05 → 95% VaR
    - When horizon_days > 1, supports multi-period aggregation:
        log_returns=True  → aggregate period log return = rolling sum of daily log returns → exp(sum)-1
        log_returns=False → aggregate period simple return = cumulative product of (1 + r) minus 1
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
        agg = np.expm1(roll_sum)  # convert to simple return
    else:
        agg = (1.0 + r).rolling(horizon_days).apply(np.prod, raw=True).dropna() - 1.0

    q = np.quantile(agg, alpha)
    return float(max(0.0, -q))
