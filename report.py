# report.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from risk import portfolio_volatility, portfolio_returns_series, var_historical


@dataclass
class PortfolioResult:
    name: str
    weights: pd.Series          # index=tickers
    allocation: pd.Series       # Capital allocation (aligned with weights)
    exp_return_annual: float    # Annualized expected return μ
    volatility_annual: float    # Annualized volatility σ
    var_alpha: float            # VaR confidence level (e.g., 0.05 = 95%)
    var_horizon_days: int       # VaR horizon in days (1=single day, >1=multi-period aggregation)
    var_value: float            # VaR loss proportion (positive value, e.g., 0.08 = 8%)
    sharpe: float               # Sharpe ratio ((μ - rf)/σ)


def evaluate_portfolio(
    name: str,
    tickers: list[str],
    weights: np.ndarray,
    capital: float,
    mu_annual: pd.Series,       # Forecasted annualized μ (index=tickers)
    Sigma_annual: pd.DataFrame, # Annualized covariance matrix Σ
    returns_daily: pd.DataFrame,# Daily returns matrix (recommended log returns, index=dates, columns=tickers)
    rf_annual: float = 0.0,     # Annualized risk-free rate
    var_alpha: float = 0.05,    # 95% VaR
    var_horizon_days: int = 1,  # 1-day VaR; can set to 5/21 for 1-week/1-month
    log_returns: bool = True,   # Keep True if returns_daily are log returns
) -> PortfolioResult:
    """
    Compute and summarize a portfolio's key risk/return metrics from weights and inputs.

    Parameters
    ----------
    name : str
        Portfolio label.
    tickers : list[str]
        Asset symbols (order must align with weights/columns).
    weights : np.ndarray
        Portfolio weights (typically sum to 1).
    capital : float
        Total capital to allocate; used to derive per-asset allocation.
    mu_annual : pd.Series
        Annualized expected returns μ for each ticker (index=tickers).
    Sigma_annual : pd.DataFrame
        Annualized covariance matrix Σ (rows/cols=tickers).
    returns_daily : pd.DataFrame
        Daily return matrix (index=dates, columns=tickers); use log returns if log_returns=True.
    rf_annual : float, default 0.0
        Annualized risk-free rate for Sharpe.
    var_alpha : float, default 0.05
        Tail probability for historical VaR (e.g., 0.05 → 95% VaR).
    var_horizon_days : int, default 1
        VaR holding period in trading days (e.g., 1, 5, 21).
    log_returns : bool, default True
        Whether `returns_daily` are log returns (affects multi-period aggregation in VaR).

    Returns
    -------
    PortfolioResult
        Structured summary with: weights, allocations, annualized expected return (μₚ),
        annualized volatility (σₚ), historical VaR, and Sharpe ratio.

    Notes
    -----
    - Assumes μ and Σ are annualized and consistent with each other.
    - Historical VaR is computed from the realized daily portfolio return series.
    """
    w = np.asarray(weights, dtype=float)
    w_series = pd.Series(w, index=tickers, name="weight")
    alloc = (capital * w_series).rename("allocation")

    # Annualized μ and σ (consistent with optimization scale)
    mu_p = float(mu_annual.loc[tickers].values @ w)                  # Portfolio annualized expected return
    sigma_p = float(portfolio_volatility(w, Sigma_annual))           # Portfolio annualized volatility

    # Portfolio daily returns (for historical VaR)
    port_ret = portfolio_returns_series(w, returns_daily)
    var_p = float(var_historical(port_ret, alpha=var_alpha,
                                 horizon_days=var_horizon_days, log_returns=log_returns))  # Loss proportion (positive)

    # Sharpe ratio
    sharpe = (mu_p - rf_annual) / sigma_p if sigma_p > 0 else float("nan")

    return PortfolioResult(
        name=name,
        weights=w_series,
        allocation=alloc,
        exp_return_annual=mu_p,
        volatility_annual=sigma_p,
        var_alpha=var_alpha,
        var_horizon_days=var_horizon_days,
        var_value=var_p,
        sharpe=sharpe,
    )


def compile_report(results: List[PortfolioResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate multiple PortfolioResult objects into summary, weights, and allocation tables.

    Parameters
    ----------
    results : List[PortfolioResult]
        Collection of evaluated portfolios to compare.

    Returns
    -------
    summary : pd.DataFrame
        One row per strategy with columns: ExpReturn(annual), Volatility(annual),
        VaR(<confidence>, <days>D), Sharpe (rounded for display).
    weights_table : pd.DataFrame
        Wide table of weights (rows=tickers, columns=strategy names).
    alloc_table : pd.DataFrame
        Wide table of capital allocations (rows=tickers, columns=strategy names).

    Notes
    -----
    - Rounds only the displayed `summary` values; underlying precision in `results` is unchanged.
    - Assumes each `PortfolioResult.weights`/`allocation` is indexed by ticker.
    """
    rows = []
    W = []
    A = []
    for r in results:
        rows.append({
            "Strategy": r.name,
            "ExpReturn(annual)": r.exp_return_annual,
            "Volatility(annual)": r.volatility_annual,
            f"VaR({int((1-r.var_alpha)*100)}%, {r.var_horizon_days}D)": r.var_value,
            "Sharpe": r.sharpe,
        })
        W.append(r.weights.rename(r.name))
        A.append(r.allocation.rename(r.name))

    summary = pd.DataFrame(rows).set_index("Strategy")
    weights_table = pd.DataFrame(W).T  # columns=strategies, rows=tickers
    alloc_table = pd.DataFrame(A).T

    # Round for nicer display (does not affect underlying precision, can be removed if not needed)
    summary_rounded = summary.copy()
    for col in summary_rounded.columns:
        summary_rounded[col] = summary_rounded[col].astype(float).round(6)

    return summary_rounded, weights_table, alloc_table
