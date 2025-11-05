# agent/tools.py
from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from data import get_prices, to_returns
from forecast import prophet_expected_returns
from risk import cov_matrix
from optimize import (
    solve_min_variance,
    solve_max_return,
    solve_max_sharpe,
    Constraints,
)
from report import evaluate_portfolio, compile_report, PortfolioResult
from news import fetch_finviz_headlines
from sentiment import score_titles as _score_titles

class ToolExecutionError(RuntimeError):
    """If a unified tool fails to execute, the outer layer can use this information to abort or retry."""
    
# ------------------------
# Log configuration
# ------------------------
logger = logging.getLogger("agent.tools")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ------------------------
# Timer + Log Decorator
# ------------------------
def _timed(tool_name: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"▶️  {tool_name} — start")
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"🛑  {tool_name} — failed: {type(e).__name__}: {e}")
                raise ToolExecutionError(f"{tool_name} failed") from e
            dur = (time.perf_counter() - start) * 1000.0
            logger.info(f"✅  {tool_name} — done in {dur:.1f} ms")
            return result
        return wrapper
    return deco

# ------------------------
# Utility functions
# ------------------------

# data.py
@_timed("fetch_prices")
def fetch_prices_tool(
    tickers: Iterable[str],
    lookback_days: int = 252,
    *,
    end: Optional[pd.Timestamp | Any] = None,
    pad_ratio: float = 2.0,
    auto_business_align: bool = True,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    return get_prices(
        tickers=tickers,
        lookback_days=lookback_days,
        end=end,
        pad_ratio=pad_ratio,
        auto_business_align=auto_business_align,
        use_adjusted_close=use_adjusted_close,
    )


@_timed("to_returns")
def to_returns_tool(
    prices: pd.DataFrame,
    method: str = "log",
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    return to_returns(prices=prices, method=method, dropna=dropna)


# forecast.py
@_timed("forecast_expected_returns")
def forecast_tool(
    prices: pd.DataFrame,
    horizon: str = "3M",
    *,
    tune: bool = False,
    cv_metric: str = "rmse",
    cv_initial_days: Optional[int] = None,
    cv_period_days: Optional[int] = None,
    param_grid: Optional[Dict[str, List]] = None,
    min_points_for_cv: int = 100,
) -> pd.Series:
    return prophet_expected_returns(
        prices=prices,
        horizon=horizon,
        tune=tune,
        cv_metric=cv_metric,
        cv_initial_days=cv_initial_days,
        cv_period_days=cv_period_days,
        param_grid=param_grid,
        min_points_for_cv=min_points_for_cv,
    )


# risk.py
@_timed("risk_cov_matrix")
def risk_tool(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    return cov_matrix(
        returns=returns,
        annualize=annualize,
        trading_days_per_year=trading_days_per_year,
    )


# optimize.py
@_timed("optimize_portfolio")
def optimize_tool(
    objective: str,
    *,
    mu_annual: Optional[pd.Series] = None,
    Sigma_annual: Optional[pd.DataFrame] = None,
    rf: float = 0.0,
    cons: Optional[Constraints] = None,
    eps: float = 1e-12,
    restarts: int = 0,
    seed: int = 0,
) -> np.ndarray:
    obj = objective.lower()
    if obj == "min_var":
        if Sigma_annual is None:
            raise ToolExecutionError("optimize(min_var) requires Sigma_annual")
        return solve_min_variance(Sigma_annual, cons=cons)

    if obj == "max_ret":
        if mu_annual is None:
            raise ToolExecutionError("optimize(max_ret) requires mu_annual")
        return solve_max_return(mu_annual, cons=cons)

    if obj == "max_sharpe":
        if mu_annual is None or Sigma_annual is None:
            raise ToolExecutionError("optimize(max_sharpe) requires mu_annual & Sigma_annual")
        return solve_max_sharpe(
            mu=mu_annual,
            Sigma=Sigma_annual,
            rf=rf,
            cons=cons,
            eps=eps,
            restarts=restarts,
            seed=seed,
        )

    raise ToolExecutionError(f"Unknown objective: {objective}")


# report.py
@_timed("evaluate_portfolio")
def evaluate_portfolio_tool(
    name: str,
    tickers: List[str],
    weights: np.ndarray,
    capital: float,
    mu_annual: pd.Series,
    Sigma_annual: pd.DataFrame,
    returns_daily: pd.DataFrame,
    *,
    rf_annual: float = 0.0,
    var_alpha: float = 0.05,
    var_horizon_days: int = 1,
    log_returns: bool = True,
) -> PortfolioResult:
    return evaluate_portfolio(
        name=name,
        tickers=tickers,
        weights=weights,
        capital=capital,
        mu_annual=mu_annual,
        Sigma_annual=Sigma_annual,
        returns_daily=returns_daily,
        rf_annual=rf_annual,
        var_alpha=var_alpha,
        var_horizon_days=var_horizon_days,
        log_returns=log_returns,
    )

@_timed("compile_report")
def compile_report_tool(
    results: List[PortfolioResult],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return compile_report(results)


# news.py
@_timed("fetch_news")
def fetch_news_tool(
    tickers: Iterable[str],
    lookback_days: int = 3,
    per_ticker_count: int = 15,
    final_cap: int = 200,
    sleep_s: float = 0.5,
) -> pd.DataFrame:
    return fetch_finviz_headlines(
        tickers=tickers,
        lookback_days=lookback_days,
        per_ticker_count=per_ticker_count,
        final_cap=final_cap,
        sleep_s=sleep_s,
    )

# sentiment.py
@_timed("score_titles")
def sentiment_score_titles_tool(
    items: Any,
    *,
    model_name: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    default_ticker: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    norm_items = _coerce_news_items(items, default_ticker=default_ticker, limit=limit)
    if not norm_items:
        logger.warning("sentiment_score_titles_tool: no items to score (empty input).")
        return []
    return _score_titles(items=norm_items, model_name=model_name, api_key=api_key)


# -----------------------------
# Help Functions
# -----------------------------
def _coerce_news_items(
    items: Any,
    *,
    default_ticker: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Normalize heterogeneous inputs into a list of {"ticker": str, "headline": str} dicts.

    Supported inputs
    ----------------
    - pd.DataFrame: must contain columns for ticker and headline
        * accepted aliases (case/space/underscore-insensitive):
          - ticker: {"ticker", "symbol"}
          - headline: {"headline", "title"}
        * rows are read in order and truncated to `limit`.
    - list[dict]: each dict should have keys {"ticker", "headline"};
        missing "ticker" falls back to `default_ticker`.
    - list[str]: each string is a headline; requires `default_ticker`.
    - str: a single headline; requires `default_ticker`.

    Parameters
    ----------
    items : Any
        The raw input to normalize (DataFrame, list[dict], list[str], or str).
    default_ticker : str | None, keyword-only
        Ticker to use when it is absent (required for list[str] and str inputs).
    limit : int, default 200
        Maximum number of items to return (applies to DataFrame rows and list inputs).

    Returns
    -------
    List[Dict[str, Any]]
        A list of dicts like {"ticker": "<TICKER>", "headline": "<HEADLINE>"}.
        Empty list if the input is an empty list.

    Raises
    ------
    ToolExecutionError
        - DataFrame missing required columns.
        - list[dict] contains no valid entries.
        - list[str] or str provided without `default_ticker`.
        - Unsupported input type.

    Notes
    -----
    - Inputs are trimmed, lower/underscore/space-insensitive for column detection.
    - Blank tickers or headlines are skipped.
    """
    import pandas as pd

    def _norm_col(c: str) -> str:
        return c.strip().lower().replace(" ", "").replace("_", "")

    out: List[Dict[str, Any]] = []

    if isinstance(items, list) and len(items) == 0:
        return []

    # 1) DataFrame
    if isinstance(items, pd.DataFrame):
        cols = {_norm_col(c): c for c in items.columns}
        if "ticker" in cols and "headline" in cols:
            tcol, hcol = cols["ticker"], cols["headline"]
        else:
            tcol = cols.get("symbol") or cols.get("ticker")
            hcol = cols.get("title") or cols.get("headline")
        if not tcol or not hcol:
            raise ToolExecutionError(
                "sentiment_score_titles_tool: DataFrame must have 'ticker' and 'headline' columns."
            )
        for _, row in items.head(limit).iterrows():
            t = str(row[tcol]).strip()
            h = str(row[hcol]).strip()
            if t and h:
                out.append({"ticker": t, "headline": h})
        return out

    # 2) list[dict]
    if isinstance(items, list) and items and isinstance(items[0], dict):
        for d in items[:limit]:
            t = str(d.get("ticker", default_ticker) or "").strip()
            h = str(d.get("headline", "") or "").strip()
            if t and h:
                out.append({"ticker": t, "headline": h})
        if not out:
            raise ToolExecutionError("sentiment_score_titles_tool: empty or invalid dict list for items.")
        return out

    # 3) list[str]
    if isinstance(items, list) and items and isinstance(items[0], str):
        if not default_ticker:
            raise ToolExecutionError("sentiment_score_titles_tool: list[str] requires default_ticker.")
        for s in items[:limit]:
            h = str(s).strip()
            if h:
                out.append({"ticker": default_ticker, "headline": h})
        return out

    # 4) str
    if isinstance(items, str):
        if not default_ticker:
            raise ToolExecutionError("sentiment_score_titles_tool: str headline requires default_ticker.")
        h = items.strip()
        if h:
            return [{"ticker": default_ticker, "headline": h}]

    raise ToolExecutionError(
        f"sentiment_score_titles_tool: unsupported items type: {type(items)}; "
        "expect DataFrame, list[dict], list[str], or str."
    )
