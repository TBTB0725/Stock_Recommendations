# agent/tools.py
from __future__ import annotations

import time
import logging
from functools import wraps
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# === 业务模块导入（按你提供的真实签名） ===
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
from sentiment import score_titles as _score_titles  # 需要 api_key

# 在 agent/tools.py 顶部其它导入下面（与 sentiment_score_titles_tool 同一文件）新增：
def _coerce_news_items(
    items: Any,
    *,
    default_ticker: str | None = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    统一把各种形态的输入规范化为:
    [{"ticker": str, "headline": str}, ...]
    支持:
      - pd.DataFrame: 需要包含列 ["ticker","headline"]（大小写不敏感）
      - list[dict]: 需要至少包含键 "ticker","headline"
      - list[str]: 需配合 default_ticker
      - str: 单条标题，需配合 default_ticker
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
            # 尝试更宽松的匹配
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

    # 4) 单条 str
    if isinstance(items, str):
        if not default_ticker:
            raise ToolExecutionError("sentiment_score_titles_tool: str headline requires default_ticker.")
        h = items.strip()
        if h:
            return [{"ticker": default_ticker, "headline": h}]

    # 其他类型
    raise ToolExecutionError(
        f"sentiment_score_titles_tool: unsupported items type: {type(items)}; "
        "expect DataFrame, list[dict], list[str], or str."
    )


# ------------------------
# 日志配置（模块级别）
# ------------------------
logger = logging.getLogger("agent.tools")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ------------------------
# 自定义异常
# ------------------------
class ToolExecutionError(RuntimeError):
    """统一的工具执行错误，外层可据此中断或重试。"""

# ------------------------
# 计时 + 日志 装饰器
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

# ========================
# 对外暴露的“工具函数”
# ========================

# --- 数据与收益 ---
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
    """
    包装 data.get_prices；返回 PriceFrame（pd.DataFrame）
    """
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
    """
    包装 data.to_returns；返回 ReturnFrame（pd.DataFrame）
    ReturnMethod 你定义为 Literal，工具层用 str 透传即可。
    """
    return to_returns(prices=prices, method=method, dropna=dropna)


# --- 预测（Prophet 年化 μ）---
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
    """
    返回年化的期望收益 μ（index=tickers）
    """
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


# --- 风险（协方差 Σ）---
@_timed("risk_cov_matrix")
def risk_tool(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    返回年化协方差矩阵 Σ（pd.DataFrame，index/columns=tickers）
    """
    return cov_matrix(
        returns=returns,
        annualize=annualize,
        trading_days_per_year=trading_days_per_year,
    )


# --- 优化器统一入口 ---
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
    """
    objective ∈ {"min_var", "max_ret", "max_sharpe"}
    - min_var: 需要 Sigma_annual
    - max_ret: 需要 mu_annual
    - max_sharpe: 需要 mu_annual + Sigma_annual
    返回 np.ndarray 权重（与 tickers 顺序对齐）
    """
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


# --- 投后评估 & 报告 ---
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
    """
    返回 PortfolioResult（见 report.py 的 @dataclass）
    """
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
    """
    返回 (summary_df, risk_df, allocation_df)
    """
    return compile_report(results)


# --- 新闻 + 情绪 ---
@_timed("fetch_news")
def fetch_news_tool(
    tickers: Iterable[str],
    lookback_days: int = 3,
    per_ticker_count: int = 15,
    final_cap: int = 200,
    sleep_s: float = 0.5,
) -> pd.DataFrame:
    """
    使用 Finviz 抓取标题（按你定义的包装）
    返回列应包含: ["ticker","headline","source","url","datetime", ...]
    """
    return fetch_finviz_headlines(
        tickers=tickers,
        lookback_days=lookback_days,
        per_ticker_count=per_ticker_count,
        final_cap=final_cap,
        sleep_s=sleep_s,
    )


@_timed("score_titles")
def sentiment_score_titles_tool(
    items: Any,
    *,
    model_name: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    default_ticker: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    更健壮的情绪打分工具：
      - 自动把 DataFrame / list[dict] / list[str] / str 规范化为 [{ticker, headline}, ...]
      - default_ticker: 当 items 是 str 或 list[str] 时需要
      - limit: 限制打分数量（避免过大批次）
    输出: 与 sentiment.score_titles 对齐
    """
    norm_items = _coerce_news_items(items, default_ticker=default_ticker, limit=limit)
    if not norm_items:
        logger.warning("sentiment_score_titles_tool: no items to score (empty input).")
        return []
    return _score_titles(items=norm_items, model_name=model_name, api_key=api_key)
