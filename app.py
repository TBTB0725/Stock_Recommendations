# app.py
import os
import math
import datetime as dt
from typing import Optional, Dict, List
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import json

# Project modules
from data import get_prices, to_returns
from forecast import prophet_expected_returns
from risk import cov_matrix, TRADING_DAYS_PER_YEAR as tdpy
from optimize import (
    solve_min_variance,
    solve_max_return,
    solve_max_sharpe,
    Constraints,
)
from report import evaluate_portfolio, compile_report
from news import fetch_finviz_headlines
from sentiment import score_titles, DEFAULT_MODEL as SENTI_DEFAULT_MODEL
from agent.agent import ChatStockAgent

_H = {"1D":1,"5D":5,"1W":5,"2W":10,"1M":21,"3M":63,"6M":126,"1Y":252}
_H_HUMAN = {"1D":"1 Day","5D":"5 Days","1W":"1 Week","2W":"2 Weeks","1M":"1 Month"}

# --------------------------
# Help Functions
# --------------------------
def _get_openai_key() -> Optional[str]:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI")

# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(
    page_title="Stock_Recommendations",
    layout="wide",
)

# --------------------------
# Agent mode
# --------------------------
def _mount_agent_mode():
    header_col, reset_col = st.columns([8, 1.5])
    with header_col:
        st.markdown("## 🤖 Agent Mode — QuantChat")
    with reset_col:
        if st.button("↺ Reset Conversation", key="qc_reset", use_container_width=True):
            if "qc_agent" in st.session_state:
                st.session_state["qc_agent"].reset()
            st.session_state["qc_history"] = []
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    # API key
    key = _get_openai_key()
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        st.warning("No OPENAI_API_KEY found. Set it in secrets or env variables.")

    # 初始化会话状态：一个 agent + 历史消息
    if "qc_agent" not in st.session_state:
        st.session_state["qc_agent"] = ChatStockAgent(
            model="gpt-4.1-mini",
            verbose=True,
        )
    if "qc_history" not in st.session_state:
        # 存简单结构：[{"role": "user"/"assistant", "content": str}, ...]
        st.session_state["qc_history"] = []

    agent = st.session_state["qc_agent"]

    # 先把历史消息画出来
    for msg in st.session_state["qc_history"]:
        role = msg["role"]
        content = msg["content"]
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message("user" if role == "user" else "assistant", avatar=avatar):
            st.markdown(content)

    # 输入框（始终在最下面）
    user_input = st.chat_input("Ask QuantChat anything within its quantitative scope...")

    if user_input is not None and user_input.strip() != "":
        user_input = user_input.strip()

        # 1) 立即渲染用户消息（右侧）
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        st.session_state["qc_history"].append(
            {"role": "user", "content": user_input}
        )

        # 2) assistant 占位 + 同步调用 agent.ask（用户此时已经看到自己发的内容了）
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            try:
                reply = agent.ask(user_input)
            except Exception as e:
                reply = f"Agent failed with error: {e}"
            placeholder.markdown(reply)

        st.session_state["qc_history"].append(
            {"role": "assistant", "content": reply}
        )

        # 不强制 rerun；下一轮输入时会带着完整 history 重绘


# === Sidebar 顶部放一个 Agent 模式开关；开则渲染 Agent UI 并停止后续渲染 ===
agent_mode = st.sidebar.toggle(
    "🤖 Agent mode",
    value=False,
    help="Show only the QuantChat agent interface and hide the manual parameter panel.",
)

if agent_mode:
    # === Sidebar: What this agent can do ===
    st.sidebar.markdown("### 🧠 What QuantChat can do")

    st.sidebar.markdown(
        """
**Data & Preprocessing**
- Fetch historical prices for selected tickers (via yfinance).
- Convert prices to daily log returns for further analysis.

**Forecasting**
- Use a Prophet-based model to estimate expected returns from price history.
- Support discrete horizons: 1D, 5D, 1W, 2W, 1M, 3M, 6M, 1Y (no guessing in-between).

**Risk Analytics**
- Compute the covariance matrix of daily returns, with optional 252-day annualization.
- Compute portfolio volatility using √(wᵀ Σ w).
- Aggregate asset returns into a single portfolio return series.
- Estimate portfolio Value-at-Risk (VaR) via historical simulation for a chosen horizon and confidence level.

**Portfolio Optimization**
- Construct portfolios using optimization:
  - Minimum-variance (`min_var`)
  - Maximum-return (`max_ret`)
  - Maximum-Sharpe (`max_sharpe`)
- Return clean weight vectors (percent allocation per asset) under your inputs and constraints.

**Portfolio Evaluation & Reporting**
- Evaluate any given portfolio using:
  - Expected return (μ), volatility (σ), Sharpe ratio
  - Historical VaR based on your daily returns and forecasts
- Compile multiple portfolio results into:
  - A summary table of key metrics
  - Wide-format weights and allocation tables for inspection or export.

**News & Sentiment**
- Fetch recent headlines from FINVIZ for selected tickers.
- Score each (ticker, headline) pair with an LLM for near-term price impact.
- Return enriched news data you can use as an additional signal.
        """
    )

    st.sidebar.markdown(
        """
**Example questions**
- "With $100,000 and tickers AAPL, MSFT, NVDA, build and evaluate a max_sharpe portfolio using the last 252 trading days and rf = 2%."
- "For the next 1M, based on your forecasts and risk metrics, is AAPL or AMZN more attractive?"
- "Here are my weights for AAPL, MSFT, AMZN — please compute μ, σ, Sharpe, and 1D 95% VaR."
- "Fetch recent news for TSLA and score the short-term impact of each headline."
        """
    )

    st.sidebar.markdown(
        """
**Limitations**
- Uses only the tools above: price/returns, Prophet forecasts, covariance & VaR, optimization, evaluation, and optional news/sentiment.
- No invented inputs or numbers; if required data is missing or invalid, the agent will ask you.
- No advice on non-quantitative topics (health, generic life coaching, unrelated coding, etc.).
        """
    )

    # Render chat UI only
    _mount_agent_mode()
    st.stop()
# ==============================================================

st.title("📈 Stock_Recommendations — Prophet (Growth) + Covariance & VaR (Risk)")
st.caption("Interactively set parameters, compute Equal-Weight / Min-Variance / Max-Return / Max-Sharpe strategies, and visualize results.")

# --------------------------
# Sidebar — Parameters
# --------------------------
st.sidebar.header("Parameters")

# === Always visible ===
tickers_str = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, AMZN, NVDA, GOOGL, TSLA, JPM, XOM, PFE, BHVN",
    help="Example: AAPL, MSFT, AMZN, NVDA, GOOGL, TSLA, JPM, XOM, PFE, BHVN",
)

capital = st.sidebar.number_input(
    "Total Capital",
    min_value=0.0, step=1000.0, value=100000.0,
    help="Example: 100000"
)

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    options=["1D","5D","1W","2W","1M"],
    index=2,
    help="Default: 1W"
)

# === Advanced toggle ===
st.sidebar.divider()
show_adv = st.sidebar.checkbox("Show advanced", value=False)

# ---- defaults when advanced settings are hidden ----
lookback_days = 504
var_alpha = 0.05
var_h_days = 1
rf = 0.042
use_cap = False
cap_val: Optional[float] = None
end_date_sel = None
sh_restarts = 0
sh_seed = 0
save_csv = False
news_open = False

prophet_tune = False
prophet_metric = "rmse"
cv_initial = 252
cv_period = 126
param_grid_json: Optional[str] = None

if show_adv:
    st.sidebar.subheader("Advanced: optional controls")

    # --- News & Sentiment ---
    news_open = st.sidebar.checkbox("Fetch Finviz news & score sentiment", value=False)
    if news_open:
        news_lookback = st.sidebar.number_input(
            "News lookback (days)", min_value=1, max_value=60, value=12, step=1
        )
        news_per_ticker = st.sidebar.slider(
            "Max headlines per ticker (raw scrape)", min_value=10, max_value=200, value=25, step=10
        )
        news_final_cap = st.sidebar.slider(
            "Final cap per ticker (after sorting)", min_value=5, max_value=100, value=5, step=5
        )
        news_model = st.sidebar.text_input(
            "Sentiment model (OpenAI)",
            value=SENTI_DEFAULT_MODEL,
            help="Must support Chat Completions JSON mode (e.g., gpt-4.1-mini)."
        )
        openai_key = _get_openai_key()
        if news_open and not openai_key:
            st.sidebar.warning("No OPENAI_API_KEY")

    # --- Lookback days ---
    if st.sidebar.checkbox("Lookback days", value=False):
        lookback_days = st.sidebar.number_input(
            "Lookback Days (trading days)",
            min_value=60, max_value=1260, step=21, value=lookback_days,
            help="Default: 504 (~1Y)",
            label_visibility="collapsed",
        )

    # --- Risk-free rate ---
    if st.sidebar.checkbox("Risk-free rate", value=False):
        rf_str = st.sidebar.text_input(
            "Risk-free rate (annual)",
            value=f"{rf:.3f}",
            label_visibility="collapsed"
        )
        try:
            rf = float(rf_str)
        except ValueError:
            st.sidebar.error("Please enter a valid number for risk-free rate.")
            rf = 0.0

    # --- Per-asset cap ---
    if st.sidebar.checkbox("Per-asset weight cap", value=False):
        use_cap = True
        cap_val = st.sidebar.slider("per-asset weight cap", min_value=0.05, max_value=1.0, step=0.05, value=0.4, label_visibility="collapsed")

    # --- Custom end date ---
    if st.sidebar.checkbox("End date", value=False):
        end_date_sel = st.sidebar.date_input(
            "end date",
            value=None,
            help="Leave empty = today (moves intraday).",
            label_visibility="collapsed"
        )

    # --- Prophet tuning (hidden until enabled) ---
    if st.sidebar.checkbox("Prophet tuning", value=False):
        prophet_tune = True
        prophet_metric = st.sidebar.selectbox(
            "CV metric",
            options=["rmse","mse","mae","mape","mdape","coverage"],
            index=0,
            help="Returns can be positive and negative; rmse/mae recommended."
        )
        cv_initial = st.sidebar.number_input(
            "CV initial (TRADING days, optional)",
            min_value=0, value=252,
            help="Default: max(252, 3*horizon). Enter 0 to use default."
        )
        cv_period = st.sidebar.number_input(
            "CV period (TRADING days, optional)",
            min_value=0, value=126,
            help="Default: max(21, horizon//2). Enter 0 to use default."
        )

        # Custom grid JSON behind its own toggle
        if st.sidebar.checkbox("Use custom param grid (JSON)", value=False):
            _default_grid = {
                "n_changepoints": [0, 5],
                "changepoint_prior_scale": [0.01, 0.03, 0.1],
                "weekly_seasonality": [False],
                "yearly_seasonality": [False],
                "daily_seasonality": [False],
                "seasonality_mode": ["additive"]
            }
            grid_json_text = st.sidebar.text_area(
                "Grid JSON",
                value=json.dumps(_default_grid, indent=2),
                height=180,
                help='Edit JSON to tune Prophet. Example keys: '
                    'n_changepoints, changepoint_prior_scale, weekly_seasonality, '
                    'yearly_seasonality, daily_seasonality, seasonality_mode'
            )
            try:
                # quick validation
                _ = json.loads(grid_json_text)
                param_grid_json = grid_json_text
            except json.JSONDecodeError as e:
                st.sidebar.error(f"Invalid JSON: {e}")
                param_grid_json = None


# --------------------------
# Cached helpers
# --------------------------
@st.cache_data(show_spinner=False)
def _fetch_prices_cached(tickers: List[str], lookback: int, end_date: Optional[dt.date]):
    return get_prices(
        tickers,
        lookback_days=lookback,
        end=end_date,
        pad_ratio=2.0,
        auto_business_align=True,
        use_adjusted_close=True,
    )

@st.cache_data(show_spinner=False)
def _returns_log_cached(prices: pd.DataFrame) -> pd.DataFrame:
    return to_returns(prices, method="log")

@st.cache_data(show_spinner=False)
def _cov_annual_cached(returns_log: pd.DataFrame) -> pd.DataFrame:
    return cov_matrix(returns_log, annualize=True)

@st.cache_data(show_spinner=False)
def _mu_prophet_cached(prices: pd.DataFrame,
                       horizon: str,
                       tune: bool,
                       metric: str,
                       cv_init: Optional[int],
                       cv_per: Optional[int],
                       param_grid_json: Optional[str]) -> pd.Series:
    param_grid = None
    if param_grid_json:
        try:
            param_grid = json.loads(param_grid_json)
        except Exception:
            param_grid = None

    try:
        kwargs = {
            "horizon": horizon,
            "tune": tune,
            "cv_metric": metric,
            "cv_initial_days": (None if (cv_init is None or cv_init <= 0) else int(cv_init)),
            "cv_period_days": (None if (cv_per is None or cv_per <= 0) else int(cv_per)),
            "param_grid": param_grid,
        }
        return prophet_expected_returns(prices, **kwargs)
    except TypeError:
        return prophet_expected_returns(prices, horizon=horizon)

@st.cache_data(show_spinner=False, ttl=600)
def _fetch_news_cached_ui(tickers: List[str], lookback_days: int, per_ticker_count: int, final_cap: int) -> pd.DataFrame:
    df = fetch_finviz_headlines(
        tickers=tickers,
        lookback_days=lookback_days,
        per_ticker_count=per_ticker_count,
        final_cap=final_cap,
        sleep_s=0.2,
    )
    cols = ["ticker", "datetime", "headline", "source", "url", "relatedTickers"]
    return df[cols] if len(df) else df

@st.cache_data(show_spinner=False, ttl=300)
def _score_news_llm(df_news: pd.DataFrame, model_name: str, api_key: Optional[str]) -> pd.DataFrame:
    if df_news is None or df_news.empty:
        return df_news

    items = df_news[["ticker", "headline"]].to_dict("records")

    try:
        scored = score_titles(items, model_name=model_name, api_key = api_key)
    except Exception as e:
        st.warning(f"Sentiment scoring failed: {e}")
        out = df_news.copy()
        out["impact"] = 0.0
        out["reason"] = ""
        return out[["ticker", "datetime", "headline", "source", "url", "impact", "reason"]]

    impacts, reasons = [], []
    for row in scored:
        try:
            impacts.append(float(row.get("impact", 0.0)))
        except Exception:
            impacts.append(0.0)
        ai_json = row.get("ai_json") if isinstance(row.get("ai_json"), dict) else {}
        reasons.append(str(ai_json.get("reason", "") or ""))

    out = df_news.copy()
    out["impact"] = (
        pd.to_numeric(pd.Series(impacts), errors="coerce")
        .fillna(0.0)
    )
    out["reason"] = reasons

    keep = ["ticker", "datetime", "headline", "source", "url", "impact", "reason"]
    for c in keep:
        if c not in out.columns:
            out[c] = "" if c in ("headline", "source", "url", "reason") else 0.0
    return out[keep]

# --------------------------
# Run button
# --------------------------
col_run, col_hint = st.columns([1, 2])
run = col_run.button("🚀 Run Analysis")
if col_hint:
    if prophet_tune:
        st.info("Tip: Enabling CV/tuning can be slow. Try fewer tickers or longer CV period.")

# --------------------------
# Pipeline
# --------------------------
if run:
    try:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        if len(tickers) == 0:
            st.error("Please enter at least one ticker.")
            st.stop()

        # Feasibility check: cap * n >= 1
        if use_cap and cap_val is not None:
            if cap_val * len(tickers) < 1.0:
                st.error(f"Infeasible cap: n={len(tickers)}, cap={cap_val} → n*cap={cap_val*len(tickers):.3f} < 1. Increase cap or add more assets.")
                st.stop()

        # Fetch prices
        with st.spinner("[1/6] Fetching prices (yfinance)..."):
            prices = _fetch_prices_cached(tickers, lookback_days, end_date_sel or None)

        # Price chart
        with st.expander("📉 Price chart (Adjusted Close)", expanded=False):
            st.line_chart(prices)

        # Daily log returns
        with st.spinner("[2/6] Computing daily log returns..."):
            rets_log = _returns_log_cached(prices)

        # Prophet μ (annualized)
        with st.spinner(f"[3/6] Forecasting annualized expected returns μ (horizon={horizon})..."):
            mu_annual = _mu_prophet_cached(
                prices, horizon,
                prophet_tune, prophet_metric,
                None if cv_initial <= 0 else cv_initial,
                None if cv_period  <= 0 else cv_period,
                param_grid_json
            )

        # Annualized covariance Σ
        with st.spinner("[4/6] Estimating annualized covariance matrix Σ..."):
            Sigma_annual = _cov_annual_cached(rets_log)

        # Align order (critical)
        prices       = prices[tickers]
        rets_log     = rets_log[tickers]
        mu_annual    = mu_annual.loc[tickers]
        Sigma_annual = Sigma_annual.loc[tickers, tickers]

        # Optimize four strategies
        with st.spinner("[5/6] Solving optimal weights for strategies..."):
            cons = Constraints(max_weight_per_asset=cap_val) if use_cap else None
            n = len(tickers)
            w_equal = np.ones(n) / n
            w_minv  = solve_min_variance(Sigma_annual, cons=cons)
            w_maxr  = solve_max_return(mu_annual.values, cons=cons)
            w_msr   = solve_max_sharpe(mu_annual.values, Sigma_annual.values, rf=rf, cons=cons,
                                       restarts=int(sh_restarts), seed=int(sh_seed))

        # Evaluate portfolios
        with st.spinner("[6/6] Evaluating portfolios (μ/σ/VaR/Sharpe)..."):
            results = []
            results.append(evaluate_portfolio("Equal Weight", tickers, w_equal, capital, mu_annual, Sigma_annual, rets_log, rf, var_alpha, var_h_days, True))
            results.append(evaluate_portfolio("Min Variance", tickers, w_minv, capital, mu_annual, Sigma_annual, rets_log, rf, var_alpha, var_h_days, True))
            results.append(evaluate_portfolio("Max Return", tickers, w_maxr, capital, mu_annual, Sigma_annual, rets_log, rf, var_alpha, var_h_days, True))
            results.append(evaluate_portfolio("Max Sharpe", tickers, w_msr, capital, mu_annual, Sigma_annual, rets_log, rf, var_alpha, var_h_days, True))

            summary, weights_tbl, alloc_tbl = compile_report(results)

        st.success("Done ✅")

        # === Convert summary from annualized → horizon (keep VaR as-is) ===
        days = _H[horizon.upper()]
        scale = days / float(tdpy) 
        sqrt_scale = math.sqrt(scale)
        h_label = _H_HUMAN.get(horizon.upper(), horizon)

        mu_a  = summary["ExpReturn(annual)"].astype(float)
        sig_a = summary["Volatility(annual)"].astype(float)
        var_1d = summary["VaR(95%, 1D)"] 

        mu_h  = (1.0 + mu_a) ** scale - 1.0
        sig_h = sig_a * sqrt_scale
        rf_h  = (1.0 + rf) ** scale - 1.0

        sharpe_h = (mu_h - rf_h) / sig_h.replace(0, np.nan)

        summary_h = pd.DataFrame({
            "Strategy": summary.index,
            f"Return({h_label})": mu_h.values,
            f"Volatility({h_label})": sig_h.values,
            "VaR(95%, 1D)": var_1d.values,
            f"Sharpe({h_label})": sharpe_h.values
        }).set_index("Strategy")

        summary = summary_h

        # --------------------------
        # Display
        # --------------------------
        # ---------- High-level textual recommendation (above all charts) ----------
        st.markdown("### 📝 Strategy Recommendation Summary")

        try:
            # 找到 horizon 回报列 & VaR 列
            ret_col = next(c for c in summary.columns if c.lower().startswith("return("))
            var_col = "VaR(95%, 1D)" if "VaR(95%, 1D)" in summary.columns else None

            def _top_allocations(strategy_name: str, top_n: int = 4) -> str:
                """Return a string like 'AAPL 30%, MSFT 25%, ...' for the given strategy."""
                if strategy_name not in weights_tbl.columns:
                    return ""
                col = weights_tbl[strategy_name].dropna()
                col = col[col > 0]
                if col.empty:
                    return ""
                col = col.sort_values(ascending=False).head(top_n)
                parts = [f"{ticker} {weight:.0%}" for ticker, weight in col.items()]
                return ", ".join(parts)

            # 各策略组合的 top holdings 文本
            ms_text = _top_allocations("Max Sharpe")
            mr_text = _top_allocations("Max Return")
            mv_text = _top_allocations("Min Variance")

            # 各策略在当前 horizon 的预期收益
            h_ret_ms = summary.loc["Max Sharpe", ret_col] if "Max Sharpe" in summary.index else None
            h_ret_mr = summary.loc["Max Return", ret_col] if "Max Return" in summary.index else None
            h_ret_mv = summary.loc["Min Variance", ret_col] if "Min Variance" in summary.index else None

            # 各策略的一天 95% VaR（保持你目前 summary 的口径）
            var_ms = summary.loc["Max Sharpe", var_col] if var_col and "Max Sharpe" in summary.index else None
            var_mr = summary.loc["Max Return", var_col] if var_col and "Max Return" in summary.index else None
            var_mv = summary.loc["Min Variance", var_col] if var_col and "Min Variance" in summary.index else None

            lines = []

            if ms_text and h_ret_ms is not None:
                line = (
                    f"- **Max-Sharpe portfolio (balanced risk/return)**: {ms_text}. "
                    f"Expected return ≈ {h_ret_ms:.2%} over {h_label}"
                )
                if var_ms is not None:
                    line += f", 1-day 95% VaR ≈ {var_ms:.2%}."
                else:
                    line += "."
                lines.append(line)

            if mr_text and h_ret_mr is not None:
                line = (
                    f"- **If you want to maximize return**: {mr_text}. "
                    f"Expected return ≈ {h_ret_mr:.2%} over {h_label}"
                )
                if var_mr is not None:
                    line += f", 1-day 95% VaR ≈ {var_mr:.2%}."
                else:
                    line += "."
                lines.append(line)

            if mv_text and h_ret_mv is not None:
                line = (
                    f"- **If you want to minimize risk**: {mv_text}. "
                    f"Expected return ≈ {h_ret_mv:.2%} over {h_label}"
                )
                if var_mv is not None:
                    line += f", 1-day 95% VaR ≈ {var_mv:.2%}."
                else:
                    line += "."
                lines.append(line)

            if lines:
                st.markdown(
                    "Based on the optimized portfolios and risk/return metrics:\n\n"
                    + "\n".join(lines)
                    + "\n\nDetailed numbers and charts are provided below for your review."
                )
            else:
                st.info(
                    "Portfolios have been computed, but a concise textual summary "
                    "could not be generated from the current results."
                )

        except Exception as e:
            st.warning(f"Failed to generate textual summary: {e}")

        days = _H[horizon.upper()]
        r_horizon = (1.0 + mu_annual) ** (days / float(tdpy)) - 1.0
        st.markdown(f"### 📈 Forecasted Return over {h_label}")
        r_df = r_horizon.reset_index()
        r_df.columns = ["Ticker", "HorizonReturn"]
        chart_r = alt.Chart(r_df).mark_bar().encode(
            x=alt.X("Ticker:N", sort=None),
            y=alt.Y("HorizonReturn:Q", title=f"Return over {h_label}",
                    axis=alt.Axis(format="%")), 
            tooltip=[
                "Ticker",
                alt.Tooltip("HorizonReturn:Q", title=f"Return {h_label}", format=".2%"), 
            ],
        )
        st.altair_chart(chart_r, use_container_width=True)

        st.markdown("#### Weights by Strategy (Stacked)")
        wt_long = weights_tbl.reset_index().melt("index", var_name="Strategy", value_name="Weight")
        wt_long.rename(columns={"index":"Ticker"}, inplace=True)
        chart_w = (
            alt.Chart(wt_long)
            .mark_bar()
            .encode(
                x=alt.X("Strategy:N"),
                y=alt.Y("Weight:Q", stack="normalize", title="Weight (stacked)", axis=alt.Axis(format="%")),
                color=alt.Color("Ticker:N"),
                tooltip=["Strategy", "Ticker", alt.Tooltip("Weight:Q", format=".2%")],
            )
        )
        st.altair_chart(chart_w, use_container_width=True)


        # Tables: Weights / Allocation
        st.subheader("🧮 Weights / Allocation")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Weights (rows=Ticker, cols=Strategy)**")
            st.dataframe(weights_tbl.style.format("{:.2%}"))
        with c4:
            st.markdown("**Allocation ($, rows=Ticker, cols=Strategy)**")
            st.dataframe(alloc_tbl.style.format("${:,.2f}"))

        st.subheader("📊 Summary (Portfolio Metrics)")
        percent_cols = [c for c in summary.columns if "sharpe" not in c.lower()]
        fmt = {c: "{:.2%}" for c in percent_cols}
        st.dataframe(summary.style.format(fmt))

        c1, c2 = st.columns(2)

        ret_col = next(c for c in summary.columns if c.lower().startswith("return("))
        vol_col = next(c for c in summary.columns if c.lower().startswith("volatility("))

        with c1:
            st.markdown(f"#### Strategy Return — {h_label}")
            ret_df = summary.reset_index()[["Strategy", ret_col]]
            chart_ret = (
                alt.Chart(ret_df)
                .mark_bar()
                .encode(
                    x=alt.X("Strategy:N", sort=None, title="Strategy"),
                    y=alt.Y(f"{ret_col}:Q", title=ret_col, axis=alt.Axis(format="%")),
                    tooltip=[
                        "Strategy",
                        alt.Tooltip(f"{ret_col}:Q", title="Return", format=".2%"),
                    ],
                )
            )
            st.altair_chart(chart_ret, use_container_width=True)

        with c2:
            st.markdown(f"#### Strategy Volatility — {h_label}")
            vol_df = summary.reset_index()[["Strategy", vol_col]]
            chart_vol = (
                alt.Chart(vol_df)
                .mark_bar()
                .encode(
                    x=alt.X("Strategy:N", sort=None, title="Strategy"),
                    y=alt.Y(f"{vol_col}:Q", title=vol_col, axis=alt.Axis(format="%")),
                    tooltip=[
                        "Strategy",
                        alt.Tooltip(f"{vol_col}:Q", title="Volatility", format=".2%"),
                    ],
                )
            )
            st.altair_chart(chart_vol, use_container_width=True)
        
        # News & Sentiment
        if news_open:
            if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI"):
                st.warning("No OPENAI_API_KEY")

            with st.spinner("[News] Fetching Finviz headlines..."):
                df_news = _fetch_news_cached_ui(
                    tickers=[t.strip().upper() for t in tickers if t.strip()],
                    lookback_days=int(news_lookback),
                    per_ticker_count=int(news_per_ticker),
                    final_cap=int(news_final_cap),
                )

            st.subheader("📰 News & Sentiment")
            if df_news is None or df_news.empty:
                st.info("No headlines fetched.")
            else:
                with st.spinner("[News] Scoring headlines..."):
                    df_scored = _score_news_llm(df_news, model_name=news_model, api_key=openai_key)

                df_scored = pd.DataFrame(df_scored)
                df_scored.loc[:, "impact"] = (
                    pd.to_numeric(df_scored["impact"], errors="coerce")
                    .fillna(0.0)
                )

                try:
                    avg_impact = (
                        df_scored.groupby("ticker", as_index=False)["impact"]
                        .mean()
                        .sort_values("impact", ascending=False)
                    )
                    ch_avg = (
                        alt.Chart(avg_impact)
                        .mark_bar()
                        .encode(
                            x=alt.X("ticker:N", title="Ticker", sort=None),
                            y=alt.Y("impact:Q", title="Avg Impact", axis=alt.Axis(format=".2f")),
                            tooltip=["ticker", alt.Tooltip("impact:Q", title="Avg Impact", format=".2f")],
                        )
                    )
                    st.altair_chart(ch_avg, use_container_width=True)
                except Exception:
                    pass

                cpos, cneg = st.columns(2)
                with cpos:
                    st.markdown("**Top Positive Headlines**")
                    top_pos = df_scored.sort_values("impact", ascending=False).head(5)
                    for _, r in top_pos.iterrows():
                        st.markdown(
                            f"- **{r['ticker']}** · {r['impact']:+.2f} — "
                            f"[{r['headline']}]({r['url']})"
                            + (f" · _{r['reason']}_"
                               if isinstance(r.get('reason'), str) and r['reason'] else "")
                        )
                with cneg:
                    st.markdown("**Top Negative Headlines**")
                    top_neg = df_scored.sort_values("impact", ascending=True).head(5)
                    for _, r in top_neg.iterrows():
                        st.markdown(
                            f"- **{r['ticker']}** · {r['impact']:+.2f} — "
                            f"[{r['headline']}]({r['url']})"
                            + (f" · _{r['reason']}_"
                               if isinstance(r.get('reason'), str) and r['reason'] else "")
                        )

                with st.expander("🔎 Full Headlines Table", expanded=False):
                    _tbl = df_scored.copy()
                    if "datetime" in _tbl.columns and _tbl["datetime"].notna().any():
                        _tbl["datetime"] = (
                            pd.to_datetime(_tbl["datetime"], utc=True, errors="coerce")
                            .dt.tz_convert("UTC")
                            .dt.strftime("%Y-%m-%d %H:%M UTC")
                        )
                    st.dataframe(
                        _tbl[["ticker", "datetime", "impact", "headline", "reason", "source", "url"]],
                        use_container_width=True,
                    )


        # --------------------------
        # Downloads
        # --------------------------
        st.subheader("⬇️ Download Results")

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_summary = summary.to_csv(index=True).encode("utf-8")
        csv_weights = weights_tbl.to_csv(index=True).encode("utf-8")
        csv_alloc   = alloc_tbl.to_csv(index=True).encode("utf-8")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"summary_{stamp}.csv", csv_summary)
            zf.writestr(f"weights_{stamp}.csv", csv_weights)
            zf.writestr(f"allocation_{stamp}.csv", csv_alloc)
        zip_buf.seek(0)

        st.download_button(
            label="Download Strategy ZIP",
            data=zip_buf.getvalue(),
            file_name=f"portfolio_reports_{stamp}.zip",
            mime="application/zip",
            key="dl_zip",
        )

        if news_open:
            st.download_button(
                "Download News + Sentiment CSV",
                data=df_scored.to_csv(index=False).encode("utf-8"),
                file_name="news_sentiment.csv",
                mime="text/csv",
                key="dl_news_csv",
            )

        # Footer
        st.caption("Note: Enabling Prophet tuning can be slow. Try fewer tickers / shorter lookback / shorter horizon, or reduce grid size.")

    except Exception as e:
        st.error(f"Error: {e}")
