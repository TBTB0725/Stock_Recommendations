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


# Project modules: ensure app.py is in the same directory as these files
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
from news import fetch_finviz_headlines, recent_headlines
from sentiment import score_headlines_grouped, impact_to_annual_uplift

_H = {"1D":1,"5D":5,"1W":5,"2W":10,"1M":21,"3M":63,"6M":126,"1Y":252}
_H_HUMAN = {"1D":"1 Day","5D":"5 Days","1W":"1 Week","2W":"2 Weeks","1M":"1 Month"}

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="Stock_Recommendations",
    layout="wide",
)

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

# === News Sentiment (LLM) ===
st.sidebar.divider()
use_news_sent = st.sidebar.checkbox("Use news sentiment (LLM)", value=False,
    help="Fetch Finviz headlines → LLM scores ∈[-1,1] → map to return uplift and blend into μ.")

debug_news = st.sidebar.checkbox("Debug news sentiment", value=False)

news_days_back = 10
news_per_ticker = 30
news_blend_w = 0.5
news_beta_h = 0.04
llm_provider_note = "Gemini (set GEMINI_API_KEY)"

if use_news_sent:
    news_days_back = st.sidebar.slider("News lookback days", min_value=3, max_value=30, value=10, step=1)
    news_per_ticker = st.sidebar.slider("Max headlines per ticker", min_value=5, max_value=100, value=30, step=5)
    news_blend_w = st.sidebar.slider("Blend weight w (μ += w * uplift_ann)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    news_beta_h = st.sidebar.slider("Impact→Horizon uplift β_h", min_value=0.01, max_value=0.10, value=0.04, step=0.005,
        help="Strong positive news (impact=+1) ⇒ +β_h over forecast horizon; negative likewise.")
    st.sidebar.caption(f"LLM provider: {llm_provider_note}")


# === Advanced toggle (everything else lives behind toggles) ===
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

prophet_tune = False
prophet_metric = "rmse"
cv_initial = 252
cv_period = 126
param_grid_json: Optional[str] = None  # keep None unless user enables & provides JSON

if show_adv:
    st.sidebar.subheader("Advanced: optional controls")

    # --- Lookback days（独立开关，折叠标签） ---
    if st.sidebar.checkbox("Lookback days", value=False):
        lookback_days = st.sidebar.number_input(
            "Lookback Days (trading days)",
            min_value=60, max_value=1260, step=21, value=lookback_days,
            help="Default: 504 (~1Y)",
            label_visibility="collapsed",   # ← 不显示标题，避免重复
        )

    # --- Risk-free rate（独立开关，折叠标签） ---
    if st.sidebar.checkbox("Risk-free rate", value=False):
        rf_str = st.sidebar.text_input(
            "Risk-free rate (annual)",
            value=f"{rf:.3f}",
            label_visibility="collapsed"    # ← 不显示标题
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
    """
    Backward compatible call into forecast.py.
    - If param_grid_json is provided, parse it and pass as `param_grid`.
    - If the forecast function doesn't accept new kwargs, fall back to legacy signature.
    """
    # Parse JSON (if any)
    param_grid = None
    if param_grid_json:
        try:
            param_grid = json.loads(param_grid_json)
        except Exception:
            # Already warned in sidebar; just ignore here
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
        # Legacy fallback
        return prophet_expected_returns(prices, horizon=horizon)

@st.cache_data(show_spinner=False)
def _fetch_news_cached(tickers: List[str]) -> pd.DataFrame:
    return fetch_finviz_headlines(tickers)

@st.cache_data(show_spinner=False)
def _score_news_cached(df_recent: pd.DataFrame,
                       provider: str,
                       key_fingerprint: str,
                       return_raw: bool) -> pd.DataFrame:
    """
    把 provider & key 指纹 & return_raw 纳入缓存键。
    """
    _ = (provider, key_fingerprint, return_raw)  # 仅用于缓存键
    return score_headlines_grouped(df_recent, return_raw=return_raw)



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
        
        # --- (Optional) News → LLM sentiment → blend into mu_annual ---
        if use_news_sent:
            with st.spinner("[3b] Fetching & scoring news (Finviz + LLM)..."):
                import os
                df_news = _fetch_news_cached(tickers)

                # ① 环境与 Provider 调试
                if debug_news:
                    with st.expander("🛠 Debug (Env & Provider)", expanded=False):
                        st.write("NEWS_LLM_PROVIDER =", os.getenv("NEWS_LLM_PROVIDER"))
                        st.write("GEMINI_API_KEY set? ", bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")))

                # ② 抓取结果调试
                with st.expander("🛠 Debug (News Fetch)", expanded=False):
                    st.write("tickers:", tickers)
                    st.write("fetched rows:", 0 if df_news is None else len(df_news))
                    if df_news is not None and not df_news.empty:
                        st.dataframe(df_news.head(10))

                # 过滤最近 N 天 & 限制每只股票的条数
                df_recent = recent_headlines(df_news, days_back=news_days_back, per_ticker=news_per_ticker)

                # ③ 过滤结果调试
                with st.expander("🛠 Debug (Recent Filter)", expanded=False):
                    st.write("after filter rows:", 0 if df_recent is None else len(df_recent))
                    if df_recent is not None and not df_recent.empty:
                        st.dataframe(df_recent.head(10))

                if df_recent.empty:
                    st.warning("No recent headlines found. Skipping news sentiment blend.")
                else:
                    # 评分
                    provider = os.getenv("NEWS_LLM_PROVIDER", "gemini").lower()
                    key_fp = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")[:8]  # 取前8位做指纹

                    df_scores = _score_news_cached(df_recent, provider, key_fp, debug_news)
                    # 若某些 ticker 没有分数，用 0 填充；并按当前 tickers 顺序对齐
                    s_impact = df_scores.set_index("ticker")["impact"].reindex(tickers).fillna(0.0)

                    # impact → 年化增量
                    days = _H[horizon.upper()]
                    uplift_ann = impact_to_annual_uplift(s_impact, horizon_days=days, beta_h=news_beta_h, tdpy=tdpy)

                    # μ ← μ + w * uplift_ann
                    mu_annual = (mu_annual + news_blend_w * uplift_ann).astype(float)

                    # 展示情绪表
                    with st.expander("📰 News Sentiment (per Ticker)", expanded=True):
                        show_df = pd.DataFrame({
                            "Impact [-1,1]": s_impact,
                            "Annual Uplift (from impact)": uplift_ann,
                        })
                        st.dataframe(show_df.style.format({
                            "Impact [-1,1]": "{:.2f}",
                            "Annual Uplift (from impact)": "{:.2%}",
                        }))


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
        scale = days / float(tdpy)             # time fraction in years
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

        # Debug：查看原始 LLM 返回（只在 debug_news=True 且 df_scores 含 raw 列时显示）
        if debug_news and ("raw" in df_scores.columns):
            with st.expander("🧾 Raw LLM outputs (first few)", expanded=False):
                # 只展示前 3 条，避免页面过长
                st.dataframe(df_scores[["ticker", "raw"]].head(3))

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
            label="Download ZIP",
            data=zip_buf.getvalue(),
            file_name=f"portfolio_reports_{stamp}.zip",
            mime="application/zip",
            key="dl_zip",
        )

        # Footer
        st.caption("Note: Enabling Prophet tuning can be slow. Try fewer tickers / shorter lookback / shorter horizon, or reduce grid size.")

    except Exception as e:
        st.error(f"Error: {e}")
