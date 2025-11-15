# 📈 Stock Portfolio Optimization & QuantChat Agent

An interactive Streamlit app for **forecasting**, **risk modeling**, **portfolio optimization**, and an optional **LLM-powered quant agent**.

It combines:

- Prophet-based expected return forecasts  
- Covariance / volatility / historical VaR  
- Min-Variance / Max-Return / Max-Sharpe / Equal-Weight portfolios  
- Optional FINVIZ news + LLM sentiment  
- A ChatGPT-style **QuantChat Agent** strictly constrained to these quantitative tools

---

## 🏗 Project Structure

**`data.py`**

- `get_prices`: Fetch historical prices from Yahoo Finance (yfinance).
- `to_returns`: Convert price levels into daily **log returns**.

**`forecast.py`**

- `prophet_expected_returns`: Use Prophet to forecast future prices / returns given price history and parameters.

**`risk.py`**

- `cov_matrix`: Compute covariance matrix of multi-asset daily returns, with optional annualization (252 trading days).
- `portfolio_volatility`: Compute portfolio volatility via √(wᵀ Σ w).
- `portfolio_returns_series`: Combine multi-asset return series into a single portfolio return series using given weights.
- `var_historical`: Compute portfolio Value-at-Risk using the **historical quantile** method for a chosen horizon & confidence.

**`optimize.py`**

- `solve_min_variance`: Minimum-variance portfolio (weights per asset).
- `solve_max_return`: Maximum-return portfolio (weights per asset).
- `solve_max_sharpe`: Maximum-Sharpe portfolio (weights per asset).

**`report.py`**

- `evaluate_portfolio`: Compute key metrics for a given portfolio:
  - Expected return (μ)
  - Volatility (σ)
  - Sharpe ratio
  - Historical VaR
  - Weights & allocations
- `compile_report`: Merge multiple portfolio results into:
  - A summary metrics table
  - Wide-format weights & allocation tables for inspection/export.

**`news.py`**

- `fetch_finviz_headlines`: Scrape and filter recent FINVIZ headlines for given tickers; returns a cleaned, time-bounded DataFrame.

**`sentiment.py`**

- `score_titles`: Call an LLM once per **(ticker, headline)** to score short-term price impact; returns inputs enriched with model outputs.

**`agent/tools.py`**

- Wraps the functions above as **callable tools** (prices, returns, forecast, risk, optimization, report, news, sentiment) for the agent.

**`agent/agent.py`**

- `ChatStockAgent`:
  - Uses OpenAI Chat Completions as the "brain".
  - Can only answer questions by calling the registered tools.
  - Enforces:
    - No invented numeric results.
    - Strict input validation & type checks.
    - Scope limited to: prices, returns, Prophet forecasts, covariance & VaR, optimization, evaluation, news & sentiment.

**`app.py`**

- Streamlit app UI that ties everything together:
  - **Analysis Mode**:
    - Configure tickers, capital, horizon, lookback, caps, risk-free rate, etc.
    - Run pipeline:
      - Fetch prices → log returns
      - Prophet-based μ forecast
      - Σ estimation
      - Min-Var / Max-Return / Max-Sharpe / Equal-Weight optimization
      - Portfolio evaluation & VaR
    - Auto-generated **English summary**:
      - Explains Max-Sharpe / Max-Return / Min-Variance portfolios:
        top holdings, expected horizon returns, 1D 95% VaR.
    - Interactive charts & tables:
      - Forecasted returns
      - Stacked weights
      - Allocations
      - Strategy-level return / volatility / Sharpe
    - Optional:
      - FINVIZ news fetch
      - LLM-based sentiment scoring & visualization
  - **🤖 Agent Mode — QuantChat**:
    - Chat UI (user messages on the right, agent on the left).
    - Uses `ChatStockAgent` under the hood.
    - Only responds using the allowed quantitative tools.
    - Reset button to clear conversation state.

---

## 💻 Installation

Recommended: **Python 3.9+**

Install dependencies (minimal example):

~~~bash
pip install \
  numpy pandas yfinance prophet scipy tabulate \
  streamlit altair openai requests beautifulsoup4
~~~

Or use your existing `requirements.txt` if provided.

---

## 🔑 OpenAI API Key (for Agent & Sentiment)

The following features require an OpenAI-compatible API key:

- 🤖 QuantChat Agent mode
- 📰 LLM-based news sentiment scoring

Set **one** of:

~~~bash
export OPENAI_API_KEY="sk-..."   # preferred
# or
export OPENAI="sk-..."
~~~

On Streamlit Cloud, you can also put it into **Secrets**.

If no key is set:

- Core analytics (prices, forecasts, optimization, VaR, charts) still work.
- Agent mode and sentiment scoring are disabled or will show a warning.

---

## 🚀 Run the App

Launch the Streamlit app locally:

~~~bash
streamlit run app.py
~~~

Then open the URL shown in your terminal (typically `http://localhost:8501`).

---

## 🕹 How to Use

### 1. Analysis Mode (default)

1. Select:
   - Tickers
   - Total capital
   - Forecast horizon (`1D`, `5D`, `1W`, `2W`, `1M`)
2. Optionally fine-tune:
   - Lookback window
   - Risk-free rate
   - Per-asset weight cap
   - End date
   - Prophet tuning & grid
   - News & sentiment
3. Click **"🚀 Run Analysis"**.

You will get:

- A concise **Strategy Recommendation Summary**:
  - Max-Sharpe portfolio (balanced risk/return)
  - Max-Return portfolio (aggressive)
  - Min-Variance portfolio (defensive)
  - Each with top allocations, expected horizon return, and 1D 95% VaR.
- Detailed charts:
  - Forecasted horizon returns per ticker
  - Stacked weights by strategy
  - Strategy-level returns & volatilities
- Tables:
  - Weights and allocations
  - Summary metrics (Return, Vol, VaR, Sharpe)
- Optional news & sentiment section, plus CSV / ZIP downloads.

### 2. 🤖 Agent Mode — QuantChat

Toggle **"🤖 Agent mode"** in the sidebar:

- Shows a clean ChatGPT-style layout:
  - You on the right, agent on the left.
  - “Reset Conversation” button to clear state.
- Ask quantitative questions such as:
  - “With \$100,000 and tickers AAPL, MSFT, NVDA, find the max_sharpe portfolio and show μ, σ, VaR.”
  - “For the next 1M, based on your tools, is AAPL or AMZN more attractive?”
  - “Here are my weights for AAPL, MSFT, AMZN — compute Sharpe and 1D 95% VaR.”
  - “Fetch recent TSLA news and score the short-term impact.”

**Scope & Guarantees**

- Only uses:
  - `get_prices`, `to_returns`
  - `prophet_expected_returns`
  - `cov_matrix`, `portfolio_volatility`, `portfolio_returns_series`, `var_historical`
  - `solve_min_variance`, `solve_max_return`, `solve_max_sharpe`
  - `evaluate_portfolio`, `compile_report`
  - `fetch_finviz_headlines`, `score_titles`
- Performs strict argument & type validation before every tool call.
- Never fabricates numeric outputs: if tools fail or inputs are missing, it explains instead of guessing.

---

## 🧠 Methodology (High Level)

- **Forecasting**: Prophet-based expected returns over discrete horizons.
- **Risk**:
  - Covariance matrix estimated from historical log returns.
  - Volatility via √(wᵀ Σ w).
  - Historical VaR at chosen confidence & horizon.
- **Optimization**:
  - Min-Variance: minimize σ.
  - Max-Return: maximize μ.
  - Max-Sharpe: maximize (μ − r_f) / σ.
  - Equal-Weight: simple baseline.
- **Explainability**:
  - Human-readable summary on top of charts.
  - All detailed tables & plots available for audit.

---

## 📄 License

MIT (or your chosen license).
