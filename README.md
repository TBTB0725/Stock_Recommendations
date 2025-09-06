# 📈 Stock Portfolio Optimization with Prophet + VaR

Analyze and optimize stock portfolios using Prophet for growth forecasts and Value-at-Risk (VaR) for risk assessment.

---

## 📦 Installation

Install dependencies:

    pip install numpy pandas yfinance prophet scipy tabulate

---

## 🚀 Usage

Run from terminal:

    python3 main.py \
      --tickers AAPL,MSFT,GOOG,AMZN \
      --capital 100000 \
      --horizon 3M \
      --lookback-days 252 \
      --confidence 0.95

---

## ⚙️ Parameters

Parameter          | Description
------------------ | -------------------------------------------
--tickers          | Comma-separated list of stock symbols
--capital          | Total capital to invest (e.g. 100000)
--horizon          | Forecast horizon (e.g. 3M, 6M)
--lookback-days    | Number of historical days to consider
--confidence       | VaR confidence level (e.g. 0.95)

---

## 📊 Output

Generates and prints:

- ✅ Minimum Risk Portfolio
- 📈 Maximum Return Portfolio
- 🔁 Sharpe Ratio Optimized Portfolio
- ⚖️ Equally Weighted Portfolio (Baseline)

Each portfolio includes:

- Ticker weights
- Expected return
- Standard deviation
- VaR at specified confidence level

---

## 📁 Example Output

    Portfolio: Sharpe Ratio Optimal Portfolio
    ------------------------------------------
    Ticker   Weight    Expected Return    Std Dev     VaR (95%)
    AAPL     0.30      0.080              0.18        -0.12
    MSFT     0.40      0.085              0.20        -0.14
    GOOG     0.20      0.090              0.22        -0.15
    AMZN     0.10      0.070              0.25        -0.16

---

## 🧠 Methodology

- Growth Forecasts: Meta Prophet
- Risk Calculation: Historical simulation Value-at-Risk (VaR)
- Optimization: Scipy to maximize Sharpe ratio and minimize volatility

---

## 🔍 Notes

- Ensure Prophet install matches Python version (may need pystan)
- Use custom tickers and time horizons freely

---

## 📚 License

MIT License
