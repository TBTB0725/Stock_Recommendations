# 📈 Stock Portfolio Optimization with Prophet + Portfolio Volatility + VaR

Analyze and optimize stock portfolios using Prophet for growth forecasts and Portfolio Volatility + VaR for risk assessment.

---

## 📦 Installation

Install dependencies:

    pip3 install numpy pandas yfinance prophet scipy tabulate

---

## 🚀 Usage

Run from terminal:

    python3 main.py \
      --tickers AAPL,MSFT,GOOG,AMZN \
      --capital 100000 \
      --horizon 3M \
      --lookback-days 252 \
      --confidence 0.95 \
      --var-alpha 0.05 \ 
      --var-horizon-days 1 \
      --rf 0.041 \ 
      --max-weight-per-asset 0.4 \ 
      --end 2025-09-04 \ 
      --sharpe-restarts 10 \ 
      --save-csv

    python3 main.py   --tickers AAPL,MSFT,GOOG,AMZN,TSLA,NVDA,META,IBM,ORCL,INTC,AMD,QCOM,CSCO,BA,CAT,KO,PEP,WMT,TGT,NKE   --capital 100000   --horizon 3M   --lookback-days 504   --var-alpha 0.05   --var-horizon-days 1   --rf 0.041   --max-weight-per-asset 0.2   --end 2025-09-04   --sharpe-restarts 10   --save-csv

---

## ⚙️ Parameters

Parameter              | Description
------------------     | -------------------------------------------
--tickers              | Comma-separated stock tickers (e.g. AAPL,MSFT,GOOG)
--capital              | Total capital to invest (e.g. 100000)
--horizon              | Forecast horizon (e.g. 1M, 3M, 6M, 1Y)
--lookback-days        | Historical lookback in business days (e.g. 252)
--var-alpha            | VaR alpha (e.g. 0.05=95%, 0.01=99%)
--var-horizon-days     | VaR horizon in trading days (e.g. 1=1D, 5≈1W, 21≈1M)
--rf                   | Annual risk-free rate (e.g. 0.041)
--max-weight-per-asset | Max allocation per stock (e.g. 0.4)
--end                  | End date (YYYY-MM-DD). Default = today
--sharpe-restarts      | Number of random restarts for Max Sharpe optimization
--save-csv             | Save results under ./outputs/

---

## 📊 Output

Generates and prints:

- ✅ Minimum Risk
- 📈 Maximum Return
- 🔁 Sharpe Ratio Optimized
- ⚖️ Equally Weighted (Baseline)

Each one includes:
- Ticker weights
- Ticker allocations
- Expected return (annual)
- Volatility(annual)
- VaR at specified confidence level
- Sharpe

---


## 🧠 Methodology

- Forecasting Prophet predicts expected log-returns, annualized
- Risk Covariance matrix + historical Value-at-Risk
- Min Variance (minimize risk), Max Return (maximize return), Max Sharpe (maximize risk-adjusted return), Equal Weight (baseline)

