# 📈 Stock Portfolio Optimization with Prophet + Portfolio Volatility + VaR

Analyze and optimize stock portfolios using Prophet for growth forecasts and Portfolio Volatility + VaR for risk assessment.

---

## 📦 Installation

Install dependencies:

    pip3 install numpy pandas yfinance prophet scipy tabulate

---

## 🚀 Usage

Run from terminal:
    for website - streamlit run app.py

    for teminal print - python3 main.py   --tickers AAPL,MSFT,GOOG,AMZN,TSLA,NVDA,META,IBM,ORCL,INTC,AMD,QCOM,CSCO,BA,CAT,KO,PEP,WMT,TGT,NKE   --capital 100000   --horizon 3M   --lookback-days 504   --var-alpha 0.05   --var-horizon-days 1   --rf 0.041   --max-weight-per-asset 0.2   --prophet-tune   --prophet-cv-metric rmse   --prophet-cv-initial 252   --prophet-cv-period 126   --end 2025-09-04   --sharpe-restarts 10   --prophet-grid '{"n_changepoints":[0,5],"changepoint_prior_scale":[0.01,0.03,0.1],"weekly_seasonality":[false],"yearly_seasonality": [False],"daily_seasonality": [False], "seasonality_mode":["additive"]}'  --save-csv

    python3 main.py \
      --tickers AAPL,MSFT,GOOG,AMZN \
      --capital 100000 \
      --horizon 3M \
      --lookback-days 252 \
      --var-alpha 0.05 \ 
      --var-horizon-days 1 \
      --rf 0.041 \ 
      --max-weight-per-asset 0.4 \ 
      --prophet-tune \
      --prophet-cv-metric rmse \
      --prophet-cv-initial 252 \
      --prophet-cv-period 126
      --end 2025-09-04 \ 
      --sharpe-restarts 10 \ 
      --prophet-grid '{"n_changepoints":[0,5],"changepoint_prior_scale":[0.01,0.03,0.1],"weekly_seasonality":[false],"yearly_seasonality": [False],"daily_seasonality": [False], "seasonality_mode":["additive"]}' \
      --save-csv

---

## ⚙️ Parameters

Parameter                  | Description
-------------------------- | --------------------------------------------------------------------------------------------
--tickers                  | Comma-separated stock tickers (e.g. AAPL,MSFT,GOOG)
--capital                  | Total capital to invest (e.g. 100000)
--horizon                  | Forecast horizon (e.g. 1M, 3M, 6M, 1Y)
--lookback-days            | Historical lookback in business days (e.g. 252)
--var-alpha                | VaR alpha (e.g. 0.05=95%, 0.01=99%)
--var-horizon-days         | VaR horizon in trading days (e.g. 1=1D, 5≈1W, 21≈1M)
--rf                       | Annual risk-free rate (e.g. 0.041)
--max-weight-per-asset     | Max allocation per stock (e.g. 0.4)
--prophet-tune             | Enable Prophet hyper-parameter tuning with time-series cross-validation
--prophet-cv-metric        | Metric to select best Prophet params (mse, rmse, mae, mape, mdape, coverage). Default=rmse
--prophet-cv-initial       | Initial training window for Prophet CV (in TRADING days). Default=max(252, 3*horizon)
--prophet-cv-period        | Step size between Prophet CV cutoffs (in TRADING days). Default=max(21, horizon//2)
--end                      | End date (YYYY-MM-DD). Default = today
--sharpe-restarts          | Number of random restarts for Max Sharpe optimization
--save-csv                 | Save results under ./outputs/
--prophet-grid             | Inline JSON defining Prophet param grid
--prophet-grid-file        | Path to JSON file containing Prophet param grid (alternative to --prophet-grid)

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
- Hyper-parameter tuning with time-series cross-validation (CV)
- Risk Covariance matrix + historical Value-at-Risk
- Min Variance (minimize risk), Max Return (maximize return), Max Sharpe (maximize risk-adjusted return), Equal Weight (baseline)

