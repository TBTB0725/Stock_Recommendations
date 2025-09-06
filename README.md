Portfolio Optimization with Prophet + portfolio volatility + VaR

Installation:

numpy
pandas
yfinance
prophet
scipy
tabulate

Usage:

Run from terminal:
python3 main.py
--tickers AAPL,MSFT,GOOG,AMZN
--capital 100000
--horizon 3M
--lookback-days 252
--var-alpha 0.05
--var-horizon-days 1
--rf 0.041
--max-weight-per-asset 0.4
--end 2025-09-04
--sharpe-restarts 10
--save-csv

Input Parameters:

--tickers Comma-separated stock tickers, e.g. AAPL,MSFT,GOOG
--capital Total investment capital, e.g. 100000
--horizon Forecast horizon (1M, 3M, 6M, 1Y)
--lookback-days Historical lookback in business days, e.g. 252
--var-alpha VaR alpha (0.05=95%, 0.01=99%)
--var-horizon-days VaR horizon in trading days (1=1D, 5≈1W, 21≈1M)
--rf Annual risk-free rate, e.g. 0.041
--max-weight-per-asset Max allocation per stock, e.g. 0.4
--end End date (YYYY-MM-DD). Default = today
--sharpe-restarts Number of random restarts for Max Sharpe optimization
--save-csv Save results under ./outputs/

Outputs:

Tables printed in terminal (Summary, Weights, Allocation)

CSV files saved under ./outputs/ (if --save-csv is enabled):
summary_<timestamp>.csv
weights_<timestamp>.csv
allocation_<timestamp>.csv

Methods:

Forecasting Prophet predicts expected log-returns, annualized
Risk Covariance matrix + historical Value-at-Risk
Min Variance (minimize risk), Max Return (maximize return),
Max Sharpe (maximize risk-adjusted return),
Equal Weight (baseline) 
