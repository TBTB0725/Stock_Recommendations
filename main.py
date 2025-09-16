# main.py
from __future__ import annotations

import argparse
import os
from datetime import datetime
import json

import numpy as np
import pandas as pd

from data import get_prices, to_returns
from forecast import prophet_expected_returns
from risk import cov_matrix
from optimize import (
    solve_min_variance,
    solve_max_return,
    solve_max_sharpe,
    equal_weight,
    Constraints,
)
from report import evaluate_portfolio, compile_report






def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Portfolio analysis with Prophet (growth) + VaR (risk)"
    )
    p.add_argument("--tickers", type=str, required=True,
                   help="Comma-separated tickers, e.g. 'AAPL,MSFT,TSLA'")
    p.add_argument("--capital", type=float, required=True,
                   help="Total investment capital, e.g. 100000")
    p.add_argument("--horizon", type=str, default="3M",
                   help="Forecast horizon for Prophet (1D,5D,1W,2W,1M,3M,6M,1Y). Default: 3M")
    p.add_argument("--lookback-days", type=int, default=252,
                   help="How many past business days to fetch. Default: 252 (~1Y)")
    p.add_argument("--var-alpha", type=float, default=0.05,
                   help="VaR alpha (0.05=95%% VaR, 0.01=99%%). Default: 0.05")
    p.add_argument("--var-horizon-days", type=int, default=1,
                   help="VaR horizon in trading days (1=1D, 5≈1W, 21≈1M). Default: 1")
    p.add_argument("--rf", type=float, default=0.0,
                   help="Annual risk-free rate, e.g. 0.02 for 2%%. Default: 0.0")
    p.add_argument("--max-weight-per-asset", type=float, default=None,
                   help="Optional upper bound per asset weight, e.g. 0.4")
    p.add_argument("--end", type=str, default=None,
                   help="End date (YYYY-MM-DD). If omitted, uses today (may move intraday).")
    # Optional stability knobs for Max Sharpe (multi-start)
    p.add_argument("--sharpe-restarts", type=int, default=0,
                   help="Number of additional random restarts for Max Sharpe (0 = disabled).")
    p.add_argument("--sharpe-seed", type=int, default=0,
                   help="Random seed for Max Sharpe multi-start (only used if restarts > 0).")
    p.add_argument("--save-csv", action="store_true",
                   help="If set, save summary/weights/allocation to ./outputs/")
    
    # Prophet hyper parameter/CV Configure
    p.add_argument("--prophet-tune", action="store_true",
                   help="Enable time-series cross validation + hyper-parameter tuning for Prophet")
    p.add_argument("--prophet-cv-metric", type=str, default="rmse",
                   choices=["mse","rmse","mae","mape","mdape","coverage"],
                   help="Metric used to pick best Prophet params (default: rmse)")
    p.add_argument("--prophet-cv-initial", type=int, default=None,
                   help="Initial window for Prophet CV (in TRADING days). Default: max(252, 3*horizon)")
    p.add_argument("--prophet-cv-period", type=int, default=None,
                   help="Step between Prophet CV cutoffs (in TRADING days). Default: max(21, horizon//2)")
    
    # Prophet param grid (either inline JSON or path to JSON file)
    p.add_argument("--prophet-grid", type=str, default=None,
                   help="Inline JSON for Prophet grid, e.g. '{\"n_changepoints\":[0,5],"
                        "\"changepoint_prior_scale\":[0.01,0.03,0.1],"
                        "\"weekly_seasonality\":[false],\"seasonality_mode\":[\"additive\"]}'")
    p.add_argument("--prophet-grid-file", type=str, default=None,
                   help="Path to a JSON file containing the Prophet param grid.")

    return p.parse_args()


def main():

    args = parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No valid tickers parsed from --tickers")
    capital = float(args.capital)
    horizon = args.horizon
    lookback = int(args.lookback_days)
    var_alpha = float(args.var_alpha)
    var_h_days = int(args.var_horizon_days)
    rf = float(args.rf)
    cons = Constraints(
        max_weight_per_asset=(None if args.max_weight_per_asset is None else float(args.max_weight_per_asset))
    )
    end = None if args.end is None else pd.to_datetime(args.end).date()







    # 1) Load data
    print(f"[1/6] Fetching prices for {tickers} over last {lookback} business days...")
    prices = get_prices(tickers, lookback_days=lookback, pad_ratio=2.0, end=end)

    # 2) Compute daily log returns for risk/VaR
    print("[2/6] Computing daily log returns...")
    rets_log = to_returns(prices, method="log")


    # 3) Prophet expected returns (annualized μ)
    # Parse Prophet param grid (inline JSON or file), optional
    param_grid = None
    if args.prophet_grid and args.prophet_grid_file:
        raise ValueError("Use either --prophet-grid or --prophet-grid-file, not both.")
    if args.prophet_grid:
        try:
            param_grid = json.loads(args.prophet_grid)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON passed to --prophet-grid: {e}") from e
    elif args.prophet_grid_file:
        if not os.path.exists(args.prophet_grid_file):
            raise FileNotFoundError(f"--prophet-grid-file not found: {args.prophet_grid_file}")
        with open(args.prophet_grid_file, "r", encoding="utf-8") as f:
            try:
                param_grid = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --prophet-grid-file: {e}") from e

    # Optional: light schema check (warn on unknown keys; Prophet will ignore but it's helpful)
    if param_grid is not None:
        allowed = {
            "n_changepoints",
            "changepoint_prior_scale",
            "weekly_seasonality",
            "yearly_seasonality",
            "daily_seasonality",
            "seasonality_mode",
        }
        unknown = set(param_grid.keys()) - allowed
        if unknown:
            print(f"[WARN] Unknown Prophet grid keys will be passed through: {sorted(unknown)}")

    print(f"[3/6] Forecasting annualized expected returns with Prophet (horizon={horizon})...")
    try:
        mu_annual = prophet_expected_returns(
            prices,
            horizon=horizon,
            tune=args.prophet_tune,
            cv_metric=args.prophet_cv_metric,
            cv_initial_days=args.prophet_cv_initial,
            cv_period_days=args.prophet_cv_period,
        )
    except KeyError as e:
        raise ValueError(
            f"Invalid --horizon '{horizon}'. Allowed: 1D, 5D, 1W, 2W, 1M, 3M, 6M, 1Y."
        ) from e

    # 4) Annualized covariance Σ
    print("[4/6] Estimating annualized covariance matrix...")
    Sigma_annual = cov_matrix(rets_log, annualize=True)

    # Align order across all inputs (critical!)
    prices       = prices[tickers]
    rets_log     = rets_log[tickers]
    mu_annual    = mu_annual.loc[tickers]
    Sigma_annual = Sigma_annual.loc[tickers, tickers]

    # 5) Optimize four strategies
    print("[5/6] Optimizing portfolios...")
    w_minvar = solve_min_variance(Sigma_annual, cons)
    w_maxret = solve_max_return(mu_annual, cons)
    w_sharpe = solve_max_sharpe(
        mu_annual, Sigma_annual, rf=rf, cons=cons,
        restarts=int(args.sharpe_restarts),  # multi-start (0 = disabled)
        seed=int(args.sharpe_seed),
    )
    w_equal  = equal_weight(len(tickers))

    # 6) Evaluate & report
    print("[6/6] Evaluating portfolios (annual metrics + VaR)...")
    results = []
    results.append(evaluate_portfolio("Min Variance", tickers, w_minvar, capital, mu_annual, Sigma_annual, rets_log,
                                      rf_annual=rf, var_alpha=var_alpha, var_horizon_days=var_h_days, log_returns=True))
    results.append(evaluate_portfolio("Max Return",   tickers, w_maxret, capital, mu_annual, Sigma_annual, rets_log,
                                      rf_annual=rf, var_alpha=var_alpha, var_horizon_days=var_h_days, log_returns=True))
    results.append(evaluate_portfolio("Max Sharpe",   tickers, w_sharpe, capital, mu_annual, Sigma_annual, rets_log,
                                      rf_annual=rf, var_alpha=var_alpha, var_horizon_days=var_h_days, log_returns=True))
    results.append(evaluate_portfolio("Equal Weight", tickers, w_equal,  capital, mu_annual, Sigma_annual, rets_log,
                                      rf_annual=rf, var_alpha=var_alpha, var_horizon_days=var_h_days, log_returns=True))

    summary, weights_tbl, alloc_tbl = compile_report(results)





    # Pretty print
    # 1) Clean tiny numerical noise (treat |x| < 1e-8 as 0 for weights)
    weights_tbl_clean = weights_tbl.copy()
    weights_tbl_clean = weights_tbl_clean.mask(weights_tbl_clean.abs() < 1e-8, 0.0)

    # 2) Formatting: convert summary metrics to percentages with fixed decimals
    summary_fmt = summary.copy()
    summary_fmt["ExpReturn(annual)"]  = (summary_fmt["ExpReturn(annual)"] * 100).round(2).astype(str) + "%"
    summary_fmt["Volatility(annual)"] = (summary_fmt["Volatility(annual)"] * 100).round(2).astype(str) + "%"
    # The VaR column name is dynamic; locate it and format as percentage
    var_cols = [c for c in summary_fmt.columns if c.startswith("VaR(")]
    if var_cols:
        vc = var_cols[0]
        summary_fmt[vc] = (summary_fmt[vc] * 100).round(2).astype(str) + "%"
    summary_fmt["Sharpe"] = summary_fmt["Sharpe"].round(3)

    # 3) Show weights as percentages
    weights_fmt = (weights_tbl_clean * 100).round(2).astype(str) + "%"

    # 4) Allocation: keep two decimals, avoid scientific notation
    alloc_fmt = alloc_tbl.round(2)

    # 5) Print tables
    print("\n=== Summary ===")
    print(summary_fmt.to_string())
    print("\n=== Weights (%) ===")
    print(weights_fmt.to_string())
    print("\n=== Allocation ($) ===")
    print(alloc_fmt.to_string())






    if args.save_csv:
        os.makedirs("outputs", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary.to_csv(f"outputs/summary_{stamp}.csv")
        weights_tbl.to_csv(f"outputs/weights_{stamp}.csv")
        alloc_tbl.to_csv(f"outputs/allocation_{stamp}.csv")
        print(f"\nSaved CSVs under ./outputs/ (timestamp {stamp})")


if __name__ == "__main__":
    main()
