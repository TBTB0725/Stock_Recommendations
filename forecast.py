# forecast.py
import numpy as np
import pandas as pd
from prophet import Prophet
from data import to_returns
from risk import TRADING_DAYS_PER_YEAR as tdpy
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product
from typing import Dict, List, Optional, Tuple
import math

_DAILY_MEAN_CLIP   = 0.003  
_DAILY_MEAN_SHRINK = 0.5

_H = {"1D":1,"5D":5,"1W":5,"2W":10,"1M":21,"3M":63,"6M":126,"1Y":252}



def _grid_dict_to_list(param_grid: Dict[str, List]) -> List[Dict]:
    """Convert {'a':[1,2], 'b':[3]} to [{'a':1,'b':3}, {'a':2,'b':3}]"""
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = []
    for tpl in product(*vals):
        combos.append({k: v for k, v in zip(keys, tpl)})
    return combos


def _trading_days_to_calendar(days: int) -> int:
    """Prophet's CV use 7 days, so need to convert"""
    return int(math.ceil(days * 7.0 / 5.0))


def _cv_score_for_model(
    m: Prophet,
    horizon_days_trading: int,
    initial_days_trading: Optional[int] = None,
    period_days_trading: Optional[int] = None,
    metric: str = "rmse",
) -> float:
    """
    Give model CV test, return score
    metric: 'mse','rmse','mae','mape','mdape','coverage'
    """
    horizon_cal = _trading_days_to_calendar(horizon_days_trading)
    initial_cal = _trading_days_to_calendar(initial_days_trading) if initial_days_trading else None
    period_cal  = _trading_days_to_calendar(period_days_trading)  if period_days_trading  else None

    horizon_str = f"{horizon_cal} days"
    initial_str = f"{initial_cal} days" if initial_cal else None
    period_str  = f"{period_cal} days"  if period_cal  else None

    df_cv = cross_validation(
        model=m,
        horizon=horizon_str,
        period=period_str,
        initial=initial_str,
        parallel="processes",
        disable_tqdm=True,
    )
    df_pm = performance_metrics(df_cv, rolling_window=1)
    if metric not in df_pm.columns:
        raise ValueError(f"Unsupported metric '{metric}'. Available: {list(df_pm.columns)}")
    return float(df_pm[metric].mean())


def _fit_prophet_on_returns(df: pd.DataFrame, params: Dict) -> Prophet:
    """
    Fit Prophet with the passed hyperparameters
    """
    m = Prophet(
        daily_seasonality=params.get("daily_seasonality", False),
        weekly_seasonality=params.get("weekly_seasonality", False),
        yearly_seasonality=params.get("yearly_seasonality", False),
        seasonality_mode=params.get("seasonality_mode", "additive"),
        n_changepoints=params.get("n_changepoints", 0),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.001),
    )
    m.fit(df)
    return m


def _choose_best_params_with_cv(
    df: pd.DataFrame,
    horizon_days_trading: int,
    param_grid: Optional[Dict[str, List]] = None,
    initial_days_trading: Optional[int] = None,
    period_days_trading: Optional[int] = None,
    metric: str = "rmse",
) -> Tuple[Dict, float]:
    """
    Simple grid search based on time series cross-validation.
    Returns (best_params, best_score); smaller scores are better.
    """
    def _normalize_param_grid(grid: Dict) -> Dict[str, List]:
        """Ensure every value is a list; keep unknown keys to allow future Prophet args."""
        norm = {}
        for k, v in grid.items():
            if isinstance(v, list):
                norm[k] = v
            else:
                norm[k] = [v]
        return norm
    
    if not param_grid:
        param_grid = {
            "n_changepoints": [0],
            "changepoint_prior_scale": [1e-4, 3e-4, 1e-3],
            "weekly_seasonality": [False],
            "yearly_seasonality": [False],
            "daily_seasonality": [False],
            "seasonality_mode": ["additive"],
        }

        #param_grid = {
        #    "n_changepoints": [0, 5, 10],
        #    "changepoint_prior_scale": [0.003, 0.01, 0.03, 0.1],
        #    "weekly_seasonality": [False, True], 
        #    "yearly_seasonality": [False],
        #    "daily_seasonality": [False],
        #    "seasonality_mode": ["additive"],
        #}

    param_grid = _normalize_param_grid(param_grid)
    combos = _grid_dict_to_list(param_grid)

    # For each set of parameters: first fit the full value once, then use Prophet's built-in cutoffs to do rolling refitting for CV    
    best_params, best_score = None, float("inf")
    for params in combos:
        m = _fit_prophet_on_returns(df, params)
        try:
            score = _cv_score_for_model(
                m,
                horizon_days_trading=horizon_days_trading,
                initial_days_trading=initial_days_trading,
                period_days_trading=period_days_trading,
                metric=metric,
            )
        except Exception as e:
            continue
        if score < best_score:
            best_score, best_params = score, params

    if best_params is None:
        best_params = {
            "n_changepoints": 0,
            "changepoint_prior_scale": 0.001,
            "weekly_seasonality": False,
            "yearly_seasonality": False,
            "daily_seasonality": False,
            "seasonality_mode": "additive",
        }
        best_score = float("inf")

    return best_params, best_score


def prophet_expected_returns(
    prices: pd.DataFrame,
    horizon: str = "3M",
    tune: bool = False,
    cv_metric: str = "rmse",
    cv_initial_days: Optional[int] = None,
    cv_period_days: Optional[int] = None,
    param_grid: Optional[Dict[str, List]] = None,
    min_points_for_cv: int = 100,
    annualize: bool = False,  
) -> pd.Series:
    """
    Use Prophet to forecast each stock's daily log returns and return the annualized expected return μ.
    prices: rows = dates (business days), columns = tickers (recommended to use data.get_prices)
    horizon: forecast horizon ('1M','3M','6M','1Y', etc.)
    tune: whether to enable time-series cross-validation + hyper-parameter tuning
    cv_metric: metric used to pick best params ('rmse','mse','mae','mape','mdape','coverage')
    cv_initial_days: initial window size for CV (trading days). If None, auto:
                     max(252, 3*horizon)
    cv_period_days:  step between CV cutoffs (trading days). If None, auto: max(21, horizon//2)
    param_grid: dict of Prophet constructor hyper-params lists to grid search
    min_points_for_cv: if series has fewer points than this, skip tuning
    """
    days = _H[horizon.upper()]
    out = {}

    for t in prices.columns:
        p = prices[t].dropna()
        
        if (p <= 0).any():
            raise ValueError(f"Non-positive prices for {t}; log-returns require positive prices.")
        
        r = np.log(p).diff().dropna()
        
        mu_hist = float(r.mean())
        r_demean = r - mu_hist
        df = pd.DataFrame({"ds": r_demean.index, "y": r_demean.values})

        do_tune = tune and (len(df) >= min_points_for_cv)

        if do_tune:
            if cv_initial_days is None:
                cv_initial_days_eff = max(252, 3 * days)
            else:
                cv_initial_days_eff = int(cv_initial_days)
            if cv_period_days is None:
                cv_period_days_eff = max(21, days // 2)
            else:
                cv_period_days_eff = int(cv_period_days)

            best_params, _ = _choose_best_params_with_cv(
                df=df,
                horizon_days_trading=days,
                param_grid=param_grid,
                initial_days_trading=cv_initial_days_eff,
                period_days_trading=cv_period_days_eff,
                metric=cv_metric,
            )
            m = _fit_prophet_on_returns(df, best_params)
        else:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                n_changepoints=0,
                changepoint_prior_scale=1e-4,
                seasonality_mode="additive",
            )
            m.fit(df)


        future = m.make_future_dataframe(periods=days, freq="B")
        fcst = m.predict(future).iloc[-days:]

        yhat_dev = fcst["yhat"].values

        yhat_log = yhat_dev + mu_hist

        yhat_daily_mean = float(np.mean(yhat_log))
        yhat_daily_mean = _DAILY_MEAN_SHRINK * yhat_daily_mean
        yhat_daily_mean = float(np.clip(yhat_daily_mean, -_DAILY_MEAN_CLIP, _DAILY_MEAN_CLIP))

        r_window = float(math.expm1(yhat_daily_mean * days))

        if annualize:
            mu_annual = math.expm1(yhat_daily_mean * tdpy)
            out[t] = float(mu_annual)
        else:
            out[t] = r_window
    
    name = "mu_annual" if annualize else f"ret_{days}d"
    return pd.Series(out, index=prices.columns, name=name)
