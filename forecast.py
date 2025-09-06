# forecast.py
import numpy as np
import pandas as pd
from prophet import Prophet

_H = {"1D":1,"5D":5,"1W":5,"2W":10,"1M":21,"3M":63,"6M":126,"1Y":252}

def prophet_expected_returns(prices: pd.DataFrame, horizon: str = "3M") -> pd.Series:
    """
    Use Prophet to forecast each stock's daily log returns and return the annualized expected return μ.
    prices: rows = dates (business days), columns = tickers (recommended to use data.get_prices)
    horizon: forecast horizon ('1M','3M','6M','1Y', etc.)
    """
    days = _H[horizon.upper()]
    tdpy = 252
    out = {}

    for t in prices.columns:
        p = prices[t].dropna()
        # Daily log returns (requires prices > 0)
        r = np.log(p).diff().dropna()

        df = pd.DataFrame({"ds": r.index, "y": r.values})
        try:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                n_changepoints=0,             
                changepoint_prior_scale=0.001, 
                random_state=0,    
            )
        except TypeError:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                n_changepoints=0,
                changepoint_prior_scale=0.001,
            )
        m.fit(df)

        future = m.make_future_dataframe(periods=days, freq="B")
        fcst = m.predict(future).iloc[-days:]

        # Sum of log returns over the horizon → convert to simple return
        r_horizon = np.expm1(fcst["yhat"].sum())
        # Annualize
        mu_annual = (1.0 + r_horizon) ** (tdpy / days) - 1.0
        out[t] = float(mu_annual)

    return pd.Series(out, index=prices.columns, name="mu_annual")
