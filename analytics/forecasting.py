import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_demand_series(orders, product_id):
    daily = orders[orders["product_id"] == product_id].copy()
    daily = daily.set_index("order_date").resample("D")["quantity"].sum()
    daily = daily.asfreq("D", fill_value=0)
    return daily


def arima_forecast(series, forecast_days=30):
    if len(series) < 30:
        return None, None

    train = series[:-30]
    test = series[-30:]

    try:
        model = ARIMA(train, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_days)
        forecast = np.maximum(forecast, 0)

        if len(test) > 0:
            in_sample = model_fit.predict(start=0, end=len(train) - 1)
            mae = mean_absolute_error(test[-min(30, len(test)):],
                                       forecast[:min(30, len(forecast))])
            rmse = np.sqrt(mean_squared_error(test[-min(30, len(test)):],
                                              forecast[:min(30, len(forecast))]))
        else:
            mae = None
            rmse = None

        return forecast, {"mae": round(mae, 2) if mae else None,
                          "rmse": round(rmse, 2) if rmse else None}

    except Exception:
        fallback = np.full(forecast_days, series.mean())
        return fallback, {"mae": None, "rmse": None, "fallback": True}


def demand_volatility(orders):
    daily = orders.groupby(["product_id", "order_date"])["quantity"].sum().reset_index()
    daily.columns = ["product_id", "date", "quantity"]
    vol = daily.groupby("product_id")["quantity"].agg(["mean", "std"]).reset_index()
    vol.columns = ["product_id", "mean_daily", "std_daily"]
    vol["cv"] = (vol["std_daily"] / vol["mean_daily"].replace(0, np.nan) * 100).round(1)
    return vol


def lead_time_demand(orders, products):
    merged = orders.merge(products[["product_id", "lead_time_days"]], on="product_id")
    merged["demand_during_lead_time"] = merged["lead_time_days"] * merged["quantity"]
    lt_demand = merged.groupby("product_id").agg(
        avg_lead_time_demand=("demand_during_lead_time", "mean"),
    ).reset_index()
    return lt_demand
