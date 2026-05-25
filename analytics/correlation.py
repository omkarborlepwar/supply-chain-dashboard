import pandas as pd
import numpy as np


def marketing_sales_correlation(marketing, orders):
    daily_sales = orders.groupby("order_date")["total"].sum().reset_index()
    daily_sales.columns = ["date", "revenue"]

    daily_mkt = marketing.groupby("date")[["spend", "impressions", "clicks", "conversions"]].sum().reset_index()

    merged = daily_mkt.merge(daily_sales, on="date", how="inner")

    corr = merged[["spend", "revenue", "impressions", "clicks", "conversions"]].corr()

    merged["spend_lag1"] = merged["spend"].shift(1)
    merged["spend_lag2"] = merged["spend"].shift(2)
    merged["spend_lag3"] = merged["spend"].shift(3)

    lag_corr = pd.DataFrame({
        "lag": ["Same Day", "1 Day Lag", "2 Day Lag", "3 Day Lag"],
        "correlation": [
            merged["spend"].corr(merged["revenue"]),
            merged["spend_lag1"].corr(merged["revenue"]),
            merged["spend_lag2"].corr(merged["revenue"]),
            merged["spend_lag3"].corr(merged["revenue"]),
        ]
    })

    return corr, lag_corr, merged


def channel_roi(marketing, orders):
    daily_sales = orders.groupby("order_date")["total"].sum().reset_index()
    daily_sales.columns = ["date", "revenue"]

    merged = marketing.merge(daily_sales, on="date", how="left")

    channel_perf = merged.groupby("channel").agg(
        total_spend=("spend", "sum"),
        total_revenue=("revenue", "sum"),
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        total_conversions=("conversions", "sum"),
    ).reset_index()

    channel_perf["ROAS"] = (channel_perf["total_revenue"] / channel_perf["total_spend"].replace(0, np.nan)).round(2)
    channel_perf["CPC"] = (channel_perf["total_spend"] / channel_perf["total_clicks"].replace(0, np.nan)).round(2)
    channel_perf["conversion_rate"] = (
        channel_perf["total_conversions"] / channel_perf["total_clicks"].replace(0, np.nan) * 100
    ).round(2)

    return channel_perf


def category_correlation(inventory, orders, products):
    cat_sales = orders.groupby(["order_date", "category"])["total"].sum().reset_index()
    pivot = cat_sales.pivot_table(index="order_date", columns="category", values="total", fill_value=0)

    if len(pivot.columns) > 1:
        corr_matrix = pivot.corr()
    else:
        corr_matrix = pd.DataFrame([[1.0]], index=pivot.columns, columns=pivot.columns)

    return corr_matrix, pivot
