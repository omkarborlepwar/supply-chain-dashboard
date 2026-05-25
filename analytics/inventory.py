import pandas as pd
import numpy as np


def load_data(data_dir="data"):
    products = pd.read_csv(f"{data_dir}/products.csv")
    orders = pd.read_csv(f"{data_dir}/orders.csv", parse_dates=["order_date"])
    inventory = pd.read_csv(f"{data_dir}/inventory.csv", parse_dates=["date"])
    deliveries = pd.read_csv(f"{data_dir}/deliveries.csv", parse_dates=["shipped_date", "delivered_date"])
    marketing = pd.read_csv(f"{data_dir}/marketing.csv", parse_dates=["date"])
    return products, orders, inventory, deliveries, marketing


def inventory_turnover(inventory, products):
    monthly = inventory.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly = monthly.groupby(["product_id", "product_name", "month"]).agg(
        avg_stock=("stock_on_hand", "mean"),
        total_sold=("daily_sold", "sum"),
    ).reset_index()
    monthly["turnover_ratio"] = monthly["total_sold"] / monthly["avg_stock"].replace(0, np.nan)
    monthly = monthly.merge(products[["product_id", "category"]], on="product_id")
    return monthly


def stockout_analysis(inventory):
    product_stats = inventory.groupby(["product_id", "product_name"]).agg(
        total_days=("date", "count"),
        stockout_days=("stockout", "sum"),
        avg_stock=("stock_on_hand", "mean"),
    ).reset_index()
    product_stats["stockout_rate"] = (product_stats["stockout_days"] / product_stats["total_days"] * 100).round(2)
    product_stats["days_above_reorder"] = product_stats["avg_stock"].apply(
        lambda x: "Low" if x < 30 else ("Medium" if x < 80 else "Healthy")
    )
    return product_stats


def slow_moving_analysis(inventory, products):
    daily = inventory.groupby(["product_id", "product_name"]).agg(
        avg_daily_sold=("daily_sold", "mean"),
        avg_stock=("stock_on_hand", "mean"),
    ).reset_index()
    daily["days_of_inventory"] = (daily["avg_stock"] / daily["avg_daily_sold"].replace(0, np.nan)).round(1)
    daily = daily.merge(products[["product_id", "category", "cost"]], on="product_id")
    daily["holding_cost_est"] = (daily["avg_stock"] * daily["cost"] * 0.25 / 365).round(2)
    return daily


def category_performance(inventory, products):
    merged = inventory.merge(products[["product_id", "price"]], on="product_id")
    monthly = merged.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    cat_perf = monthly.groupby(["category", "month"]).agg(
        total_sold=("daily_sold", "sum"),
        avg_stock=("stock_on_hand", "mean"),
        revenue=("daily_sold", lambda x: (x * monthly.loc[x.index, "price"]).sum()),
    ).reset_index()
    return cat_perf
