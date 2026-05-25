import pandas as pd
import numpy as np


def fulfillment_kpis(orders, deliveries):
    orders_clean = orders.drop(columns=["region"], errors="ignore")
    merged = orders_clean.merge(deliveries, on="order_id", how="left")
    merged = merged[merged["delivery_status"] == "Delivered"].copy()

    merged["delivery_time_days"] = (merged["delivered_date"] - merged["shipped_date"]).dt.days
    merged["order_to_delivery_days"] = (merged["delivered_date"] - merged["order_date"]).dt.days

    carrier_stats = merged.groupby("carrier").agg(
        total_orders=("order_id", "count"),
        avg_delivery_days=("delivery_time_days", "mean"),
        median_delivery_days=("delivery_time_days", "median"),
        total_shipping_cost=("shipping_cost", "sum"),
    ).reset_index()
    carrier_stats["avg_delivery_days"] = carrier_stats["avg_delivery_days"].round(1)

    region_stats = merged.groupby("region").agg(
        total_orders=("order_id", "count"),
        avg_delivery_days=("delivery_time_days", "mean"),
        on_time_delivery=("delivery_time_days", lambda x: (x <= 5).sum()),
    ).reset_index()
    region_stats["on_time_rate"] = (
        region_stats["on_time_delivery"] / region_stats["total_orders"] * 100
    ).round(1)

    merged["month"] = merged["order_date"].dt.to_period("M").astype(str)
    monthly = merged.groupby("month").agg(
        total_orders=("order_id", "count"),
        avg_delivery_days=("delivery_time_days", "mean"),
    ).reset_index()
    monthly["avg_delivery_days"] = monthly["avg_delivery_days"].round(1)

    return merged, carrier_stats, region_stats, monthly


def fulfillment_rate(orders, deliveries):
    total = len(orders)
    delivered = deliveries[deliveries["delivery_status"] == "Delivered"]["order_id"].nunique()
    return round(delivered / total * 100, 1) if total > 0 else 0


def shipping_cost_analysis(orders, deliveries):
    merged = orders.merge(deliveries, on="order_id")
    merged["shipping_ratio"] = (merged["shipping_cost"] / merged["total"] * 100).round(1)
    return merged
