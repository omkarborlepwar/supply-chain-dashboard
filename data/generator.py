import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

PRODUCTS = {
    "Wireless Headphones": {"category": "Electronics", "cost": 45, "price": 120, "lead_time": 7},
    "Smart Watch": {"category": "Electronics", "cost": 80, "price": 200, "lead_time": 10},
    "Bluetooth Speaker": {"category": "Electronics", "cost": 30, "price": 80, "lead_time": 5},
    "USB-C Hub": {"category": "Electronics", "cost": 15, "price": 45, "lead_time": 4},
    "Cotton T-Shirt": {"category": "Fashion", "cost": 8, "price": 25, "lead_time": 6},
    "Denim Jacket": {"category": "Fashion", "cost": 35, "price": 90, "lead_time": 8},
    "Running Shoes": {"category": "Fashion", "cost": 40, "price": 110, "lead_time": 9},
    "Leather Belt": {"category": "Fashion", "cost": 10, "price": 35, "lead_time": 5},
    "Yoga Mat": {"category": "Sports", "cost": 12, "price": 35, "lead_time": 4},
    "Dumbbell Set": {"category": "Sports", "cost": 25, "price": 65, "lead_time": 7},
    "Resistance Bands": {"category": "Sports", "cost": 8, "price": 20, "lead_time": 3},
    "Protein Powder 1kg": {"category": "Sports", "cost": 28, "price": 55, "lead_time": 6},
    "Ceramic Mug Set": {"category": "Home", "cost": 10, "price": 30, "lead_time": 5},
    "LED Desk Lamp": {"category": "Home", "cost": 18, "price": 50, "lead_time": 6},
    "Throw Pillow Set": {"category": "Home", "cost": 14, "price": 40, "lead_time": 7},
    "Plant Pot Trio": {"category": "Home", "cost": 12, "price": 35, "lead_time": 4},
}

REGIONS = ["North", "South", "East", "West"]
CARRIERS = ["Delhivery", "Blue Dart", "DTDC", "Ecom Express"]
CHANNELS = ["Google Ads", "Facebook", "Instagram", "Email", "Organic"]


def generate_products():
    rows = []
    for i, (name, attrs) in enumerate(PRODUCTS.items(), 1):
        rows.append({
            "product_id": i,
            "product_name": name,
            "category": attrs["category"],
            "cost": attrs["cost"],
            "price": attrs["price"],
            "margin": round((attrs["price"] - attrs["cost"]) / attrs["price"] * 100, 1),
            "lead_time_days": attrs["lead_time"],
            "reorder_point": random.randint(20, 60),
            "safety_stock": random.randint(10, 30),
            "supplier_latency": round(random.uniform(0.5, 2.0), 1),
        })
    return pd.DataFrame(rows)


def seasonal_factor(date):
    """Simulate seasonal demand patterns."""
    month = date.month
    # Higher in Nov-Dec (holiday), lower in Jan-Feb
    base = 1.0 + 0.3 * np.sin((month - 1) * np.pi / 6)
    # Weekend boost
    base *= 1.15 if date.weekday() >= 5 else 1.0
    # Random noise
    base *= random.uniform(0.85, 1.15)
    return base


def generate_orders(products_df, start_date="2025-01-01", days=365):
    end = pd.Timestamp(start_date) + timedelta(days=days - 1)
    dates = pd.date_range(start_date, end, freq="D")

    orders = []
    order_id = 1000

    for date in dates:
        sf = seasonal_factor(date)
        for _, prod in products_df.iterrows():
            base_daily = random.uniform(3, 15) if prod["category"] == "Electronics" else random.uniform(5, 25)
            qty = max(1, int(np.random.poisson(base_daily * sf)))
            if qty == 0:
                continue
            for _ in range(qty):
                order_id += 1
                qty_ordered = random.randint(1, 4)
                unit_price = prod["price"] * random.uniform(0.9, 1.0)
                orders.append({
                    "order_id": order_id,
                    "order_date": date,
                    "product_id": prod["product_id"],
                    "product_name": prod["product_name"],
                    "category": prod["category"],
                    "quantity": qty_ordered,
                    "unit_price": round(unit_price, 2),
                    "total": round(qty_ordered * unit_price, 2),
                    "cost": round(qty_ordered * prod["cost"], 2),
                    "region": random.choice(REGIONS),
                })
    df = pd.DataFrame(orders)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def generate_deliveries(orders_df):
    deliveries = []
    for _, order in orders_df.iterrows():
        shipped = order["order_date"] + timedelta(
            days=random.randint(0, 3)
        )
        delay = max(0, int(np.random.exponential(0.5)))
        delivered = shipped + timedelta(days=random.randint(2, 6) + delay)
        status = "Delivered" if delivered < pd.Timestamp("2025-12-30") else "In Transit"
        if delivered > pd.Timestamp("2025-12-31"):
            status = "In Transit"

        deliveries.append({
            "order_id": order["order_id"],
            "shipped_date": shipped,
            "delivered_date": delivered if status == "Delivered" else pd.NaT,
            "carrier": random.choice(CARRIERS),
            "region": order["region"],
            "delivery_status": status,
            "shipping_cost": round(random.uniform(20, 80), 2),
            "distance_km": random.randint(50, 2000),
        })
    return pd.DataFrame(deliveries)


def generate_inventory(products_df, orders_df, start_date="2025-01-01", days=365):
    end = pd.Timestamp(start_date) + timedelta(days=days - 1)
    dates = pd.date_range(start_date, end, freq="D")

    initial_stock = {p["product_id"]: random.randint(150, 400) for _, p in products_df.iterrows()}
    stock = initial_stock.copy()
    inventory_records = []

    for date in dates:
        for _, prod in products_df.iterrows():
            pid = prod["product_id"]
            daily_sales = orders_df[
                (orders_df["product_id"] == pid) & (orders_df["order_date"] == date)
            ]["quantity"].sum()

            stock[pid] = max(0, stock[pid] - daily_sales)

            restock = 0
            if stock[pid] <= prod["reorder_point"]:
                restock = random.randint(80, 200)
                stock[pid] += restock

            inventory_records.append({
                "date": date,
                "product_id": pid,
                "product_name": prod["product_name"],
                "category": prod["category"],
                "stock_on_hand": stock[pid],
                "daily_sold": int(daily_sales),
                "restock_quantity": restock,
                "stockout": 1 if stock[pid] == 0 else 0,
            })
    df = pd.DataFrame(inventory_records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_marketing(start_date="2025-01-01", days=365):
    end = pd.Timestamp(start_date) + timedelta(days=days - 1)
    dates = pd.date_range(start_date, end, freq="D")

    records = []
    for date in dates:
        for channel in CHANNELS:
            base_spend = {
                "Google Ads": random.uniform(500, 2000),
                "Facebook": random.uniform(300, 1500),
                "Instagram": random.uniform(200, 1200),
                "Email": random.uniform(50, 300),
                "Organic": 0,
            }[channel]
            spend = base_spend * seasonal_factor(date)
            if channel == "Organic":
                impressions = int(random.uniform(2000, 8000) * seasonal_factor(date))
                clicks = int(impressions * random.uniform(0.02, 0.06))
            else:
                impressions = int(spend * random.uniform(30, 60))
                clicks = int(impressions * random.uniform(0.01, 0.05))

            records.append({
                "date": date,
                "channel": channel,
                "spend": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": int(clicks * random.uniform(0.02, 0.10)),
            })
    return pd.DataFrame(records)


def generate_all(output_dir="."):
    products = generate_products()
    orders = generate_orders(products)
    deliveries = generate_deliveries(orders)
    inventory = generate_inventory(products, orders)
    marketing = generate_marketing()

    products.to_csv(f"{output_dir}/products.csv", index=False)
    orders.to_csv(f"{output_dir}/orders.csv", index=False)
    deliveries.to_csv(f"{output_dir}/deliveries.csv", index=False)
    inventory.to_csv(f"{output_dir}/inventory.csv", index=False)
    marketing.to_csv(f"{output_dir}/marketing.csv", index=False)

    print("Data generated successfully!")
    return products, orders, deliveries, inventory, marketing


if __name__ == "__main__":
    import os
    generate_all(os.path.dirname(os.path.abspath(__file__)))
