import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Auto-generate data if CSV files don't exist (needed for Streamlit Cloud)
if not os.path.exists(os.path.join(DATA_DIR, "orders.csv")):
    from data.generator import generate_all
    generate_all(DATA_DIR)

from analytics.inventory import (
    load_data, inventory_turnover, stockout_analysis,
    slow_moving_analysis, category_performance
)
from analytics.fulfillment import (
    fulfillment_kpis, fulfillment_rate, shipping_cost_analysis
)
from analytics.forecasting import (
    prepare_demand_series, arima_forecast, demand_volatility
)
from analytics.correlation import (
    marketing_sales_correlation, channel_roi, category_correlation
)

st.set_page_config(page_title="Supply Chain Analytics Dashboard", layout="wide")

@st.cache_data
def load_all_data():
    return load_data(DATA_DIR)

products, orders, inventory, deliveries, marketing = load_all_data()

st.title("E-Commerce Operations & Supply Chain Analytics")
st.markdown("Interactive dashboard for inventory optimization, fulfillment tracking, demand forecasting, and marketing ROI analysis.")

tab_overview, tab_inventory, tab_fulfillment, tab_forecast, tab_marketing, tab_insights = st.tabs([
    "Overview", "Inventory Analytics", "Fulfillment & Delivery",
    "Demand Forecasting", "Marketing ROI", "Insights & Recommendations"
])

with tab_overview:
    st.header("Executive Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    total_revenue = orders["total"].sum()
    total_orders = orders["order_id"].nunique()
    total_products = products["product_id"].nunique()
    avg_order_value = total_revenue / total_orders
    fulfill_rate = fulfillment_rate(orders, deliveries)

    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Products", total_products)
    col4.metric("Avg Order Value", f"₹{avg_order_value:,.0f}")
    col5.metric("Fulfillment Rate", f"{fulfill_rate}%")

    col1, col2 = st.columns(2)

    with col1:
        daily_revenue = orders.groupby("order_date")["total"].sum().reset_index()
        fig = px.line(daily_revenue, x="order_date", y="total",
                      title="Daily Revenue Trend",
                      labels={"total": "Revenue (₹)", "order_date": ""})
        fig.update_traces(line_color="#2E86AB")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_revenue = orders.groupby("category")["total"].sum().reset_index()
        fig = px.pie(cat_revenue, values="total", names="category",
                     title="Revenue by Category", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        monthly_orders = orders.copy()
        monthly_orders["month"] = monthly_orders["order_date"].dt.to_period("M").astype(str)
        monthly = monthly_orders.groupby("month")["order_id"].nunique().reset_index()
        fig = px.bar(monthly, x="month", y="order_id",
                     title="Monthly Orders",
                     labels={"order_id": "Orders", "month": ""},
                     color_discrete_sequence=["#A23B72"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        inv_summary = inventory.groupby("date")["stockout"].sum().reset_index()
        fig = px.area(inv_summary, x="date", y="stockout",
                      title="Daily Stockout Incidents",
                      labels={"stockout": "Products Out of Stock", "date": ""},
                      color_discrete_sequence=["#F18F01"])
        st.plotly_chart(fig, use_container_width=True)

with tab_inventory:
    st.header("Inventory Analytics")

    stockout = stockout_analysis(inventory)
    turnover = inventory_turnover(inventory, products)
    slow = slow_moving_analysis(inventory, products)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Stockout Rate", f"{stockout['stockout_rate'].mean():.1f}%")
    col2.metric("Avg Inventory Turnover",
                f"{turnover['turnover_ratio'].mean():.2f}x")
    col3.metric("Avg Days of Inventory", f"{slow['days_of_inventory'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        top_stockout = stockout.sort_values("stockout_rate", ascending=False).head(10)
        fig = px.bar(top_stockout, x="product_name", y="stockout_rate",
                     title="Top 10 Products by Stockout Rate",
                     labels={"stockout_rate": "Stockout Rate (%)", "product_name": ""},
                     color="stockout_rate", color_continuous_scale="Reds")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(stockout, x="avg_stock", y="stockout_rate",
                         size="total_days", hover_name="product_name",
                         title="Stockout Rate vs Average Stock Level",
                         labels={"avg_stock": "Avg Stock Level", "stockout_rate": "Stockout Rate (%)"},
                         color="stockout_rate", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inventory Turnover by Category")
    turn_cat = turnover.groupby(["category", "month"])["turnover_ratio"].mean().reset_index()
    fig = px.line(turn_cat, x="month", y="turnover_ratio", color="category",
                  title="Monthly Inventory Turnover by Category",
                  labels={"turnover_ratio": "Turnover Ratio", "month": ""})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Slow-Moving Inventory")
    slow_sorted = slow.sort_values("days_of_inventory", ascending=False).head(15)
    fig = px.bar(slow_sorted, x="product_name", y="days_of_inventory",
                 title="Products with Highest Days of Inventory (Slow Movers)",
                 labels={"days_of_inventory": "Days of Inventory", "product_name": ""},
                 color="holding_cost_est", color_continuous_scale="Blues",
                 hover_data=["category", "avg_stock", "avg_daily_sold", "holding_cost_est"])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category Performance Over Time")
    cat_perf = category_performance(inventory, products)
    fig = px.line(cat_perf, x="month", y="revenue", color="category",
                  title="Monthly Revenue by Category",
                  labels={"revenue": "Revenue (₹)", "month": ""})
    st.plotly_chart(fig, use_container_width=True)

with tab_fulfillment:
    st.header("Fulfillment & Delivery Analytics")

    merged, carrier_stats, region_stats, monthly_fulfill = fulfillment_kpis(orders, deliveries)
    shipping = shipping_cost_analysis(orders, deliveries)

    avg_delivery = merged["delivery_time_days"].mean()
    on_time = (merged["delivery_time_days"] <= 5).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Delivery Time", f"{avg_delivery:.1f} days")
    col2.metric("On-Time Rate (≤5 days)", f"{on_time:.1f}%")
    col3.metric("Total Delivered Orders", f"{len(merged):,}")
    col4.metric("Avg Shipping Cost", f"₹{shipping['shipping_cost'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(carrier_stats, x="carrier", y="avg_delivery_days",
                     title="Average Delivery Time by Carrier",
                     labels={"avg_delivery_days": "Avg Days", "carrier": ""},
                     color="avg_delivery_days", color_continuous_scale="RdYlGn_r",
                     hover_data=["total_orders", "median_delivery_days"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(region_stats, x="region", y="on_time_rate",
                     title="On-Time Delivery Rate by Region",
                     labels={"on_time_rate": "On-Time Rate (%)", "region": ""},
                     color="on_time_rate", color_continuous_scale="RdYlGn",
                     hover_data=["total_orders", "avg_delivery_days"])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(monthly_fulfill, x="month", y="avg_delivery_days",
                      title="Monthly Average Delivery Time Trend",
                      labels={"avg_delivery_days": "Avg Days", "month": ""})
        fig.update_traces(line_color="#A23B72")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        shipping["shipping_ratio_bucket"] = pd.cut(
            shipping["shipping_ratio"], bins=[0, 5, 10, 20, 50, 100],
            labels=["0-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
        )
        ship_dist = shipping["shipping_ratio_bucket"].value_counts().sort_index().reset_index()
        fig = px.bar(ship_dist, x="shipping_ratio_bucket", y="count",
                     title="Distribution of Shipping Cost as % of Order Value",
                     labels={"count": "Orders", "shipping_ratio_bucket": "Shipping Ratio"},
                     color_discrete_sequence=["#2E86AB"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delivery Time Distribution")
    fig = px.histogram(merged, x="delivery_time_days", nbins=20,
                       title="Delivery Time Distribution (Days)",
                       labels={"delivery_time_days": "Days"},
                       color_discrete_sequence=["#2E86AB"])
    fig.add_vline(x=5, line_dash="dash", line_color="red",
                  annotation_text="Target (5 days)")
    st.plotly_chart(fig, use_container_width=True)

with tab_forecast:
    st.header("Demand Forecasting")

    col1, col2 = st.columns([1, 3])
    with col1:
        product_names = products["product_name"].tolist()
        selected_product = st.selectbox("Select Product", product_names, index=0)
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)

    pid = products[products["product_name"] == selected_product]["product_id"].values[0]
    series = prepare_demand_series(orders, pid)
    forecast, metrics = arima_forecast(series, forecast_days)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Demand", f"{series.mean():.1f}")
    col2.metric("Demand Std Dev", f"{series.std():.1f}")
    if metrics and metrics.get("mae"):
        col3.metric("Forecast MAE (30d)", f"₹{metrics['mae']}")
    elif metrics and metrics.get("fallback"):
        col3.metric("Forecast", "Using historical avg")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index[-90:], y=series.values[-90:],
        mode="lines", name="Historical Demand",
        line=dict(color="#2E86AB", width=2)
    ))
    if forecast is not None:
        future_dates = pd.date_range(series.index[-1] + timedelta(days=1), periods=forecast_days)
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast,
            mode="lines+markers", name="Forecast",
            line=dict(color="#F18F01", width=2, dash="dash"),
            marker=dict(size=5)
        ))
    fig.update_layout(title=f"Demand Forecast: {selected_product}",
                      xaxis_title="", yaxis_title="Units Sold",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Demand Volatility Analysis")
    vol = demand_volatility(orders)
    vol = vol.merge(products[["product_id", "product_name", "category"]], on="product_id")
    vol = vol.sort_values("cv", ascending=False)

    fig = px.bar(vol.head(15), x="product_name", y="cv",
                 title="Top 15 Most Volatile Products (Coefficient of Variation)",
                 labels={"cv": "CV (%)", "product_name": ""},
                 color="cv", color_continuous_scale="OrRd",
                 hover_data=["mean_daily", "std_daily", "category"])
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    vol_cat = vol.groupby("category")["cv"].mean().reset_index()
    fig = px.bar(vol_cat, x="category", y="cv",
                 title="Average Demand Volatility by Category",
                 labels={"cv": "Avg CV (%)", "category": ""},
                 color="cv", color_continuous_scale="OrRd")
    st.plotly_chart(fig, use_container_width=True)

with tab_marketing:
    st.header("Marketing ROI & Correlation Analysis")

    corr_matrix, lag_corr, merged = marketing_sales_correlation(marketing, orders)
    chnl_roi = channel_roi(marketing, orders)

    col1, col2, col3, col4 = st.columns(4)
    total_spend = marketing["spend"].sum()
    total_rev = orders["total"].sum()
    col1.metric("Total Marketing Spend", f"₹{total_spend:,.0f}")
    col2.metric("Overall ROAS", f"{total_rev / total_spend:.2f}x")
    col3.metric("Best Channel", chnl_roi.sort_values("ROAS", ascending=False).iloc[0]["channel"])
    col4.metric("Conversion Rate (Avg)", f"{chnl_roi['conversion_rate'].mean():.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(chnl_roi, x="channel", y="ROAS",
                     title="Return on Ad Spend (ROAS) by Channel",
                     labels={"ROAS": "Revenue / Spend", "channel": ""},
                     color="ROAS", color_continuous_scale="Viridis",
                     hover_data=["total_spend", "total_revenue", "conversion_rate"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(chnl_roi, x="channel", y="CPC",
                     title="Cost Per Click by Channel",
                     labels={"CPC": "Cost Per Click (₹)", "channel": ""},
                     color="CPC", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Marketing Spend vs Revenue Correlation")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Matrix: Marketing vs Sales",
                        aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(lag_corr, x="lag", y="correlation",
                     title="Spend-to-Revenue Correlation by Lag",
                     labels={"correlation": "Pearson Correlation", "lag": ""},
                     color="correlation", color_continuous_scale="RdYlBu",
                     range_color=[-0.5, 0.5])
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Spend vs Revenue")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["spend"],
                             name="Marketing Spend", line=dict(color="#F18F01")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["revenue"],
                             name="Revenue", line=dict(color="#2E86AB")),
                  secondary_y=True)
    fig.update_layout(title="Marketing Spend vs Daily Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category Sales Correlation")
    cat_corr, cat_pivot = category_correlation(inventory, orders, products)
    if cat_corr.shape[0] > 1:
        fig = px.imshow(cat_corr, text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Cross-Category Sales Correlation",
                        aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient category data for cross-correlation.")

with tab_insights:
    st.header("Key Insights & Recommendations")

    stockout = stockout_analysis(inventory)
    slow = slow_moving_analysis(inventory, products)
    merged, _, _, _ = fulfillment_kpis(orders, deliveries)
    chnl_roi = channel_roi(marketing, orders)
    vol = demand_volatility(orders)

    high_stockout = stockout[stockout["stockout_rate"] > 5]
    slow_movers = slow[slow["days_of_inventory"] > 60]
    worst_carrier = merged.groupby("carrier")["delivery_time_days"].mean().idxmax()
    worst_region = merged.groupby("region")["delivery_time_days"].mean().idxmax()
    best_channel = chnl_roi.sort_values("ROAS", ascending=False).iloc[0]
    worst_channel = chnl_roi.sort_values("ROAS").iloc[0]

    st.subheader("📦 Inventory")
    col1, col2 = st.columns(2)
    with col1:
        st.warning(f"**Stockout Risk:** {len(high_stockout)} products have stockout rates > 5%")
        if len(high_stockout) > 0:
            for _, row in high_stockout.head(5).iterrows():
                st.markdown(f"- **{row['product_name']}**: {row['stockout_rate']}% stockout rate")
    with col2:
        st.warning(f"**Slow-Moving Inventory:** {len(slow_movers)} products have > 60 days of inventory")
        if len(slow_movers) > 0:
            for _, row in slow_movers.head(5).iterrows():
                st.markdown(f"- **{row['product_name']}**: {row['days_of_inventory']} days (₹{row['holding_cost_est']:.0f}/day holding cost)")

    st.subheader("🚚 Fulfillment")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Worst Carrier:** {worst_carrier} has highest avg delivery time. Consider renegotiating SLA or redistributing volume.")
    with col2:
        st.info(f"**Worst Region:** {worst_region} has highest avg delivery time. Evaluate regional warehouse or alternative carrier.")

    st.subheader("📈 Marketing")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Best Channel:** {best_channel['channel']} (ROAS: {best_channel['ROAS']}x)")
    with col2:
        st.error(f"**Underperforming Channel:** {worst_channel['channel']} (ROAS: {worst_channel['ROAS']}x) — consider reallocating budget")

    st.subheader("⚡ Demand Volatility")
    high_vol = vol[vol["cv"] > 80].merge(products[["product_id", "product_name"]], on="product_id")
    if len(high_vol) > 0:
        st.warning(f"{len(high_vol)} products have very high demand volatility (CV > 80%). Consider safety stock optimization.")
        for _, row in high_vol.head(5).iterrows():
            st.markdown(f"- **{row['product_name']}**: CV {row['cv']}% (mean: {row['mean_daily']:.1f}/day)")
    else:
        st.success("Demand volatility is within manageable range across all products.")

    st.subheader("🎯 Recommended Actions")
    recs = [
        f"1. **Increase safety stock** for {len(high_stockout)} high-stockout-risk products to reduce lost revenue.",
        f"2. **Run clearance or bundle promotions** for {len(slow_movers)} slow-moving items to free up working capital.",
        f"3. **Audit {worst_carrier} SLA** and explore alternatives for {worst_region} region deliveries.",
        f"4. **Shift budget toward {best_channel['channel']}** (ROAS: {best_channel['ROAS']}x) and reduce {worst_channel['channel']} spend.",
        f"5. **Implement demand-based reorder points** for high-volatility products to balance stockouts vs holding costs.",
    ]
    for rec in recs:
        st.markdown(rec)

st.markdown("---")
st.caption("Supply Chain Analytics Dashboard | Built with Python, Streamlit, Pandas, Plotly, statsmodels")
