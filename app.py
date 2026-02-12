import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import validate_and_parse_csv, engineer_features, prepare_training_data
from training import train_model, forecast
from visualization import (
    plot_forecast,
    plot_loss_curves,
    plot_seasonal_pattern,
    plot_yearly_comparison,
    compute_seasonal_insights,
    format_currency,
    CURRENCY_SYMBOLS,
)

st.set_page_config(
    page_title="Sales Forecast",
    page_icon="📈",
    layout="wide",
)

st.title("Sales Forecasting Tool")
st.caption("Upload monthly sales data to generate forecasts using LSTM neural network")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV with 2 columns: date (e.g. 2024-01) and sales amount",
    )

    currency = st.selectbox("Currency", options=["USD", "TRY"], index=0)

    horizon = st.selectbox(
        "Forecast Horizon",
        options=[1, 3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
    )

    train_button = st.button("Train & Forecast", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Model Details**")
    st.markdown(
        "- **Architecture:** 2-layer LSTM\n"
        "- **Loss:** Huber (SmoothL1)\n"
        "- **Optimizer:** AdamW\n"
        "- **Scheduler:** ReduceLROnPlateau"
    )

# ── Main Area ────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.markdown(
        "**Expected CSV format:**\n\n"
        "| Date | Sales |\n"
        "|------|-------|\n"
        "| 2022-01 | 45000 |\n"
        "| 2022-02 | 52000 |\n"
        "| ... | ... |\n\n"
        "- Minimum **24 months** of data required\n"
        "- Column names are auto-detected\n"
        "- Date formats like `2024-01`, `Jan 2024`, `01/2024` are supported"
    )
    st.stop()

# Read and validate the CSV
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

try:
    dates, sales = validate_and_parse_csv(raw_df)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ── Data Preview ─────────────────────────────────────────────────────────────
st.header("Data Preview")
col1, col2, col3, col4 = st.columns(4)
symbol = CURRENCY_SYMBOLS.get(currency, "$")
col1.metric("Total Months", len(dates))
col2.metric("Min Sales", format_currency(sales.min(), currency))
col3.metric("Max Sales", format_currency(sales.max(), currency))
col4.metric("Average", format_currency(sales.mean(), currency))

preview_df = pd.DataFrame({
    "Date": dates.dt.strftime("%Y-%m"),
    f"Sales ({currency})": sales.apply(lambda x: format_currency(x, currency)),
})
with st.expander("Show raw data"):
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

# ── Feature Engineering ──────────────────────────────────────────────────────
st.header("Feature Engineering")
feat_df = engineer_features(dates, sales)

with st.expander("Show engineered features"):
    display_feat = feat_df.copy()
    display_feat["date"] = display_feat["date"].dt.strftime("%Y-%m")
    st.dataframe(
        display_feat.round(4),
        use_container_width=True,
        hide_index=True,
    )

st.markdown(f"**{len(feat_df)}** usable data points after feature engineering")

# ── Training & Forecasting ───────────────────────────────────────────────────
if not train_button:
    st.info('Click **"Train & Forecast"** in the sidebar to start.')
    st.stop()

if len(feat_df) < 6:
    st.error("Not enough data points after feature engineering. Provide more monthly data.")
    st.stop()

st.header("Training")

# Prepare data
data = prepare_training_data(feat_df)

# Training with progress
progress_bar = st.progress(0, text="Training LSTM model...")
status_text = st.empty()

def on_progress(epoch, max_epochs, train_loss, val_loss):
    pct = min((epoch + 1) / max_epochs, 1.0)
    progress_bar.progress(pct, text=f"Epoch {epoch + 1}/{max_epochs}")
    status_text.text(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

result = train_model(data, progress_callback=on_progress)

progress_bar.progress(1.0, text="Training complete!")
status_text.text(
    f"Best epoch: {result['best_epoch'] + 1}/{result['total_epochs']} | "
    f"Best val loss: {result['best_val_loss']:.6f}"
)

# ── Loss Curves ──────────────────────────────────────────────────────────────
st.header("Training Loss")
loss_fig = plot_loss_curves(result["train_losses"], result["val_losses"])
st.plotly_chart(loss_fig, use_container_width=True)

# ── Forecast ─────────────────────────────────────────────────────────────────
st.header(f"Forecast ({horizon} month{'s' if horizon > 1 else ''})")

predictions = forecast(result["model"], feat_df, data, horizon)

# Build forecast dates
last_date = feat_df["date"].iloc[-1]
forecast_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(horizon)]

# Forecast plot
forecast_fig = plot_forecast(
    feat_df["date"], feat_df["sales"], forecast_dates, predictions, currency
)
st.plotly_chart(forecast_fig, use_container_width=True)

# Forecast table
forecast_table = pd.DataFrame({
    "Date": [d.strftime("%Y-%m") for d in forecast_dates],
    f"Forecasted Sales ({currency})": [format_currency(v, currency) for v in predictions],
    "Raw Value": [round(v, 2) for v in predictions],
})
st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# Download button
csv_download = pd.DataFrame({
    "Date": [d.strftime("%Y-%m") for d in forecast_dates],
    f"Forecast_{currency}": [round(v, 2) for v in predictions],
})
st.download_button(
    label="Download Forecast CSV",
    data=csv_download.to_csv(index=False),
    file_name="sales_forecast.csv",
    mime="text/csv",
)

# ── Seasonal Analysis ────────────────────────────────────────────────────────
st.header("Seasonal Analysis")

insights = compute_seasonal_insights(feat_df["date"], feat_df["sales"])

# Key metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak Month", insights["peak_month"], format_currency(insights["peak_value"], currency))
m2.metric("Low Month", insights["low_month"], format_currency(insights["low_value"], currency))
m3.metric("Seasonal Strength", f"{insights['seasonal_strength']:.1f}%",
          help="Coefficient of variation of monthly averages. Higher = more seasonal.")
m4.metric("Peak / Low Ratio", f"{insights['peak_to_low_ratio']:.2f}x",
          help="How many times higher the best month is vs the worst month.")

# Monthly pattern chart
st.subheader("Monthly Pattern")
st.caption("Average sales per month with min–max range band. Bars colored by quarter.")
seasonal_fig = plot_seasonal_pattern(feat_df["date"], feat_df["sales"], currency)
st.plotly_chart(seasonal_fig, use_container_width=True)

# Year-over-year comparison
if insights["num_years"] >= 2:
    st.subheader("Year-over-Year Comparison")
    st.caption("Each line represents one year's monthly sales, making trends and shifts easy to spot.")
    yoy_fig = plot_yearly_comparison(feat_df["date"], feat_df["sales"], currency)
    st.plotly_chart(yoy_fig, use_container_width=True)

    # YoY growth cards
    if insights["yoy_growth"]:
        st.subheader("Annual Growth")
        growth_cols = st.columns(len(insights["yoy_growth"]))
        for col, g in zip(growth_cols, insights["yoy_growth"]):
            delta_color = "normal" if g["growth_pct"] >= 0 else "inverse"
            col.metric(
                g["period"],
                format_currency(g["curr_total"], currency),
                f"{g['growth_pct']:+.1f}%",
                delta_color=delta_color,
            )

# Quarter breakdown
st.subheader("Quarter Breakdown")
q_cols = st.columns(len(insights["quarter_data"]))
for col, qd in zip(q_cols, insights["quarter_data"]):
    sign = "+" if qd["vs_overall"] >= 0 else ""
    col.metric(
        qd["quarter"],
        format_currency(qd["avg_sales"], currency),
        f"{sign}{qd['vs_overall']:.1f}% vs avg",
        delta_color="normal" if qd["vs_overall"] >= 0 else "inverse",
    )

# Textual insights
st.subheader("Insights")
insight_lines = []
insight_lines.append(
    f"**{insights['peak_month']}** is your strongest month with average sales of "
    f"**{format_currency(insights['peak_value'], currency)}**, while "
    f"**{insights['low_month']}** is the weakest at "
    f"**{format_currency(insights['low_value'], currency)}**."
)

if insights["seasonal_strength"] > 15:
    insight_lines.append(
        f"Your business shows **strong seasonality** ({insights['seasonal_strength']:.1f}% variation). "
        "Plan inventory and staffing around peak and low months."
    )
elif insights["seasonal_strength"] > 5:
    insight_lines.append(
        f"Your business shows **moderate seasonality** ({insights['seasonal_strength']:.1f}% variation). "
        "Some months consistently outperform others."
    )
else:
    insight_lines.append(
        f"Your sales are **relatively stable** across months ({insights['seasonal_strength']:.1f}% variation). "
        "No strong seasonal swings detected."
    )

insight_lines.append(
    f"Best performing quarter: **{insights['best_quarter']}** — "
    f"Weakest quarter: **{insights['worst_quarter']}**."
)

if insights["yoy_growth"]:
    latest = insights["yoy_growth"][-1]
    direction = "grew" if latest["growth_pct"] >= 0 else "declined"
    insight_lines.append(
        f"Annual sales **{direction} {abs(latest['growth_pct']):.1f}%** "
        f"in the most recent period ({latest['period']})."
    )

insight_lines.append(
    f"Month-to-month volatility is **{insights['mom_volatility']:.1f}%**, "
    + ("which is high — expect significant swings between consecutive months."
       if insights["mom_volatility"] > 15
       else "indicating fairly smooth transitions between months."
       if insights["mom_volatility"] < 8
       else "which is moderate.")
)

for line in insight_lines:
    st.markdown(f"- {line}")
