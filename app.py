import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import validate_and_parse_csv, engineer_features, prepare_training_data
from training import train_model, forecast
from visualization import (
    plot_forecast,
    plot_loss_curves,
    plot_seasonal_pattern,
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

# ── Seasonal Pattern ─────────────────────────────────────────────────────────
st.header("Seasonal Pattern")
seasonal_fig = plot_seasonal_pattern(feat_df["date"], feat_df["sales"], currency)
st.plotly_chart(seasonal_fig, use_container_width=True)
