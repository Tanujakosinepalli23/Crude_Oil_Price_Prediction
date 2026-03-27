import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Crude Oil Price Forecasting Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -------------------------
# TITLE
# -------------------------
st.title("🌍 Crude Oil Price Forecasting Dashboard")

st.write(
"""
This app forecasts crude oil prices using a time series model.  
You can use the default dataset or upload your own CSV file.
"""
)

# -------------------------
# DATA SOURCE
# -------------------------
st.subheader("📂 Data Source")

option = st.radio(
    "Choose data source:",
    ["Use Default Dataset", "Upload Your Own CSV"]
)

df = None

# Default dataset
if option == "Use Default Dataset":
    DATA_PATH = Path("Crude oil.csv")

    if not DATA_PATH.exists():
        st.error("❌ Default dataset not found")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    st.success("✅ Default dataset loaded")

# Upload dataset
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded dataset loaded")
    else:
        st.info("Please upload a CSV file to continue")
        st.stop()

# -------------------------
# PREPROCESS
# -------------------------
try:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    series = df["Close/Last"].astype(float).ffill().bfill()
except:
    st.error("❌ CSV must contain 'Date' and 'Close/Last' columns")
    st.stop()

# -------------------------
# DATA PREVIEW
# -------------------------
st.subheader("📊 Historical Data Sample")
st.dataframe(df.head(10), use_container_width=True)

# -------------------------
# HISTORICAL CHART
# -------------------------
st.subheader("📈 Historical Closing Prices")
st.line_chart(series)

# -------------------------
# FORECAST SECTION
# -------------------------
st.subheader("🔮 Forecast Future Prices")

col1, col2 = st.columns([2,1])

with col1:
    horizon = st.slider("Select number of days to forecast", 5, 60, 30)

with col2:
    generate = st.button("🚀 Generate Forecast")

# -------------------------
# MAIN LOGIC
# -------------------------
if generate:

    # -------------------------
    # DIRECTIONAL ACCURACY (FIXED)
    # -------------------------
    train = series[:-horizon]
    test = series[-horizon:]

    history = list(train)
    preds = []

    for t in range(len(test)):
        yhat = np.mean(history[-5:])
        preds.append(yhat)
        history.append(test.iloc[t])

    preds = np.array(preds)
    actual = np.array(test)

    actual_diff = np.sign(np.diff(actual))
    pred_diff = np.sign(np.diff(preds))

    min_len = min(len(actual_diff), len(pred_diff))

    if min_len > 0:
        directional_accuracy = np.mean(
            actual_diff[:min_len] == pred_diff[:min_len]
        ) * 100
    else:
        directional_accuracy = 0

    st.metric("🎯 Directional Accuracy (%)", f"{directional_accuracy:.2f}%")

    # -------------------------
    # FUTURE FORECAST
    # -------------------------
    history = list(series)
    future_preds = []

    for _ in range(horizon):
        yhat = np.mean(history[-5:])
        future_preds.append(yhat)
        history.append(yhat)

    future_index = pd.date_range(
        series.index[-1] + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast_series = pd.Series(future_preds, index=future_index)

    st.success("✅ Forecast generated successfully!")

    # -------------------------
    # CHART
    # -------------------------
    forecast_df = pd.DataFrame({
        "Recent Data": series[-200:],
        "Forecast": forecast_series
    })

    st.line_chart(forecast_df)

    # -------------------------
    # TABLE (INDEX STARTS FROM 1)
    # -------------------------
    st.subheader("📅 Forecast Data")

    forecast_df_display = forecast_series.reset_index()
    forecast_df_display.columns = ["Date", "Forecast"]

    forecast_df_display.index = forecast_df_display.index + 1

    st.dataframe(forecast_df_display, use_container_width=True)

    # -------------------------
    # DOWNLOAD
    # -------------------------
    csv = forecast_series.to_csv().encode("utf-8")
    st.download_button("⬇️ Download Forecast", csv, "forecast.csv")
