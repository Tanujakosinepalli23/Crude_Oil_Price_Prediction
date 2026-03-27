import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Crude Oil Forecast",
    page_icon="⛽",
    layout="wide"
)

# -------------------------
# Header
# -------------------------
st.title("⛽ Crude Oil Price Forecasting")
st.caption("Time Series Forecasting Dashboard")

# -------------------------
# Load Data
# -------------------------
DATA_PATH = Path("Crude oil.csv")

if not DATA_PATH.exists():
    st.error("Upload 'Crude oil.csv' to continue")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

series = df["Close/Last"].astype(float).ffill().bfill()

# -------------------------
# Sidebar Controls
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    lags = st.slider("Lag Window", 1, 20, 5)
    split_ratio = st.slider("Train Size (%)", 60, 90, 80)
    horizon = st.slider("Forecast Days", 5, 60, 30)

# -------------------------
# Split Data
# -------------------------
split_idx = int(len(series) * split_ratio / 100)
train = series[:split_idx]
test = series[split_idx:]

# -------------------------
# Model
# -------------------------
def predict_ar(train_data, test_data, lags):
    history = list(train_data)
    preds = []

    for t in range(len(test_data)):
        yhat = np.mean(history[-lags:]) if len(history) >= lags else np.mean(history)
        preds.append(yhat)
        history.append(test_data.iloc[t])

    return pd.Series(preds, index=test_data.index)

test_preds = predict_ar(train, test, lags)

# -------------------------
# Metrics
# -------------------------
mae = np.mean(np.abs(test - test_preds))
rmse = np.sqrt(np.mean((test - test_preds) ** 2))

# -------------------------
# TABS (Professional Layout)
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Model", "🔮 Forecast"])

# -------------------------
# TAB 1 - Overview
# -------------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("Total Records", len(series))

    st.markdown("### Recent Price Trend")
    st.line_chart(series.tail(200), height=350)

# -------------------------
# TAB 2 - Model Performance
# -------------------------
with tab2:
    st.markdown("### Actual vs Predicted")

    chart_df = pd.DataFrame({
        "Actual": test,
        "Predicted": test_preds
    })

    st.line_chart(chart_df, height=400)

    st.dataframe(chart_df.tail(50), use_container_width=True)

# -------------------------
# TAB 3 - Forecast
# -------------------------
with tab3:
    history = list(series)
    future_preds = []

    for _ in range(horizon):
        yhat = np.mean(history[-lags:]) if len(history) >= lags else np.mean(history)
        future_preds.append(yhat)
        history.append(yhat)

    future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    forecast_series = pd.Series(future_preds, index=future_index)

    st.markdown("### Future Forecast")

    forecast_df = pd.DataFrame({
        "Recent": series[-200:],
        "Forecast": forecast_series
    })

    st.line_chart(forecast_df, height=400)

    st.dataframe(
        forecast_series.reset_index().rename(columns={"index":"Date",0:"Forecast"}),
        use_container_width=True
    )

    csv = forecast_series.to_csv().encode("utf-8")
    st.download_button("Download Forecast", csv, "forecast.csv")
