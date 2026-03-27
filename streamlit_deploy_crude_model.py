import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Load dataset
# -------------------------
DATA_PATH = Path("Crude oil.csv")

if not DATA_PATH.exists():
    st.error("❌ 'Crude oil.csv' not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

series = df["Close/Last"].astype(float).ffill().bfill()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Settings")

lags = st.sidebar.slider("Lags", 1, 20, 5)
split_ratio = st.sidebar.slider("Train %", 60, 90, 80)
horizon = st.sidebar.slider("Forecast Days", 5, 60, 30)

# -------------------------
# Split
# -------------------------
split_idx = int(len(series) * split_ratio / 100)
train = series[:split_idx]
test = series[split_idx:]

# -------------------------
# Manual AR model (no statsmodels)
# -------------------------
def predict_ar(train_data, test_data, lags):
    history = list(train_data)
    predictions = []

    for t in range(len(test_data)):
        if len(history) < lags:
            yhat = np.mean(history)
        else:
            yhat = np.mean(history[-lags:])

        predictions.append(yhat)
        history.append(test_data.iloc[t])

    return pd.Series(predictions, index=test_data.index)

test_preds = predict_ar(train, test, lags)

# -------------------------
# Metrics
# -------------------------
def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return {"MAE": mae, "RMSE": rmse}

metrics = evaluate(test, test_preds)

# -------------------------
# UI
# -------------------------
st.title("⛽ Crude Oil Forecast (No statsmodels)")

st.subheader("📊 Metrics")
st.write(metrics)

# -------------------------
# Chart
# -------------------------
st.subheader("📈 Actual vs Predicted")

chart_df = pd.DataFrame({
    "Train": train,
    "Actual": test,
    "Predicted": test_preds
})

st.line_chart(chart_df)

# -------------------------
# Future Forecast
# -------------------------
history = list(series)
future_preds = []

for _ in range(horizon):
    if len(history) < lags:
        yhat = np.mean(history)
    else:
        yhat = np.mean(history[-lags:])

    future_preds.append(yhat)
    history.append(yhat)

future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
forecast_series = pd.Series(future_preds, index=future_index)

# -------------------------
# Forecast chart
# -------------------------
st.subheader("🔮 Future Forecast")

forecast_df = pd.DataFrame({
    "Recent": series[-200:],
    "Forecast": forecast_series
})

st.line_chart(forecast_df)

# -------------------------
# Table
# -------------------------
st.subheader("📅 Forecast Data")

st.dataframe(forecast_series.reset_index().rename(columns={"index":"Date",0:"Forecast"}))
