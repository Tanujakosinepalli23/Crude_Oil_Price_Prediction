import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Statsmodels
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
DATA_PATH = Path("Crude oil.csv")
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)
df.index = pd.DatetimeIndex(df["Date"])
target_col = "Close/Last"

series = df[target_col].astype(float).ffill().bfill()

# Sidebar Controls
st.sidebar.header("⚙️ Model Settings")

lags = st.sidebar.slider("Number of Lags", 5, 30, 14)
split_ratio = st.sidebar.slider("Train/Test Split (%)", 60, 95, 80)
horizon = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 30)

# Train/Test split
split_idx = int(len(series) * (split_ratio/100))
train, test = series.iloc[:split_idx], series.iloc[split_idx:]

# Evaluation function
def evaluate(y_true, y_pred):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-8, y_true))) * 100
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}

st.title("⛽ Crude Oil Price Forecasting App")
st.markdown("This app forecasts crude oil prices using AutoRegressive models.")

# -------------------------
# Fit AutoReg on train
# -------------------------
model = AutoReg(train, lags=lags, old_names=False).fit()

# -------------------------
# Manual one-step-ahead walk-forward predictions on test
# -------------------------
params = model.params.copy()
intercept = 0.0
if 'const' in params.index:
    intercept = float(params['const'])
    ar_coefs = params.drop('const').values
elif 'Intercept' in params.index:
    intercept = float(params['Intercept'])
    ar_coefs = params.drop('Intercept').values
else:
    ar_coefs = params.values

ar_coefs = np.asarray(ar_coefs, dtype=float)
p = len(ar_coefs)

test_preds = []
for t in range(len(test)):
    if t == 0:
        hist = train.values
    else:
        hist = np.concatenate([train.values, test.values[:t]])
    k = min(p, len(hist))
    yhat = intercept
    for i in range(k):
        yhat += ar_coefs[i] * hist[-(i+1)]
    test_preds.append(yhat)

test_preds = pd.Series(test_preds, index=test.index)
metrics = evaluate(test, test_preds)

# Metrics Display
st.subheader("📊 Model Performance on Test Data")
st.write(pd.DataFrame([metrics]))

# Actual vs Predicted Plot (fixed)
st.subheader("📈 Actual vs Predicted (Test set)")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(train.index, train, label="Train", linewidth=1)
ax.plot(test.index, test, label="Test (Actual)", color="black", linewidth=1)
ax.plot(test_preds.index, test_preds.values, label="Predicted (one-step)", color="red", linewidth=1)
ax.set_title(f"AutoReg(lags={lags}) - Test Predictions (one-step-ahead)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# -------------------------
# Future Forecast (recursive)
# -------------------------
model_full = AutoReg(series, lags=lags, old_names=False).fit()
params_full = model_full.params.copy()
intercept_full = float(params_full.get('const', params_full.get('Intercept', 0.0)))
ar_coefs_full = params_full.drop(labels=[k for k in params_full.index if k.lower() in ('const','intercept')]).values
ar_coefs_full = np.asarray(ar_coefs_full, dtype=float)
p_full = len(ar_coefs_full)

history = list(series.values)
future_preds = []
for _ in range(horizon):
    k = min(p_full, len(history))
    yhat = intercept_full
    for i in range(k):
        yhat += ar_coefs_full[i] * history[-(i+1)]
    future_preds.append(yhat)
    history.append(yhat)

last_date = series.index[-1]
freq = pd.infer_freq(series.index)
if freq is None:
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
else:
    future_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)

forecast_series = pd.Series(future_preds, index=future_index)

st.subheader(f"🔮 Forecast for next {horizon} days")
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(series.index[-200:], series.values[-200:], label="Recent Actual")
ax2.plot(forecast_series.index, forecast_series.values, label="Forecast", color="orange")
ax2.set_title(f"AutoReg(lags={lags}) — {horizon}-Day Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# Forecast Table
st.subheader("📅 Forecasted Values")
st.dataframe(forecast_series.reset_index().rename(columns={"index":"Date",0:"Forecasted Price"}))

# Download Option
csv = forecast_series.to_csv().encode("utf-8")
st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")
