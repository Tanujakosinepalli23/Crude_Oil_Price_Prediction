import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Statsmodels
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Load dataset
# -------------------------
DATA_PATH = Path("Crude oil.csv")

if not DATA_PATH.exists():
    st.error("❌ 'Crude oil.csv' not found. Please upload it.")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)
df.index = pd.DatetimeIndex(df["Date"])

target_col = "Close/Last"
series = df[target_col].astype(float).ffill().bfill()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Model Settings")

lags = st.sidebar.slider("Number of Lags", 5, 30, 14)
split_ratio = st.sidebar.slider("Train/Test Split (%)", 60, 95, 80)
horizon = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 30)

# -------------------------
# Split data
# -------------------------
split_idx = int(len(series) * (split_ratio/100))
train, test = series.iloc[:split_idx], series.iloc[split_idx:]

# -------------------------
# Evaluation function
# -------------------------
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

# -------------------------
# Title
# -------------------------
st.title("⛽ Crude Oil Price Forecasting")
st.write("AutoReg Model (No matplotlib version)")

# -------------------------
# Train model
# -------------------------
model = AutoReg(train, lags=lags, old_names=False).fit()
params = model.params.copy()

intercept = float(params.get('const', params.get('Intercept', 0.0)))
ar_coefs = params.drop(labels=[k for k in params.index if k.lower() in ('const','intercept')]).values
ar_coefs = np.asarray(ar_coefs, dtype=float)

# -------------------------
# Predictions
# -------------------------
test_preds = []

for t in range(len(test)):
    hist = train.values if t == 0 else np.concatenate([train.values, test.values[:t]])

    yhat = intercept
    for i in range(min(len(ar_coefs), len(hist))):
        yhat += ar_coefs[i] * hist[-(i+1)]

    test_preds.append(yhat)

test_preds = pd.Series(test_preds, index=test.index)

# -------------------------
# Metrics
# -------------------------
metrics = evaluate(test, test_preds)

st.subheader("📊 Performance")
st.dataframe(pd.DataFrame([metrics]))

# -------------------------
# Chart (NO matplotlib)
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
model_full = AutoReg(series, lags=lags, old_names=False).fit()
params_full = model_full.params.copy()

intercept_full = float(params_full.get('const', params_full.get('Intercept', 0.0)))
ar_coefs_full = params_full.drop(labels=[k for k in params_full.index if k.lower() in ('const','intercept')]).values

history = list(series.values)
future_preds = []

for _ in range(horizon):
    yhat = intercept_full
    for i in range(min(len(ar_coefs_full), len(history))):
        yhat += ar_coefs_full[i] * history[-(i+1)]

    future_preds.append(yhat)
    history.append(yhat)

future_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon)
forecast_series = pd.Series(future_preds, index=future_index)

# -------------------------
# Forecast Chart
# -------------------------
st.subheader("🔮 Forecast")

forecast_df = pd.DataFrame({
    "Recent": series[-200:],
    "Forecast": forecast_series
})

st.line_chart(forecast_df)

# -------------------------
# Table + Download
# -------------------------
st.subheader("📅 Forecast Data")

forecast_df_display = forecast_series.reset_index()
forecast_df_display.columns = ["Date", "Forecast"]

st.dataframe(forecast_df_display)

csv = forecast_series.to_csv().encode("utf-8")
st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")
