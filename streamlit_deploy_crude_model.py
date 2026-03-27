import streamlit as st
import pandas as pd
import numpy as np

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Crude Oil Price Forecasting Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -------------------------
# TITLE SECTION
# -------------------------
st.title("🌍 Crude Oil Price Forecasting Dashboard")

st.write(
"""
This app forecasts crude oil prices using a simple time series model.  
You can visualize historical data, generate forecasts, and evaluate performance.
"""
)

# -------------------------
# FILE UPLOAD (LIKE YOUR UI)
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your Crude Oil CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -------------------------
    # DATA PREPROCESSING
    # -------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)

    series = df["Close/Last"].astype(float).ffill().bfill()

    st.success("✅ Data loaded successfully!")

    # -------------------------
    # SHOW SAMPLE DATA
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

    horizon = st.slider("Select number of days to forecast", 5, 60, 30)

    if st.button("🚀 Generate Forecast"):

        history = list(series)
        future_preds = []

        for _ in range(horizon):
            yhat = np.mean(history[-5:])  # simple model
            future_preds.append(yhat)
            history.append(yhat)

        future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
        forecast_series = pd.Series(future_preds, index=future_index)

        st.success("✅ Forecast generated successfully!")

        # -------------------------
        # FORECAST CHART
        # -------------------------
        forecast_df = pd.DataFrame({
            "Recent Data": series[-200:],
            "Forecast": forecast_series
        })

        st.line_chart(forecast_df)

        # -------------------------
        # TABLE
        # -------------------------
        st.subheader("📅 Forecast Data")
        st.dataframe(
            forecast_series.reset_index().rename(columns={"index":"Date",0:"Forecast"}),
            use_container_width=True
        )

        # -------------------------
        # DOWNLOAD
        # -------------------------
        csv = forecast_series.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Forecast", csv, "forecast.csv")
