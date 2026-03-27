# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ------------------------
# 1. Page Config & Style
# ------------------------
st.set_page_config(page_title="Crude Oil Price Forecast", layout="wide")

# Custom CSS for background and style
st.markdown("""
    <style>
        /* Background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #e3f2fd, #ffffff);
            background-attachment: fixed;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #bbdefb, #e3f2fd);
        }
        /* Titles */
        h1, h2, h3 {
            color: #0d47a1;
        }
        /* Success and info boxes */
        .stSuccess, .stInfo {
            border-radius: 10px;
        }
        /* DataFrame style */
        .stDataFrame {
            background-color: #fafafa;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# 2. App Title
# ------------------------
st.title("🌍 Crude Oil Price Forecasting Dashboard")
st.markdown("""
This app forecasts **crude oil prices** using the **ARIMA(5,1,0)** model.  
You can visualize historical data, generate future forecasts, and evaluate directional accuracy.
""")

# ------------------------
# 3. File Upload
# ------------------------
uploaded_file = st.file_uploader("📂 Upload your Crude Oil CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df.rename(columns={'Close/Last': 'Close'}, inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(subset=['Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    st.success("✅ Data loaded successfully!")
    
    # Show sample data
    st.subheader("Historical Data Sample")
    st.dataframe(df.tail(10))
    
    # ------------------------
    # 4. Historical Visualization
    # ------------------------
    st.subheader("📊 Historical Closing Prices")
    st.line_chart(df['Close'])
    
    # ------------------------
    # 5. Forecasting Section
    # ------------------------
    st.subheader("🔮 Forecast Future Prices")
    forecast_steps = st.slider("Select number of days to forecast", min_value=1, max_value=200, value=30)
    
    if st.button("🚀 Generate Forecast"):
        with st.spinner("⏳ Fitting ARIMA model and generating forecast..."):
            model = ARIMA(df['Close'], order=(5, 1, 0))
            result = model.fit()
            
            forecast_result = result.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_result.predicted_mean
            forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Close': forecast_mean.values})
            
            # Plot forecast
            st.subheader(f"📈 {forecast_steps}-Day Forecast")
            fig, ax = plt.subplots(figsize=(12,6))
            df['Close'][-200:].plot(ax=ax, label='Historical Price (Last 200 Days)')
            forecast_df.set_index('Date')['Predicted_Close'].plot(ax=ax, label='Forecast', linestyle='--', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            st.subheader("📋 Forecasted Prices Table")
            st.dataframe(forecast_df)
            
            st.success("🎯 Forecast generated successfully!")
    
    # ------------------------
    # 6. Directional Accuracy Section
    # ------------------------
    if st.checkbox("📏 Show Directional Accuracy (Test Data)"):
        forecast_steps_test = min(forecast_steps, len(df) - 1)
        
        train = df['Close'][:-forecast_steps_test]
        test = df['Close'][-forecast_steps_test:]
        
        with st.spinner("🔍 Calculating directional accuracy..."):
            model_test = ARIMA(train, order=(5, 1, 0))
            result_test = model_test.fit()
            
            forecast_test = result_test.get_forecast(steps=forecast_steps_test).predicted_mean
            forecast_test.index = test.index
            
            # Compute directional accuracy
            actual_diff = test.diff().dropna()
            pred_diff = forecast_test.diff().dropna()
            directional_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
            
            st.metric("Directional Accuracy (%)", f"{directional_accuracy:.2f}%")
            
            # Plot Actual vs Predicted
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(test.index, test.values, label='Actual Prices', color='blue')
            ax.plot(forecast_test.index, forecast_test.values, label='Predicted Prices', color='red', linestyle='--')
            ax.set_title(f"Directional Accuracy Comparison ({forecast_steps_test} Days)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Actual vs Predicted Table
            st.subheader("📊 Actual vs Predicted Price Comparison")
            comparison_df = pd.DataFrame({
                'Date': test.index,
                'Actual_Price': test.values,
                'Predicted_Price': forecast_test.values
            }).reset_index(drop=True)
            
            st.dataframe(comparison_df)
            st.success("✅ Directional accuracy test completed!")

else:
    st.info("📥 Please upload a CSV file to begin. The file should have columns **'Date'** and **'Close/Last'**.")
