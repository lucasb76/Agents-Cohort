import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📈", layout="wide")

st.title("📈 AI-Powered Revenue Forecasting")

# Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file with Date and Revenue columns", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())
    
    # Ensure correct column names
    if not all(col in df.columns for col in ["Date", "Revenue"]):
        st.error("The file must contain 'Date' and 'Revenue' columns.")
        st.stop()
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    
    # Prepare Data for Prophet
    df_prophet = df.rename(columns={"Date": "ds", "Revenue": "y"})
    
    # Train Prophet Model
    model = Prophet()
    model.fit(df_prophet)
    
    # Create Future Dataframe
    future = model.make_future_dataframe(periods=90)  # Forecast for 90 days
    forecast = model.predict(future)
    
    # Plot Forecast
    st.write("### Revenue Forecast")
    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    st.pyplot(fig)
    
    # Display Forecast Data
    st.write("### Forecasted Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Insights Summary
    st.write("### Key Insights from Forecast")
    latest_forecast = forecast.iloc[-1]
    st.write(f"- Projected revenue in {latest_forecast['ds'].date()}: ${latest_forecast['yhat']:.2f}")
    st.write(f"- Lower bound: ${latest_forecast['yhat_lower']:.2f}, Upper bound: ${latest_forecast['yhat_upper']:.2f}")
    
    # AI Commentary
    st.subheader("🤖 AI-Generated Forecast Insights")
    st.write("(AI-generated commentary using financial forecasting analysis coming soon!)")
