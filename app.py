import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

# Load trained model
model = joblib.load('StockPredictor/model.pkl')

# UI
st.title("📈 Stock Price Predictor")
st.markdown("Predict the next day's stock price based on historical close price.")

ticker = st.text_input("Enter stock ticker (e.g. AAPL):", "AAPL")

if st.button("Predict"):
    # Fetch data
    df = yf.download(ticker, period="60d", interval="1d")
    if df.empty:
        st.error("Invalid ticker or no data found.")
    else:
        latest_price = df['Close'].iloc[-1]
        predicted_price = model.predict([[latest_price.item()]])
        st.success(f"Predicted next-day price: **${predicted_price[0]:.2f}**")

        # Optional: show chart
        st.line_chart(df['Close'])
