import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

st.title("CAPM and Sharpe Ratio Calculator")

stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
index_ticker = st.text_input("Enter index ticker (e.g., ^GSPC for S&P 500)", "^GSPC")

if st.button("Calculate"):
    try:
        # Simplified date calculation
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5*365)
        
        # Use the built-in download function without custom session
        with st.spinner("Downloading stock data..."):
            stock_data = yf.download(
                stock_ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

        with st.spinner("Downloading index data..."):
            index_data = yf.download(
                index_ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
 
        if stock_data.empty:
            st.error(f"No stock data for {stock_ticker}. Check ticker on Yahoo Finance.")
            st.stop()
        if index_data.empty:
            st.error(f"No index data for {index_ticker}. Check ticker on Yahoo Finance.")
            st.stop()

        # Rest of your code remains the same...
        stock_close = stock_data['Close'].squeeze()
        index_close = index_data['Close'].squeeze()

        stock_returns = np.log(1 + stock_close.pct_change().dropna())
        index_returns = np.log(1 + index_close.pct_change().dropna())

        common_dates = stock_returns.index.intersection(index_returns.index)
        if len(common_dates) < 2:  
            st.error("Insufficient overlapping data points between stock and index.")
            st.stop()

        stock_returns_aligned = stock_returns.loc[common_dates]
        index_returns_aligned = index_returns.loc[common_dates]
        covariance = np.cov(stock_returns_aligned, index_returns_aligned)[0, 1]
        market_variance = index_returns_aligned.var()
        beta = float(covariance / market_variance) 
        rf = 0.0137
        market_return = float(index_returns_aligned.mean() * 252)
        capm_return = float(rf + beta * (market_return - rf))
        stock_volatility = float(stock_returns_aligned.std() * np.sqrt(252))
        sharpe_ratio = float((capm_return - rf) / stock_volatility)

        st.success(f"""
        **Results for {stock_ticker}:**
        - Beta: {beta:.2f}
        - CAPM Return: {capm_return*100:.2f}%
        - Sharpe Ratio: {sharpe_ratio:.2f}
        """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()
