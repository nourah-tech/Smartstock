import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(tickers, period="1y"):
    """
    Fetch historical stock price data for the given tickers.
    
    Parameters:
    - tickers (list): List of stock ticker symbols
    - period (str): Period of historical data to retrieve
    
    Returns:
    - dict: Dictionary with tickers as keys and pandas.Series of closing prices as values
    """
    if not tickers:
        return {}
        
    # Make sure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    result = {}
    
    try:
        # Fetch data for each ticker individually using Ticker object for more reliable data
        for ticker in tickers:
            try:
                # Create a Ticker object for more detailed control
                ticker_obj = yf.Ticker(ticker)
                
                # Fetch history using the Ticker object which can be more reliable
                ticker_data = ticker_obj.history(period=period)
                
                # Check if we have Close data and it's valid
                if 'Close' in ticker_data.columns and not ticker_data['Close'].empty:
                    close_prices = ticker_data['Close']
                    
                    # Additional validation of the first price
                    first_price = close_prices.iloc[0] if len(close_prices) > 0 else None
                    
                    if first_price is not None and pd.notna(first_price) and np.isfinite(first_price) and first_price > 0:
                        result[ticker] = close_prices
                    else:
                        st.warning(f"Invalid initial price data for {ticker}")
                else:
                    st.warning(f"No closing price data available for {ticker}")
            except Exception as ticker_error:
                st.warning(f"Could not fetch data for {ticker}: {str(ticker_error)}")
                continue
        
        if not result:
            st.error("Could not fetch valid data for any of the selected stocks.")
            return None
            
        return result
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_fundamentals(tickers):
    """
    Fetch fundamental data for the given stock tickers.
    
    Parameters:
    - tickers (list): List of stock ticker symbols
    
    Returns:
    - pandas.DataFrame: DataFrame containing fundamental metrics for each stock
    """
    if not tickers:
        return None
    
    # Make sure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    fundamentals = {}
    
    for ticker in tickers:
        try:
            # Fetch data for this specific ticker
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid info data
            if not info or not isinstance(info, dict):
                st.warning(f"Could not fetch fundamental data for {ticker}")
                continue
                
            metrics = {
                'P/E Ratio': info.get('trailingPE', np.nan),
                'Forward P/E': info.get('forwardPE', np.nan),
                'P/S Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'P/B Ratio': info.get('priceToBook', np.nan),
                'EPS (TTM)': info.get('trailingEps', np.nan),
                'Dividend Yield (%)': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'ROE (%)': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan,
                'ROA (%)': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') else np.nan,
                'Debt to Equity': info.get('debtToEquity', np.nan) / 100 if info.get('debtToEquity') else np.nan,
                'Current Ratio': info.get('currentRatio', np.nan),
                'Profit Margin (%)': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') else np.nan,
                'Operating Margin (%)': info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') else np.nan,
                'Market Cap (B)': info.get('marketCap', np.nan) / 1e9 if info.get('marketCap') else np.nan,
                'Beta': info.get('beta', np.nan),
                '52W High': info.get('fiftyTwoWeekHigh', np.nan),
                '52W Low': info.get('fiftyTwoWeekLow', np.nan),
            }
            
            # Check for valid metric values
            valid_metrics = {}
            for key, value in metrics.items():
                # Convert any invalid values to np.nan
                if value is None or not np.isfinite(value):
                    valid_metrics[key] = np.nan
                else:
                    valid_metrics[key] = value
            
            fundamentals[ticker] = valid_metrics
            
        except Exception as e:
            st.warning(f"Error fetching fundamentals for {ticker}: {str(e)}")
            continue
    
    if fundamentals:
        try:
            # Create DataFrame from collected data
            fundamentals_df = pd.DataFrame.from_dict(fundamentals, orient='index')
            
            # Replace infinities and fill missing values
            fundamentals_df = fundamentals_df.replace([np.inf, -np.inf], np.nan)
            
            # Only use median fill for columns that have some valid data
            valid_columns = [col for col in fundamentals_df.columns if fundamentals_df[col].notna().any()]
            for col in valid_columns:
                median_val = fundamentals_df[col].median()
                if np.isfinite(median_val):
                    fundamentals_df[col] = fundamentals_df[col].fillna(median_val)
            
            return fundamentals_df
        except Exception as e:
            st.error(f"Error processing fundamentals data: {str(e)}")
            return None
    else:
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_indices():
    """
    Fetch current values and changes for major market indices.
    
    Returns:
    - pandas.DataFrame: DataFrame containing index values and changes
    """
    # Common market indices
    indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
    index_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']
    
    try:
        # Fetch data
        data = yf.download(indices, period="2d", group_by='ticker')
        
        if data.empty:
            return None
        
        # Prepare results
        results = {}
        
        for i, idx in enumerate(indices):
            if len(indices) == 1:
                # Handle single index case
                if data.empty or len(data) < 2:
                    continue
                
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[-2]
                change = current - previous
                change_percent = (change / previous) * 100
                
                results[index_names[i]] = {
                    'price': current,
                    'change': change,
                    'change_percent': change_percent
                }
            else:
                # Handle multiple indices
                if idx not in data.columns or 'Close' not in data[idx].columns or len(data[idx]) < 2:
                    continue
                
                current = data[idx]['Close'].iloc[-1]
                previous = data[idx]['Close'].iloc[-2]
                change = current - previous
                change_percent = (change / previous) * 100
                
                results[index_names[i]] = {
                    'price': current,
                    'change': change,
                    'change_percent': change_percent
                }
        
        if results:
            return pd.DataFrame.from_dict(results, orient='index')
        else:
            return None
            
    except Exception as e:
        st.error(f"Error fetching market indices: {str(e)}")
        return None

def get_stock_universe_by_goal(goal, risk_tolerance):
    """
    Get a list of stock tickers appropriate for the given investment goal and risk tolerance.
    
    Parameters:
    - goal (str): Investment goal
    - risk_tolerance (str): Risk tolerance level
    
    Returns:
    - list: List of stock ticker symbols
    """
    # Base sets of stocks for different goals and risk levels
    stock_sets = {
        'Long-term Growth': {
            'Low': ['MSFT', 'AAPL', 'JNJ', 'PG', 'KO', 'VZ', 'PEP', 'WMT', 'MRK', 'PFE'],
            'Moderate': ['AMZN', 'GOOGL', 'V', 'HD', 'MA', 'UNH', 'DIS', 'ADBE', 'CRM', 'COST', 'ACN', 'ABT'],
            'High': ['TSLA', 'NVDA', 'AMD', 'SQ', 'PYPL', 'SHOP', 'ROKU', 'ZM', 'CRWD', 'SNAP', 'NET', 'DDOG']
        },
        'Dividend Income': {
            'Low': ['JNJ', 'PG', 'KO', 'PEP', 'T', 'VZ', 'MO', 'PM', 'XOM', 'CVX', 'IBM', 'MMM'],
            'Moderate': ['MSFT', 'AAPL', 'PFE', 'MRK', 'ABT', 'HD', 'TGT', 'WMT', 'CSCO', 'INTC', 'ADP', 'TXN'],
            'High': ['SPG', 'O', 'STOR', 'DLR', 'WPC', 'IRM', 'VTR', 'KMI', 'ET', 'EPD', 'MMP', 'MPLX']
        },
        'Value Investing': {
            'Low': ['JNJ', 'PG', 'PEP', 'KO', 'WMT', 'PFE', 'MRK', 'VZ', 'T', 'IBM'],
            'Moderate': ['INTC', 'CVX', 'XOM', 'C', 'WFC', 'BAC', 'JPM', 'GM', 'F', 'DIS', 'CSCO', 'MMM'],
            'High': ['UAA', 'GPS', 'M', 'KSS', 'DAL', 'UAL', 'CCL', 'RCL', 'OXY', 'HAL', 'MRO', 'DVN']
        },
        'Balanced Approach': {
            'Low': ['MSFT', 'AAPL', 'JNJ', 'PG', 'V', 'MA', 'KO', 'PEP', 'WMT', 'HD', 'MRK', 'PFE'],
            'Moderate': ['AMZN', 'GOOGL', 'FB', 'CSCO', 'INTC', 'ADBE', 'CRM', 'ABT', 'UNH', 'VZ', 'T', 'DIS'],
            'High': ['PYPL', 'SQ', 'AMD', 'NVDA', 'NFLX', 'TSLA', 'BABA', 'SHOP', 'ROKU', 'ZM', 'NET', 'DDOG']
        },
        'Short-term Trading': {
            'Low': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB'],
            'Moderate': ['TSLA', 'NVDA', 'AMD', 'SQ', 'ROKU', 'ZM', 'PYPL', 'SHOP', 'NFLX', 'BABA', 'DDOG', 'NET'],
            'High': ['GME', 'AMC', 'PLTR', 'NIO', 'SPCE', 'PLUG', 'MARA', 'RIOT', 'TLRY', 'COIN', 'HOOD', 'LCID']
        }
    }
    
    # Get base set of stocks for the goal and risk level
    base_stocks = stock_sets.get(goal, {}).get(risk_tolerance, [])
    
    # Add some stocks from other risk levels to diversify
    all_goal_stocks = []
    for risk_level in ['Low', 'Moderate', 'High']:
        if risk_level != risk_tolerance:
            all_goal_stocks.extend(stock_sets.get(goal, {}).get(risk_level, []))
    
    # Randomly select a few stocks from other risk levels
    np.random.seed(42)  # For reproducibility
    additional_stocks = np.random.choice(all_goal_stocks, size=min(5, len(all_goal_stocks)), replace=False).tolist()
    
    # Add some ETFs for diversification
    etfs = ['SPY', 'QQQ', 'VTI', 'VOO', 'IVV']
    
    # Combine all stocks, removing duplicates
    combined_stocks = list(set(base_stocks + additional_stocks + etfs))
    
    return combined_stocks
