import pandas as pd
import numpy as np
import yfinance as yf
from stock_analyzer import analyze_stocks, calculate_risk_score, estimate_expected_returns, get_recommendation_score
from data_fetcher import get_stock_universe_by_goal
import streamlit as st

def get_recommendations(goal, risk_tolerance, investment_amount, time_horizon, desired_returns, risk_score):
    """
    Generate stock recommendations based on user preferences.
    
    Parameters:
    - goal (str): User's investment goal ('Long-term Growth', 'Dividend Income', etc.)
    - risk_tolerance (str): User's risk tolerance ('Low', 'Moderate', 'High')
    - investment_amount (float): Amount to invest in dollars
    - time_horizon (str): Investment time horizon
    - desired_returns (float): Target annual returns in percentage
    - risk_score (float): Calculated risk score from questionnaire (0-1)
    
    Returns:
    - pandas.DataFrame: DataFrame containing recommended stocks and their metrics
    """
    try:
        # Cache function to avoid re-running expensive operations
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def fetch_and_analyze_stocks(goal_param, risk_tolerance_param):
            try:
                # Get a list of stocks based on the investment goal
                stock_universe = get_stock_universe_by_goal(goal_param, risk_tolerance_param)
                
                if not stock_universe:
                    st.warning("Could not get stock universe. Using default stocks.")
                    # Fallback to a few reliable stock tickers if universe is empty
                    stock_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ', 'PG', 'KO', 'VZ', 'PEP', 'WMT']
                
                # Analyze the stocks
                analyzed_stocks = analyze_stocks(stock_universe)
                
                if analyzed_stocks is None or analyzed_stocks.empty:
                    st.error("Could not analyze any stocks. Please try again later.")
                    return None
                    
                return analyzed_stocks
            except Exception as e:
                st.error(f"Error fetching and analyzing stocks: {str(e)}")
                return None
        
        # Get and analyze stocks
        stock_data = fetch_and_analyze_stocks(goal, risk_tolerance)
        
        if stock_data is None or stock_data.empty:
            # If we couldn't get any stocks, return empty DataFrame
            st.error("No stock data could be retrieved.")
            return pd.DataFrame()
        
        # Calculate risk scores for each stock
        try:
            risk_scores = calculate_risk_score(stock_data)
            if risk_scores is not None:
                stock_data['Risk Score'] = risk_scores
            else:
                # Use default mid-range risk score if calculation fails
                stock_data['Risk Score'] = 0.5
        except Exception as e:
            st.warning(f"Error calculating risk scores: {str(e)}")
            stock_data['Risk Score'] = 0.5
        
        # Estimate expected returns
        try:
            returns = estimate_expected_returns(stock_data)
            if returns is not None:
                stock_data['Expected Annual Return (%)'] = returns
            else:
                # Use default expected return
                stock_data['Expected Annual Return (%)'] = 8.0
        except Exception as e:
            st.warning(f"Error estimating returns: {str(e)}")
            stock_data['Expected Annual Return (%)'] = 8.0
        
        # Calculate recommendation scores based on user profile
        try:
            rec_scores = get_recommendation_score(stock_data, goal, risk_tolerance, desired_returns)
            if rec_scores is not None:
                stock_data['Recommendation Score'] = rec_scores
            else:
                # Use default score based on risk match if calculation fails
                if 'Risk Score' in stock_data.columns:
                    risk_level = {'Low': 0.2, 'Moderate': 0.5, 'High': 0.8}
                    target_risk = risk_level.get(risk_tolerance, 0.5)
                    stock_data['Recommendation Score'] = 1.0 - abs(stock_data['Risk Score'] - target_risk)
                else:
                    stock_data['Recommendation Score'] = 0.5
        except Exception as e:
            st.warning(f"Error calculating recommendation scores: {str(e)}")
            stock_data['Recommendation Score'] = 0.5
        
        # Sort by recommendation score in descending order
        recommendations = stock_data.sort_values('Recommendation Score', ascending=False)
        
        # Select top recommendations (up to 20)
        top_recommendations = recommendations.head(20)
        
        # Make sure we have all required columns
        required_columns = [
            'Ticker', 'Company', 'Sector', 'Current Price', 'P/E Ratio',
            'Dividend Yield (%)', 'Risk Score', 'Expected Annual Return (%)',
            'Recommendation Score'
        ]
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in top_recommendations.columns:
                if col == 'Ticker' or col == 'Company' or col == 'Sector':
                    top_recommendations[col] = 'Unknown'
                else:
                    top_recommendations[col] = np.nan
        
        # Create final recommendation DataFrame with selected columns
        result = top_recommendations[required_columns].copy()
        
        # Rename column for clarity
        result = result.rename(columns={'Current Price': 'Price ($)'})
        
        # Replace any remaining NaN, inf, or -inf values
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with appropriate defaults
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if result[col].isna().all():
                # If all values are NaN, use a sensible default
                if col == 'Price ($)':
                    result[col] = 100
                elif col == 'P/E Ratio':
                    result[col] = 20
                elif col == 'Dividend Yield (%)':
                    result[col] = 1.5
                elif col == 'Risk Score':
                    result[col] = 0.5
                elif col == 'Expected Annual Return (%)':
                    result[col] = 8.0
                elif col == 'Recommendation Score':
                    result[col] = 0.5
            else:
                # Use median for columns with some valid data
                median_val = result[col].median()
                if np.isfinite(median_val):
                    result[col] = result[col].fillna(median_val)
                else:
                    # Fallback defaults if median is not valid
                    if col == 'Price ($)':
                        result[col] = result[col].fillna(100)
                    elif col == 'P/E Ratio':
                        result[col] = result[col].fillna(20)
                    elif col == 'Dividend Yield (%)':
                        result[col] = result[col].fillna(1.5)
                    elif col == 'Risk Score':
                        result[col] = result[col].fillna(0.5)
                    elif col == 'Expected Annual Return (%)':
                        result[col] = result[col].fillna(8.0)
                    elif col == 'Recommendation Score':
                        result[col] = result[col].fillna(0.5)
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        # Create empty DataFrame with required columns as fallback
        result = pd.DataFrame(columns=[
            'Ticker', 'Company', 'Sector', 'Price ($)', 'P/E Ratio',
            'Dividend Yield (%)', 'Risk Score', 'Expected Annual Return (%)',
            'Recommendation Score'
        ])
    
    return result
