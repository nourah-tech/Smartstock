import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def analyze_stocks(tickers, period="1y"):
    """
    Analyze a list of stock tickers to extract key metrics and performance indicators.
    
    Parameters:
    - tickers (list): List of stock ticker symbols
    - period (str): Time period for historical data analysis
    
    Returns:
    - pandas.DataFrame: DataFrame containing analysis results for each stock
    """
    results = []
    
    # Use yfinance to get stock data
    for ticker in tickers:
        try:
            # Get stock information
            stock = yf.Ticker(ticker)
            
            # Get historical market data
            hist = stock.history(period=period)
            
            if hist.empty:
                continue
                
            # Basic information
            info = stock.info
            
            # Make sure we have valid info
            if not info or not isinstance(info, dict):
                print(f"No valid info data for {ticker}")
                continue
                
            # Get the last close price
            try:
                current_price = hist['Close'].iloc[-1]
                if not np.isfinite(current_price):
                    print(f"Invalid price for {ticker}")
                    continue
            except Exception as e:
                print(f"Error getting price for {ticker}: {str(e)}")
                continue
            
            # Calculate price change safely
            try:
                first_price = hist['Close'].iloc[0]
                if np.isfinite(first_price) and first_price > 0:
                    price_change = ((current_price / first_price) - 1) * 100
                else:
                    price_change = np.nan
            except Exception:
                price_change = np.nan
                
            # Extract key metrics with safe handling of each value
            data = {
                'Ticker': ticker,
                'Company': info.get('shortName', 'Unknown'),
                'Sector': info.get('sector', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'Current Price': current_price,
                'Price Change (%)': price_change,
                'Market Cap': info.get('marketCap', np.nan),
                'P/E Ratio': info.get('trailingPE', np.nan),
                'Forward P/E': info.get('forwardPE', np.nan),
                'PEG Ratio': info.get('pegRatio', np.nan),
                'Price/Book': info.get('priceToBook', np.nan),
                'Dividend Yield (%)': info.get('dividendYield', 0) * 100 if info.get('dividendYield') and np.isfinite(info.get('dividendYield', 0)) else 0,
                'Dividend Rate': info.get('dividendRate', 0) if info.get('dividendRate') and np.isfinite(info.get('dividendRate', 0)) else 0,
                'EPS': info.get('trailingEps', np.nan),
                'Beta': info.get('beta', np.nan),
                'ROE (%)': info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') and np.isfinite(info.get('returnOnEquity', np.nan)) else np.nan,
                'ROA (%)': info.get('returnOnAssets', np.nan) * 100 if info.get('returnOnAssets') and np.isfinite(info.get('returnOnAssets', np.nan)) else np.nan,
                'Debt to Equity': info.get('debtToEquity', np.nan) / 100 if info.get('debtToEquity') and np.isfinite(info.get('debtToEquity', np.nan)) else np.nan,
                'Current Ratio': info.get('currentRatio', np.nan),
                'Quick Ratio': info.get('quickRatio', np.nan),
                'Profit Margin (%)': info.get('profitMargins', np.nan) * 100 if info.get('profitMargins') and np.isfinite(info.get('profitMargins', np.nan)) else np.nan,
                'Operating Margin (%)': info.get('operatingMargins', np.nan) * 100 if info.get('operatingMargins') and np.isfinite(info.get('operatingMargins', np.nan)) else np.nan,
                'Analyst Target Price': info.get('targetMeanPrice', np.nan),
                'Analyst Rating': info.get('recommendationKey', 'Unknown'),
            }
            
            # Verify that numeric values are all finite
            for key, value in data.items():
                if key != 'Ticker' and key != 'Company' and key != 'Sector' and key != 'Industry' and key != 'Analyst Rating':
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        data[key] = np.nan
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = hist['Close'].pct_change().dropna()
            data['Daily Volatility (%)'] = daily_returns.std() * 100
            data['Annualized Volatility (%)'] = data['Daily Volatility (%)'] * np.sqrt(252)  # ~252 trading days in a year
            
            # Calculate Sharpe Ratio (assuming risk-free rate of 1.5%)
            risk_free_rate = 0.015  # 1.5% annual risk-free rate
            daily_risk_free = risk_free_rate / 252
            excess_return = daily_returns - daily_risk_free
            data['Sharpe Ratio'] = (excess_return.mean() / daily_returns.std()) * np.sqrt(252)
            
            # Technical indicators
            # 50-day and 200-day moving averages
            if len(hist) >= 200:
                data['50-Day MA'] = hist['Close'].rolling(window=50).mean().iloc[-1]
                data['200-Day MA'] = hist['Close'].rolling(window=200).mean().iloc[-1]
                data['50-200 Day MA Difference (%)'] = ((data['50-Day MA'] / data['200-Day MA']) - 1) * 100
            else:
                data['50-Day MA'] = np.nan
                data['200-Day MA'] = np.nan
                data['50-200 Day MA Difference (%)'] = np.nan
            
            # Relative Strength Index (RSI) - 14-day
            if len(hist) >= 14:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                
                # Calculate RS and RSI
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                data['RSI (14-day)'] = rsi.iloc[-1]
            else:
                data['RSI (14-day)'] = np.nan
            
            results.append(data)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            continue
    
    # Convert to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # Fill missing values with appropriate defaults
        results_df = results_df.replace([np.inf, -np.inf], np.nan)
        numerical_columns = results_df.select_dtypes(include=[np.number]).columns
        results_df[numerical_columns] = results_df[numerical_columns].fillna(results_df[numerical_columns].median())
        
        return results_df
    else:
        return None

def calculate_risk_score(stock_data):
    """
    Calculate a risk score for each stock based on multiple factors.
    
    Parameters:
    - stock_data (pandas.DataFrame): DataFrame with stock analysis data
    
    Returns:
    - pandas.Series: Risk scores for each stock
    """
    if stock_data is None or stock_data.empty:
        return None
    
    # Define risk factors and their weights
    risk_factors = {
        'Beta': 0.3,
        'Annualized Volatility (%)': 0.3,
        'Debt to Equity': 0.15,
        'P/E Ratio': 0.1,
        'Current Ratio': -0.15  # Negative weight because higher current ratio means lower risk
    }
    
    # Initialize risk score
    risk_score = pd.Series(0, index=stock_data.index)
    
    # Calculate weighted risk components
    for factor, weight in risk_factors.items():
        if factor in stock_data.columns:
            # Normalize the factor values to 0-1 scale
            scaler = MinMaxScaler()
            
            # Handle the Current Ratio separately (higher is better, so invert)
            if factor == 'Current Ratio' and 'Current Ratio' in stock_data.columns:
                values = 1 / stock_data['Current Ratio'].values.reshape(-1, 1)
                scaled_values = scaler.fit_transform(values)
                risk_score += abs(weight) * scaled_values.flatten()
            else:
                # For all other factors, higher values mean higher risk
                values = stock_data[factor].values.reshape(-1, 1)
                scaled_values = scaler.fit_transform(values)
                risk_score += weight * scaled_values.flatten()
    
    # Normalize final risk score to 0-1 range
    normalized_risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    
    return normalized_risk_score

def estimate_expected_returns(stock_data):
    """
    Estimate expected annual returns based on multiple factors.
    
    Parameters:
    - stock_data (pandas.DataFrame): DataFrame with stock analysis data
    
    Returns:
    - pandas.Series: Estimated annual returns for each stock
    """
    if stock_data is None or stock_data.empty:
        return None
    
    # Define return factors and their weights
    return_factors = {
        'Analyst Target Price': 0.3,  # Analyst price target vs current price
        'ROE (%)': 0.25,              # Return on Equity
        'Profit Margin (%)': 0.2,     # Profit Margin
        'Price Change (%)': 0.15,     # Historical price performance
        'Dividend Yield (%)': 0.1     # Dividend yield
    }
    
    # Initialize expected return
    expected_return = pd.Series(0.0, index=stock_data.index)
    
    # Calculate expected return based on analyst target price
    if 'Analyst Target Price' in stock_data.columns and 'Current Price' in stock_data.columns:
        price_growth = ((stock_data['Analyst Target Price'] / stock_data['Current Price']) - 1) * 100
        # Cap the growth at reasonable levels (-30% to +50%)
        price_growth = price_growth.clip(-30, 50)
        expected_return += return_factors['Analyst Target Price'] * price_growth / 100 * 100  # Convert to percentage
    
    # Add ROE component
    if 'ROE (%)' in stock_data.columns:
        # Scale ROE to a reasonable return range (0-30%)
        roe_component = stock_data['ROE (%)'].clip(0, 30) * return_factors['ROE (%)'] / 100 * 100
        expected_return += roe_component
    
    # Add profit margin component
    if 'Profit Margin (%)' in stock_data.columns:
        margin_component = stock_data['Profit Margin (%)'].clip(0, 30) * return_factors['Profit Margin (%)'] / 100 * 100
        expected_return += margin_component
    
    # Add historical performance component, annualized
    if 'Price Change (%)' in stock_data.columns:
        # Annualize the historical return (assuming the period is 1 year)
        historical_component = stock_data['Price Change (%)'].clip(-20, 40) * return_factors['Price Change (%)'] / 100 * 100
        expected_return += historical_component
    
    # Add dividend yield component
    if 'Dividend Yield (%)' in stock_data.columns:
        expected_return += stock_data['Dividend Yield (%)'] * return_factors['Dividend Yield (%)']
    
    # Ensure returns are within reasonable bounds (0% to 30%)
    expected_return = expected_return.clip(0, 30)
    
    return expected_return

def get_recommendation_score(stock_data, user_goal, user_risk_tolerance, desired_returns):
    """
    Calculate a recommendation score for each stock based on user preferences.
    
    Parameters:
    - stock_data (pandas.DataFrame): DataFrame with stock analysis data
    - user_goal (str): User's investment goal
    - user_risk_tolerance (str): User's risk tolerance level
    - desired_returns (float): User's desired annual returns
    
    Returns:
    - pandas.Series: Recommendation scores for each stock
    """
    if stock_data is None or stock_data.empty:
        return None
    
    # Initialize recommendation score
    recommendation_score = pd.Series(0.0, index=stock_data.index)
    
    # Factor 1: Match between stock risk and user risk tolerance
    risk_levels = {
        'Low': (0, 0.4),
        'Moderate': (0.4, 0.7),
        'High': (0.7, 1.0)
    }
    
    risk_range = risk_levels[user_risk_tolerance]
    
    # Calculate how well the stock's risk matches user's tolerance
    try:
        # Use explicit pandas operations to avoid Series truth value ambiguity
        mid_risk = (risk_range[0] + risk_range[1]) / 2
        risk_diff = (stock_data['Risk Score'] - mid_risk).abs() / 0.5
        risk_match = 1.0 - risk_diff
        risk_match = risk_match.clip(0, 1)  # Ensure values are between 0 and 1
    except Exception as e:
        print(f"Error calculating risk match: {str(e)}")
        # Fallback to middle score if there's an error
        risk_match = pd.Series(0.5, index=stock_data.index)
    
    # Factor 2: Match between expected returns and desired returns
    try:
        # Make sure desired returns is not zero to avoid division by zero
        safe_desired_returns = max(0.1, desired_returns)
        # Use element-wise operations with pandas to avoid Series truth value issues
        return_diff = (stock_data['Expected Annual Return (%)'] - safe_desired_returns).abs() / safe_desired_returns
        return_match = 1.0 - return_diff.clip(0, 1)  # Higher score for closer match
    except Exception as e:
        print(f"Error calculating return match: {str(e)}")
        # Fallback to middle score if there's an error
        return_match = pd.Series(0.5, index=stock_data.index)
    
    # Factor 3: Goal-specific suitability
    goal_match = pd.Series(0.5, index=stock_data.index)  # Default middle score
    
    if user_goal == 'Long-term Growth':
        # Favor stocks with higher expected growth and moderate to high P/E
        if 'ROE (%)' in stock_data.columns and 'P/E Ratio' in stock_data.columns:
            growth_potential = (stock_data['ROE (%)'] / 20).clip(0, 1)  # Normalize ROE to 0-1
            pe_factor = 1.0 - ((stock_data['P/E Ratio'] - 15).abs() / 15).clip(0, 1)  # Prefer P/E around 15
            goal_match = (growth_potential * 0.7 + pe_factor * 0.3).clip(0, 1)
    
    elif user_goal == 'Dividend Income':
        # Favor stocks with higher dividend yields
        if 'Dividend Yield (%)' in stock_data.columns:
            div_yield_factor = (stock_data['Dividend Yield (%)'] / 5).clip(0, 1)  # Normalize dividend yield to 0-1
            goal_match = div_yield_factor
    
    elif user_goal == 'Value Investing':
        # Favor stocks with lower P/E, P/B, and higher dividend yield
        if 'P/E Ratio' in stock_data.columns and 'Price/Book' in stock_data.columns:
            pe_factor = 1.0 - ((stock_data['P/E Ratio'] / 25).clip(0, 1))  # Lower P/E is better
            pb_factor = 1.0 - ((stock_data['Price/Book'] / 3).clip(0, 1))  # Lower P/B is better
            value_score = (pe_factor * 0.6 + pb_factor * 0.4).clip(0, 1)
            goal_match = value_score
    
    elif user_goal == 'Balanced Approach':
        # Favor stocks with moderate metrics across the board
        if 'P/E Ratio' in stock_data.columns and 'Dividend Yield (%)' in stock_data.columns:
            pe_factor = 1.0 - ((stock_data['P/E Ratio'] - 18).abs() / 18).clip(0, 1)  # Prefer P/E around 18
            div_factor = ((stock_data['Dividend Yield (%)'] / 3).clip(0, 1))  # Some dividend is good
            balanced_score = (pe_factor * 0.5 + div_factor * 0.5).clip(0, 1)
            goal_match = balanced_score
    
    elif user_goal == 'Short-term Trading':
        # Favor stocks with higher volatility and momentum
        if 'RSI (14-day)' in stock_data.columns and 'Annualized Volatility (%)' in stock_data.columns:
            # RSI factor: prefer RSI in the 40-60 range (not overbought or oversold)
            rsi_factor = 1.0 - ((stock_data['RSI (14-day)'] - 50).abs() / 30).clip(0, 1)
            # Volatility factor: higher volatility preferable for trading
            vol_factor = (stock_data['Annualized Volatility (%)'] / 30).clip(0, 1)
            trading_score = (rsi_factor * 0.4 + vol_factor * 0.6).clip(0, 1)
            goal_match = trading_score
    
    # Combine all factors into final recommendation score
    recommendation_score = (
        risk_match * 0.35 +   # Weight for risk match
        return_match * 0.35 + # Weight for return match
        goal_match * 0.3      # Weight for goal-specific match
    )
    
    # Ensure final scores are between 0 and 1
    recommendation_score = recommendation_score.clip(0, 1)
    
    return recommendation_score
