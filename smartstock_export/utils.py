import pandas as pd
import numpy as np
import streamlit as st

def get_risk_score(answers, questions):
    """
    Calculate risk score based on questionnaire answers.
    
    Parameters:
    - answers (dict): Dictionary of question indices and selected answers
    - questions (list): List of question dictionaries with scores
    
    Returns:
    - float: Normalized risk score between 0 and 1
    """
    if not all(answer is not None for answer in answers.values()):
        return None
    
    total_score = 0
    max_possible_score = 0
    
    for i, question in enumerate(questions):
        if i in answers and answers[i] is not None:
            selected_option = answers[i]
            option_index = question["options"].index(selected_option)
            total_score += question["scores"][option_index]
            max_possible_score += max(question["scores"])
    
    # Normalize score to 0-1 range
    normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0.5
    
    return normalized_score

def filter_stocks_by_goal(stocks, goal):
    """
    Filter stocks based on investment goal.
    
    Parameters:
    - stocks (pandas.DataFrame): DataFrame containing stock data
    - goal (str): Investment goal
    
    Returns:
    - pandas.DataFrame: Filtered DataFrame
    """
    if stocks is None or stocks.empty:
        return pd.DataFrame()
    
    if goal == 'Long-term Growth':
        # Favor stocks with strong growth metrics
        return stocks[(stocks['ROE (%)'] > 10) & (stocks['P/E Ratio'] > 0)]
    
    elif goal == 'Dividend Income':
        # Favor stocks with higher dividend yields
        return stocks[stocks['Dividend Yield (%)'] > 1.5]
    
    elif goal == 'Value Investing':
        # Favor stocks with better value metrics
        return stocks[(stocks['P/E Ratio'] > 0) & (stocks['P/E Ratio'] < 20) & (stocks['Price/Book'] < 3)]
    
    elif goal == 'Balanced Approach':
        # Favor stocks with balanced metrics
        return stocks[(stocks['P/E Ratio'] > 0) & (stocks['P/E Ratio'] < 25) & (stocks['Dividend Yield (%)'] > 0.5)]
    
    elif goal == 'Short-term Trading':
        # Favor stocks with higher volatility for trading
        return stocks[stocks['Daily Volatility (%)'] > 1.5]
    
    # Default: return all stocks
    return stocks

def format_currency(amount):
    """
    Format a number as currency.
    
    Parameters:
    - amount (float): Amount to format
    
    Returns:
    - str: Formatted currency string
    """
    if amount >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.1f}K"
    else:
        return f"${amount:.2f}"
