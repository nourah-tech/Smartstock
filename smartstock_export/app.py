import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from stock_analyzer import analyze_stocks
from recommendation_engine import get_recommendations
from data_fetcher import fetch_stock_data, fetch_market_indices, fetch_stock_fundamentals
from utils import get_risk_score, filter_stocks_by_goal, format_currency
from auth import initialize_auth, show_auth_ui, logout, show_portfolio, add_to_portfolio

# Set page configuration
st.set_page_config(
    page_title="StockSmart - AI Stock Recommendations",
    page_icon="📈",
    layout="wide"
)

# Initialize authentication
initialize_auth()

# Initialize session state variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'goal': None,
        'risk_tolerance': None,
        'investment_amount': None,
        'contribution_frequency': 'None',
        'contribution_amount': None,
        'time_horizon': None,
        'desired_returns': None,
        'risk_score': None,
    }
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []

# Function to navigate between pages
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Welcome page
def show_welcome_page():
    st.title("StockSmart: AI-Powered Stock Recommendations")

    st.markdown("""
    Welcome to StockSmart, your intelligent assistant for finding stocks 
    that match your financial goals and risk tolerance.

    **How it works:**
    1. Tell us about your investment goals and risk tolerance
    2. Provide details about your financial situation
    3. Get personalized stock recommendations backed by AI analysis
    4. Compare and evaluate recommended stocks
    5. Make informed investment decisions
    """)

    # Fetch and display some market indices for context
    try:
        indices = fetch_market_indices()
        if indices is not None and not indices.empty:
            st.subheader("Current Market Overview")
            cols = st.columns(len(indices))
            for i, (index_name, index_data) in enumerate(indices.iterrows()):
                with cols[i]:
                    st.metric(
                        index_name, 
                        f"{index_data['price']:.2f}", 
                        f"{index_data['change_percent']:.2f}%",
                        delta_color="normal" if index_data['change_percent'] >= 0 else "inverse"
                    )
    except Exception as e:
        st.warning(f"Unable to fetch current market data: {str(e)}")

    st.button("Get Started", on_click=navigate_to, args=('goals',), use_container_width=True)

# Goals page
def show_goals_page():
    st.title("Investment Goals")
    st.markdown("Select the investment objective that best aligns with your financial goals.")

    goals = {
        "Long-term Growth": "Focus on stocks with strong growth potential over 5+ years",
        "Dividend Income": "Prioritize stocks that pay regular dividends for income",
        "Value Investing": "Look for undervalued stocks with potential for appreciation",
        "Balanced Approach": "A mix of growth and income-generating stocks",
        "Short-term Trading": "Active trading for short-term gains (higher risk)"
    }

    col1, col2 = st.columns([3, 2])

    with col1:
        selected_goal = st.radio(
            "Select your primary investment goal:",
            list(goals.keys()),
            index=None
        )

        if selected_goal:
            st.session_state.user_profile['goal'] = selected_goal

    with col2:
        if selected_goal:
            st.info(goals[selected_goal])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            navigate_to('welcome')
    with col2:
        if selected_goal and st.button("Continue", use_container_width=True):
            navigate_to('risk_assessment')

# Risk assessment page
def show_risk_assessment():
    st.title("Risk Tolerance Assessment")
    st.markdown("Answer these questions to help us understand your risk tolerance level.")

    questions = [
        {
            "question": "How would you react if your investment lost 20% of its value in a month?",
            "options": [
                "Sell immediately to prevent further losses",
                "Sell a portion to reduce risk",
                "Hold and wait for recovery",
                "Buy more at the lower price"
            ],
            "scores": [1, 2, 3, 4]
        },
        {
            "question": "Which statement best describes your investment philosophy?",
            "options": [
                "Preserving capital is more important than growth",
                "Willing to accept moderate fluctuations for better returns",
                "Comfortable with volatility for potentially higher returns",
                "Aggressive growth is my priority, even with significant volatility"
            ],
            "scores": [1, 2, 3, 4]
        },
        {
            "question": "How much of your investable assets are you putting into stocks?",
            "options": [
                "Less than 25%",
                "25% to 50%",
                "50% to 75%",
                "More than 75%"
            ],
            "scores": [1, 2, 3, 4]
        },
        {
            "question": "How experienced are you with stock market investing?",
            "options": [
                "Beginner with little experience",
                "Some experience but still learning",
                "Moderately experienced investor",
                "Experienced investor with extensive knowledge"
            ],
            "scores": [1, 2, 3, 4]
        },
        {
            "question": "What is your typical response to market news and volatility?",
            "options": [
                "I closely follow the news and sometimes make reactive decisions",
                "I stay informed but rarely make changes based on short-term news",
                "I mostly ignore day-to-day news and focus on long-term trends",
                "I see market downturns as buying opportunities"
            ],
            "scores": [1, 3, 2, 4]
        }
    ]

    answers = {}
    for i, q in enumerate(questions):
        answers[i] = st.radio(
            q["question"],
            q["options"],
            index=None,
            key=f"q{i}"
        )

    # Check if all questions are answered
    all_answered = all(answer is not None for answer in answers.values())

    if all_answered:
        # Calculate risk score
        risk_score = sum(q["scores"][q["options"].index(answers[i])] for i, q in enumerate(questions))
        max_score = sum(max(q["scores"]) for q in questions)
        normalized_score = risk_score / max_score

        risk_level = ""
        if normalized_score < 0.4:
            risk_level = "Low"
        elif normalized_score < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        st.session_state.user_profile['risk_score'] = normalized_score
        st.session_state.user_profile['risk_tolerance'] = risk_level

        st.success(f"Your risk tolerance has been assessed as: **{risk_level}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            navigate_to('goals')
    with col2:
        if all_answered and st.button("Continue", use_container_width=True):
            navigate_to('financial_input')

# Financial input page
def show_financial_input():
    st.title("Financial Information")
    st.markdown("Provide details about your investment parameters.")

    col1, col2 = st.columns(2)

    with col1:
        investment_amount = st.number_input(
            "Initial Investment Amount ($)",
            min_value=0,
            max_value=10000000,
            value=10000 if st.session_state.user_profile.get('investment_amount') is None else st.session_state.user_profile['investment_amount'],
            step=1000,
            format="%d"
        )

        # Contribution frequency options
        contribution_frequency = st.selectbox(
            "Contribution Frequency",
            options=["None", "Weekly", "Monthly"],
            index=0 if st.session_state.user_profile.get('contribution_frequency') is None else 
                  ["None", "Weekly", "Monthly"].index(st.session_state.user_profile.get('contribution_frequency', "None"))
        )

        # Show contribution amount input if frequency is not None
        contribution_amount = 0
        if contribution_frequency != "None":
            contribution_amount = st.number_input(
                f"{contribution_frequency} Contribution Amount ($)",
                min_value=10,
                max_value=100000,
                value=500 if st.session_state.user_profile.get('contribution_amount') is None else st.session_state.user_profile['contribution_amount'],
                step=50,
                format="%d"
            )

        # Time horizon options mapped to months for calculation
        time_horizon_options = {
            "< 1 year": 6,
            "1-3 years": 24,
            "3-5 years": 48,
            "5-10 years": 84,
            "> 10 years": 120
        }

        time_horizon = st.selectbox(
            "Investment Time Horizon",
            options=list(time_horizon_options.keys()),
            index=2 if st.session_state.user_profile.get('time_horizon') is None else list(time_horizon_options.keys()).index(st.session_state.user_profile['time_horizon'])
        )

    with col2:
        desired_returns = st.slider(
            "Target Annual Returns (%)",
            min_value=1.0,
            max_value=30.0,
            value=10.0 if st.session_state.user_profile.get('desired_returns') is None else st.session_state.user_profile['desired_returns'],
            step=0.5
        )

        # Display the approximate value after the time period based on desired returns
        selected_months = time_horizon_options[time_horizon]
        years = selected_months / 12

        # Calculate final value based on initial investment and growth
        initial_growth = investment_amount * ((1 + (desired_returns/100)) ** years)

        # Calculate additional value from periodic contributions if any
        periodic_contribution_value = 0
        contribution_message = ""

        if contribution_frequency != "None" and contribution_amount > 0:
            # Calculate number of contributions based on frequency
            if contribution_frequency == "Monthly":
                num_contributions = selected_months
                contribution_period = "month"
            else:  # Weekly
                num_contributions = (selected_months * 30.44) / 7  # Approximate weeks
                contribution_period = "week"

            # Simple Future Value of Annuity formula: P * ((1 + r)^n - 1) / r
            # Where P is periodic payment, r is rate per period, n is number of periods
            rate_per_period = (desired_returns/100) / (12 if contribution_frequency == "Monthly" else 52)
            periodic_contribution_value = contribution_amount * (
                ((1 + rate_per_period) ** num_contributions - 1) / rate_per_period
            )

            contribution_message = f" with ${contribution_amount:,.2f} added every {contribution_period},"

        final_value = initial_growth + periodic_contribution_value

        st.info(f"Based on your inputs, your ${investment_amount:,.2f} initial investment{contribution_message} could grow to approximately **${final_value:,.2f}** over {years:.1f} years.")

    # Update session state
    st.session_state.user_profile['investment_amount'] = investment_amount
    st.session_state.user_profile['contribution_frequency'] = contribution_frequency
    if contribution_frequency != "None":
        st.session_state.user_profile['contribution_amount'] = contribution_amount
    st.session_state.user_profile['time_horizon'] = time_horizon
    st.session_state.user_profile['desired_returns'] = desired_returns

    # Display a summary of user inputs
    st.subheader("Your Investment Profile")

    # Create parameters and values lists
    parameters = ["Investment Goal", "Risk Tolerance", "Initial Investment Amount"]
    values = [
        st.session_state.user_profile['goal'],
        st.session_state.user_profile['risk_tolerance'],
        f"${investment_amount:,.2f}"
    ]

    # Add contribution information if applicable
    if contribution_frequency != "None":
        parameters.append(f"{contribution_frequency} Contribution")
        values.append(f"${contribution_amount:,.2f}")

    # Add remaining parameters
    parameters.extend(["Time Horizon", "Target Returns"])
    values.extend([time_horizon, f"{desired_returns:.1f}%"])

    # Create DataFrame
    profile = pd.DataFrame({
        "Parameter": parameters,
        "Value": values
    })
    st.table(profile)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            navigate_to('risk_assessment')
    with col2:
        if st.button("Get Recommendations", use_container_width=True):
            with st.spinner("Analyzing stocks and generating recommendations..."):
                try:
                    # Get stock recommendations based on user profile
                    recommendations = get_recommendations(
                        goal=st.session_state.user_profile['goal'],
                        risk_tolerance=st.session_state.user_profile['risk_tolerance'],
                        investment_amount=st.session_state.user_profile['investment_amount'],
                        time_horizon=st.session_state.user_profile['time_horizon'],
                        desired_returns=st.session_state.user_profile['desired_returns'],
                        risk_score=st.session_state.user_profile['risk_score']
                    )

                    if recommendations is not None and not recommendations.empty:
                        st.session_state.recommendations = recommendations
                        navigate_to('recommendations')
                    else:
                        st.error("Unable to generate recommendations. Please try adjusting your criteria.")
                except Exception as e:
                    st.error(f"An error occurred while generating recommendations: {str(e)}")

# Recommendations page
def show_recommendations():
    st.title("Your Personalized Stock Recommendations")

    if st.session_state.recommendations is None or st.session_state.recommendations.empty:
        st.error("No recommendations found. Please go back and adjust your criteria.")
        if st.button("Back to Financial Input"):
            navigate_to('financial_input')
        return

    # Display recommendations
    recommendations = st.session_state.recommendations

    # Summary statistics
    st.subheader("Recommendation Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Stocks", len(recommendations))
    with col2:
        avg_risk = recommendations['Risk Score'].mean()
        risk_level = "Low" if avg_risk < 0.4 else "Moderate" if avg_risk < 0.7 else "High"
        st.metric("Average Risk Level", risk_level)
    with col3:
        avg_return = recommendations['Expected Annual Return (%)'].mean()
        st.metric("Avg. Expected Return", f"{avg_return:.2f}%")

    # Detailed recommendations table
    st.subheader("Recommended Stocks")

    # Make ticker column clickable links to comparison page
    def make_clickable(ticker):
        # Check if ticker is already selected
        is_selected = ticker in st.session_state.selected_stocks
        if is_selected:
            return ticker + " ✓"
        return ticker

    # Display the recommendations with checkboxes
    selection_df = recommendations.copy()
    ticker_col = st.dataframe(
        selection_df,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="Stock ticker symbol",
                width="small",
            ),
            "Company": st.column_config.TextColumn(
                "Company Name",
                width="medium",
            ),
            "Sector": st.column_config.TextColumn(
                "Sector",
                width="medium",
            ),
            "Price ($)": st.column_config.NumberColumn(
                "Current Price ($)",
                format="%.2f",
                width="small",
            ),
            "P/E Ratio": st.column_config.NumberColumn(
                "P/E Ratio",
                format="%.2f",
                width="small",
            ),
            "Dividend Yield (%)": st.column_config.NumberColumn(
                "Dividend Yield (%)",
                format="%.2f",
                width="small",
            ),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Level",
                help="0-0.4: Low, 0.4-0.7: Moderate, >0.7: High",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Expected Annual Return (%)": st.column_config.NumberColumn(
                "Expected Return (%)",
                format="%.2f",
                width="small",
            ),
            "Recommendation Score": st.column_config.ProgressColumn(
                "Recommendation Score",
                help="Higher values indicate stronger recommendations",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Select": st.column_config.CheckboxColumn(
                "Compare",
                help="Select stocks to compare",
                default=False
            )
        },
        hide_index=True,
        use_container_width=True,
        column_order=["Ticker", "Company", "Sector", "Price ($)", "P/E Ratio", "Dividend Yield (%)", 
                      "Risk Score", "Expected Annual Return (%)", "Recommendation Score", "Select"],
    )

    # Visualization of recommendations
    st.subheader("Visual Comparison")

    # Risk vs Return scatter plot
    fig = px.scatter(
        recommendations,
        x="Risk Score",
        y="Expected Annual Return (%)",
        size="Recommendation Score",
        color="Sector",
        hover_name="Company",
        hover_data=["Ticker", "P/E Ratio", "Dividend Yield (%)"],  # Show ticker in hover instead of as text
        # Removed text="Ticker" to eliminate text over circles
        size_max=20,
        title="Risk vs. Expected Return by Sector"
    )
    # Removed textposition setting as we're not displaying text anymore
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Selected stocks for comparison
    selected = st.multiselect(
        "Select stocks to compare (max 4):",
        options=recommendations['Ticker'].tolist(),
        default=st.session_state.selected_stocks[:2] if st.session_state.selected_stocks else []
    )

    # Update selected stocks in session state
    st.session_state.selected_stocks = selected[:4]  # Limit to 4 stocks

    # If stocks are selected for comparison
    if st.session_state.selected_stocks:
        if st.button("Compare Selected Stocks", use_container_width=True):
            navigate_to('comparison')

    # Add to Portfolio section for authenticated users
    if st.session_state.authenticated:
        st.subheader("Add Recommended Stocks to Portfolio")
        st.markdown("Select a stock from your recommendations to add to your personal portfolio.")

        col1, col2, col3 = st.columns(3)
        with col1:
            add_ticker = st.selectbox(
                "Select Stock",
                options=recommendations['Ticker'].tolist(),
                format_func=lambda x: f"{x} - {recommendations[recommendations['Ticker'] == x]['Company Name'].iloc[0]}" 
                if 'Company Name' in recommendations.columns else x
            )

        with col2:
            add_shares = st.number_input("Number of Shares", min_value=1, step=1, key="add_shares")
            # Get current price of selected ticker
            current_price = 0.0
            if add_ticker:
                current_price = recommendations[recommendations['Ticker'] == add_ticker]['Current Price'].iloc[0] if 'Current Price' in recommendations.columns else 0.0

            add_price = st.number_input(
                "Purchase Price per Share ($)", 
                min_value=0.01, 
                value=float(current_price) if current_price > 0 else 1.0,
                step=0.01, 
                key="add_price"
            )

        with col3:
            add_date = st.date_input("Purchase Date", key="add_date")

            if st.button("Add to My Portfolio", key="add_to_portfolio"):
                if add_ticker and add_shares and add_price and add_date:
                    if add_to_portfolio(add_ticker, add_shares, add_price, add_date.strftime("%Y-%m-%d")):
                        st.success(f"Added {add_ticker} to your portfolio.")
                        st.rerun()
                else:
                    st.warning("Please fill in all fields.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Financial Input", use_container_width=True):
            navigate_to('financial_input')
    with col2:
        if st.button("View Personal Dashboard", use_container_width=True):
            navigate_to('dashboard')

    # Portfolio button for authenticated users or login prompt
    if st.session_state.authenticated:
        if st.button("View My Portfolio", use_container_width=True):
            navigate_to('portfolio')
    else:
        if st.button("Login to Track Portfolio", use_container_width=True):
            navigate_to('auth')

# Comparison page
def show_comparison():
    st.title("Stock Comparison")

    if not st.session_state.selected_stocks:
        st.warning("No stocks selected for comparison.")
        if st.button("Back to Recommendations"):
            navigate_to('recommendations')
        return

    # Get detailed data for selected stocks
    selected_tickers = st.session_state.selected_stocks

    with st.spinner("Fetching detailed data for selected stocks..."):
        try:
            # Fetch historical price data
            historical_data = fetch_stock_data(selected_tickers, period="1y")

            # Fetch fundamental data
            fundamentals = fetch_stock_fundamentals(selected_tickers)

            if historical_data is None or fundamentals is None:
                st.error("Unable to fetch stock data. Please try again.")
                if st.button("Back"):
                    navigate_to('recommendations')
                return

        except Exception as e:
            st.error(f"An error occurred while fetching stock data: {str(e)}")
            if st.button("Back"):
                navigate_to('recommendations')
            return

    # Display stock price chart
    st.subheader("Stock Price Performance (1 Year)")

    # Create the line chart for historical prices
    fig = go.Figure()

    # Track successfully plotted tickers
    plotted_tickers = []

    for ticker in selected_tickers:
        if ticker in historical_data and not historical_data[ticker].empty:
            try:
                # Normalize to percentage change from first day for better comparison
                prices = historical_data[ticker]
                first_price = prices.iloc[0]

                # Check if the first price is positive and a valid number
                if isinstance(first_price, (int, float)) and first_price > 0:
                    baseline = first_price
                    normalized = (prices / baseline - 1) * 100

                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode='lines',
                        name=ticker,
                        hovertemplate='%{x}<br>%{y:.2f}%<extra></extra>'
                    ))
                    plotted_tickers.append(ticker)
                else:
                    st.warning(f"Could not plot {ticker}: initial price is invalid")
            except Exception as e:
                st.warning(f"Could not plot {ticker}: {str(e)}")

    # Display a message if no stocks could be plotted
    if not plotted_tickers:
        st.warning("Could not plot any stock price data. Try selecting different stocks.")
    else:
        fig.update_layout(
            title=f"Percentage Change in Stock Prices (1 Year) - {', '.join(plotted_tickers)}",
            xaxis_title="Date",
            yaxis_title="Percentage Change (%)",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # Display key metrics comparison
    st.subheader("Key Metrics Comparison")

    if fundamentals is not None and not fundamentals.empty:
        # Transpose for better display
        display_metrics = fundamentals.transpose()
        st.dataframe(display_metrics, use_container_width=True)

        # Radar chart for visual comparison of key metrics
        metrics_for_radar = ['P/E Ratio', 'Dividend Yield (%)', 'ROE (%)', 'Debt to Equity', 'Current Ratio']
        available_metrics = [m for m in metrics_for_radar if m in fundamentals.columns]

        if available_metrics and len(available_metrics) >= 3:  # Need at least 3 metrics for a meaningful radar chart
            # Create radar chart
            fig = go.Figure()

            # Track which tickers we can plot
            plotted_radar_tickers = []

            for ticker in selected_tickers:
                if ticker in fundamentals.index:
                    # Check if we have enough valid data points for this ticker
                    values = fundamentals.loc[ticker, available_metrics].tolist()
                    if sum(1 for v in values if not pd.isna(v)) >= 3:  # Need at least 3 valid metrics
                        # For radar charts, we need to replace NaN with 0 or other value
                        values = [0 if pd.isna(v) else v for v in values]
                        # Close the loop for radar chart
                        values.append(values[0])

                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=available_metrics + [available_metrics[0]],  # Close the loop
                            fill='toself',
                            name=ticker
                        ))
                        plotted_radar_tickers.append(ticker)

            if plotted_radar_tickers:
                fig.update_layout(
                    title=f"Key Metrics Comparison - {', '.join(plotted_radar_tickers)}",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        ),
                    ),
                    showlegend=True,
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough valid metrics available for radar chart visualization. Try selecting different stocks.")

    # Pros and cons for each stock
    st.subheader("Pros and Cons Analysis")

    for ticker in selected_tickers:
        if ticker in st.session_state.recommendations['Ticker'].values:
            stock_data = st.session_state.recommendations[st.session_state.recommendations['Ticker'] == ticker].iloc[0]

            st.markdown(f"### {ticker}: {stock_data['Company']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Pros")
                pros = []

                # Generate pros based on metrics
                if 'Dividend Yield (%)' in stock_data and stock_data['Dividend Yield (%)'] > 2:
                    pros.append(f"Strong dividend yield of {stock_data['Dividend Yield (%)']:.2f}%")

                if 'P/E Ratio' in stock_data and stock_data['P/E Ratio'] < 20:
                    pros.append(f"Attractive P/E ratio of {stock_data['P/E Ratio']:.2f}")

                if 'ROE (%)' in fundamentals and ticker in fundamentals.index and fundamentals.loc[ticker, 'ROE (%)'] > 15:
                    pros.append(f"Strong return on equity of {fundamentals.loc[ticker, 'ROE (%)']:.2f}%")

                if 'Expected Annual Return (%)' in stock_data and stock_data['Expected Annual Return (%)'] > 10:
                    pros.append(f"High expected return of {stock_data['Expected Annual Return (%)']:.2f}%")

                if 'Risk Score' in stock_data:
                    if stock_data['Risk Score'] < 0.4 and st.session_state.user_profile['risk_tolerance'] == 'Low':
                        pros.append("Low risk profile matching your risk tolerance")
                    elif 0.4 <= stock_data['Risk Score'] <= 0.7 and st.session_state.user_profile['risk_tolerance'] == 'Moderate':
                        pros.append("Moderate risk profile matching your risk tolerance")
                    elif stock_data['Risk Score'] > 0.7 and st.session_state.user_profile['risk_tolerance'] == 'High':
                        pros.append("Growth potential with higher risk matching your tolerance")

                # If we don't have enough pros, add general ones
                if len(pros) < 3:
                    if st.session_state.user_profile['goal'] == 'Long-term Growth':
                        pros.append("Potential for long-term capital appreciation")
                    elif st.session_state.user_profile['goal'] == 'Dividend Income':
                        pros.append("Provides regular income through dividends")
                    elif st.session_state.user_profile['goal'] == 'Value Investing':
                        pros.append("Currently trading at a potentially undervalued price")

                for pro in pros:
                    st.markdown(f"- {pro}")

            with col2:
                st.markdown("#### Cons")
                cons = []

                # Generate cons based on metrics
                if 'Dividend Yield (%)' in stock_data and stock_data['Dividend Yield (%)'] < 1 and st.session_state.user_profile['goal'] == 'Dividend Income':
                    cons.append(f"Low dividend yield of {stock_data['Dividend Yield (%)']:.2f}%")

                if 'P/E Ratio' in stock_data and stock_data['P/E Ratio'] > 30:
                    cons.append(f"High P/E ratio of {stock_data['P/E Ratio']:.2f} may indicate overvaluation")

                if 'Debt to Equity' in fundamentals and ticker in fundamentals.index and fundamentals.loc[ticker, 'Debt to Equity'] > 1.5:
                    cons.append(f"High debt-to-equity ratio of {fundamentals.loc[ticker, 'Debt to Equity']:.2f}")

                if 'Risk Score' in stock_data:
                    if stock_data['Risk Score'] > 0.7 and st.session_state.user_profile['risk_tolerance'] == 'Low':
                        cons.append("Risk profile higher than your stated tolerance")
                    elif stock_data['Risk Score'] < 0.4 and st.session_state.user_profile['risk_tolerance'] == 'High':
                        cons.append("May provide lower returns than desired given your risk tolerance")

                # If we don't have enough cons, add general ones
                if len(cons) < 2:
                    cons.append("Past performance doesn't guarantee future results")
                    if 'Current Ratio' in fundamentals and ticker in fundamentals.index and fundamentals.loc[ticker, 'Current Ratio'] < 1.5:
                        cons.append(f"Current ratio of {fundamentals.loc[ticker, 'Current Ratio']:.2f} indicates potential liquidity concerns")
                    else:
                        cons.append("Market volatility could impact short-term performance")

                for con in cons:
                    st.markdown(f"- {con}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Recommendations", use_container_width=True):
            navigate_to('recommendations')
    with col2:
        if st.button("View Dashboard", use_container_width=True):
            navigate_to('dashboard')

# Dashboard page
def show_dashboard():
    st.title("Your Personalized Investment Dashboard")

    if st.session_state.recommendations is None or st.session_state.recommendations.empty:
        st.error("No recommendations found. Please go back and adjust your criteria.")
        if st.button("Back to Financial Input"):
            navigate_to('financial_input')
        return

    # User profile summary
    st.sidebar.subheader("Your Investment Profile")
    st.sidebar.info(f"""
    **Goal:** {st.session_state.user_profile['goal']}

    **Risk Tolerance:** {st.session_state.user_profile['risk_tolerance']}

    **Investment Amount:** ${st.session_state.user_profile['investment_amount']:,.2f}

    **Time Horizon:** {st.session_state.user_profile['time_horizon']}

    **Target Returns:** {st.session_state.user_profile['desired_returns']:.1f}%
    """)

    # Top recommendations overview
    st.subheader("Top Stock Recommendations")
    top_picks = st.session_state.recommendations.head(5)

    # Display metrics cards for top 5 recommendations
    cols = st.columns(len(top_picks))
    for i, (_, stock) in enumerate(top_picks.iterrows()):
        with cols[i]:
            st.metric(
                f"{stock['Ticker']} - {stock['Company']}",
                f"${stock['Price ($)']:.2f}",
                f"{stock['Expected Annual Return (%)']:.2f}%"
            )
            st.progress(float(stock['Recommendation Score']), text=f"Match Score: {stock['Recommendation Score']:.2f}")

    # Sector allocation recommendations
    st.subheader("Recommended Sector Allocation")

    # Calculate sector weights based on recommendation scores
    sector_allocation = st.session_state.recommendations.groupby('Sector')['Recommendation Score'].sum()
    total_score = sector_allocation.sum()
    sector_allocation = (sector_allocation / total_score * 100).sort_values(ascending=False)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Pie chart for sector allocation
        fig = px.pie(
            values=sector_allocation.values,
            names=sector_allocation.index,
            title="Recommended Sector Allocation",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Table with sector allocations
        allocation_df = pd.DataFrame({
            'Sector': sector_allocation.index,
            'Allocation (%)': sector_allocation.values.round(2)
        })
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)

    # Stock price prediction visualization
    st.subheader("Projected Value Growth")

    # Create investment projection based on user inputs
    time_horizon_map = {
        "< 1 year": 1,
        "1-3 years": 3,
        "3-5 years": 5,
        "5-10 years": 10,
        "> 10 years": 15
    }

    max_years = time_horizon_map[st.session_state.user_profile['time_horizon']]
    investment_amount = st.session_state.user_profile['investment_amount']

    # Check if periodic contributions are enabled
    has_contributions = False
    contribution_amount = 0
    contribution_frequency = "None"
    contributions_per_year = 0

    if 'contribution_frequency' in st.session_state.user_profile and st.session_state.user_profile['contribution_frequency'] != "None":
        has_contributions = True
        contribution_frequency = st.session_state.user_profile['contribution_frequency']
        contribution_amount = st.session_state.user_profile.get('contribution_amount', 0)

        # Calculate contributions per year based on frequency
        if contribution_frequency == "Monthly":
            contributions_per_year = 12
        elif contribution_frequency == "Weekly":
            contributions_per_year = 52

    # Generate three scenarios
    conservative_return = max(st.session_state.user_profile['desired_returns'] - 3, 1)
    expected_return = st.session_state.user_profile['desired_returns']
    optimistic_return = st.session_state.user_profile['desired_returns'] + 3

    years = list(range(max_years + 1))

    # Function to calculate growth with periodic contributions
    def calculate_growth_with_contributions(initial_amount, annual_rate, years_list, periodic_amount, periods_per_year):
        values = []
        for year in years_list:
            if year == 0:
                values.append(initial_amount)
                continue

            if periodic_amount == 0 or periods_per_year == 0:
                # Simple compound interest without contributions
                values.append(initial_amount * ((1 + annual_rate/100) ** year))
            else:
                # Calculate with periodic contributions
                # Initial investment growth
                initial_growth = initial_amount * ((1 + annual_rate/100) ** year)

                # Growth of periodic contributions
                rate_per_period = annual_rate / 100 / periods_per_year
                total_periods = periods_per_year * year

                # Future Value of Annuity formula
                if rate_per_period > 0:
                    periodic_growth = periodic_amount * (((1 + rate_per_period) ** total_periods - 1) / rate_per_period)
                else:
                    periodic_growth = periodic_amount * total_periods

                values.append(initial_growth + periodic_growth)
        return values

    # Calculate values for each scenario
    conservative_values = calculate_growth_with_contributions(
        investment_amount, conservative_return, years, 
        contribution_amount, contributions_per_year if has_contributions else 0
    )

    expected_values = calculate_growth_with_contributions(
        investment_amount, expected_return, years, 
        contribution_amount, contributions_per_year if has_contributions else 0
    )

    optimistic_values = calculate_growth_with_contributions(
        investment_amount, optimistic_return, years, 
        contribution_amount, contributions_per_year if has_contributions else 0
    )

    # Create projection chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years,
        y=conservative_values,
        mode='lines',
        name=f'Conservative ({conservative_return:.1f}%)',
        line=dict(dash='dash', color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=expected_values,
        mode='lines',
        name=f'Expected ({expected_return:.1f}%)',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=years,
        y=optimistic_values,
        mode='lines',
        name=f'Optimistic ({optimistic_return:.1f}%)',
        line=dict(dash='dot', color='red')
    ))

    # Create title that includes contribution information if applicable
    title_text = f"Projected Investment Growth (${investment_amount:,.0f} initial investment"
    if has_contributions:
        title_text += f" with ${contribution_amount:,.0f} {contribution_frequency.lower()} contributions"
    title_text += ")"

    fig.update_layout(
        title=title_text,
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=500
    )

    # Format y-axis as currency
    fig.update_layout(yaxis=dict(tickprefix="$", tickformat=","))

    st.plotly_chart(fig, use_container_width=True)

    # Next steps guidance
    st.subheader("Next Steps")

    st.markdown("""
    ### How to Act on These Recommendations:

    1. **Research Further:** Before investing, research each company more thoroughly, including recent news and financial reports.

    2. **Consider Diversification:** Spread your investment across multiple stocks and sectors to reduce risk.

    3. **Consult a Financial Advisor:** For large investments, consider consulting with a professional financial advisor.

    4. **Start Small:** If you're new to investing, consider starting with a smaller amount to get comfortable with market fluctuations.

    5. **Regular Review:** Set a schedule to review your investments and adjust as needed based on performance and changing goals.
    """)

    # Button to return to recommendations
    if st.button("Back to Recommendations", use_container_width=True):
        navigate_to('recommendations')

# Portfolio page
def show_portfolio_page():
    st.title("My Investment Portfolio")

    # Display the portfolio
    show_portfolio()

    # Add a stock to portfolio section
    if st.session_state.authenticated:
        st.subheader("Add Stock to Portfolio")

        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Stock Ticker Symbol", key="portfolio_ticker")
            shares = st.number_input("Number of Shares", min_value=1, step=1, key="portfolio_shares")

        with col2:
            purchase_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, step=0.01, key="portfolio_price")
            purchase_date = st.date_input("Purchase Date", key="portfolio_date")

        if st.button("Add to Portfolio", key="add_portfolio_button"):
            if ticker and shares and purchase_price and purchase_date:
                if add_to_portfolio(ticker, shares, purchase_price, purchase_date.strftime("%Y-%m-%d")):
                    st.success(f"Added {ticker} to your portfolio.")
                    st.rerun()
            else:
                st.warning("Please fill in all fields.")

    # Button to return to dashboard
    if st.button("Back to Dashboard", use_container_width=True):
        navigate_to('dashboard')

# Authentication page
def show_auth_page():
    st.title("User Authentication")

    # Display authentication UI from auth.py
    if show_auth_ui():
        # If successfully authenticated, navigate to dashboard
        navigate_to('dashboard')

    # Option to continue as guest
    st.divider()
    st.markdown("**Or continue without logging in**")
    if st.button("Continue as Guest", use_container_width=True):
        navigate_to('welcome')

# Main app logic
def main():
    # Sidebar content
    with st.sidebar:

        # Authentication buttons
        if st.session_state.authenticated:
            st.write(f"Logged in as: **{st.session_state.user_email}**")
            if st.button("View Portfolio"):
                navigate_to('portfolio')
            if st.button("Logout"):
                logout()
                navigate_to('welcome')
        else:
            st.write("Not logged in")
            if st.button("Login / Register"):
                navigate_to('auth')

    # Display the appropriate page based on session state
    if st.session_state.page == 'welcome':
        show_welcome_page()
    elif st.session_state.page == 'goals':
        show_goals_page()
    elif st.session_state.page == 'risk_assessment':
        show_risk_assessment()
    elif st.session_state.page == 'financial_input':
        show_financial_input()
    elif st.session_state.page == 'recommendations':
        show_recommendations()
    elif st.session_state.page == 'comparison':
        show_comparison()
    elif st.session_state.page == 'dashboard':
        show_dashboard()
    elif st.session_state.page == 'auth':
        show_auth_page()
    elif st.session_state.page == 'portfolio':
        show_portfolio_page()
    else:
        show_welcome_page()

if __name__ == "__main__":
    main()