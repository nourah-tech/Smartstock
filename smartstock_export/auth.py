import streamlit as st
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
def get_supabase_client() -> Client:
    """
    Get authenticated Supabase client using environment variables.
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("Supabase credentials not found in environment variables.")
        return None
    
    return create_client(supabase_url, supabase_key)

def initialize_auth():
    """
    Initialize authentication-related session state variables.
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []

def signup(email: str, password: str, name: str) -> bool:
    """
    Register a new user with Supabase and automatically log them in.
    
    Parameters:
    - email: User's email
    - password: User's password
    - name: User's full name
    
    Returns:
    - bool: Success status
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Sign up the user
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "name": name
                },
                # Auto-confirm email, no email verification needed
                "email_confirm": True
            }
        })
        
        if response.user:
            # Since we're auto-confirming, we can log the user in right away
            st.success("Registration successful!")
            
            # Automatically log the user in
            st.session_state.authenticated = True
            st.session_state.user_id = response.user.id
            st.session_state.user_email = response.user.email
            
            # Initialize an empty portfolio for the new user
            st.session_state.portfolio = []
            
            return True
        return False
    except Exception as e:
        st.error(f"Error during registration: {str(e)}")
        return False

def login(email: str, password: str) -> bool:
    """
    Log in an existing user with Supabase.
    
    Parameters:
    - email: User's email
    - password: User's password
    
    Returns:
    - bool: Success status
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Log in the user
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            st.session_state.authenticated = True
            st.session_state.user_id = response.user.id
            st.session_state.user_email = response.user.email
            
            # Fetch user portfolio
            fetch_user_portfolio()
            return True
        return False
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False

def logout():
    """
    Log out the current user.
    """
    try:
        supabase = get_supabase_client()
        if supabase:
            supabase.auth.sign_out()
        
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.portfolio = []
    except Exception as e:
        st.error(f"Error during logout: {str(e)}")

def fetch_user_portfolio():
    """
    Fetch the user's portfolio from Supabase.
    """
    if not st.session_state.authenticated or not st.session_state.user_id:
        return
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return
            
        # Fetch portfolio items for the current user
        response = supabase.table("portfolios").select("*").eq("user_id", st.session_state.user_id).execute()
        
        if response.data:
            st.session_state.portfolio = response.data
        else:
            st.session_state.portfolio = []
    except Exception as e:
        st.error(f"Error fetching portfolio: {str(e)}")

def add_to_portfolio(ticker, shares, purchase_price, purchase_date):
    """
    Add a stock to the user's portfolio.
    
    Parameters:
    - ticker: Stock ticker symbol
    - shares: Number of shares purchased
    - purchase_price: Price per share at purchase
    - purchase_date: Date of purchase
    
    Returns:
    - bool: Success status
    """
    if not st.session_state.authenticated or not st.session_state.user_id:
        st.warning("Please log in to save stocks to your portfolio.")
        return False
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Add the stock to the portfolio
        portfolio_item = {
            "user_id": st.session_state.user_id,
            "ticker": ticker,
            "shares": shares,
            "purchase_price": purchase_price,
            "purchase_date": purchase_date
        }
        
        response = supabase.table("portfolios").insert(portfolio_item).execute()
        
        if response.data:
            # Refresh the portfolio
            fetch_user_portfolio()
            return True
        return False
    except Exception as e:
        st.error(f"Error adding to portfolio: {str(e)}")
        return False

def remove_from_portfolio(portfolio_id):
    """
    Remove a stock from the user's portfolio.
    
    Parameters:
    - portfolio_id: ID of the portfolio item to remove
    
    Returns:
    - bool: Success status
    """
    if not st.session_state.authenticated or not st.session_state.user_id:
        return False
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
            
        # Remove the stock from the portfolio
        response = supabase.table("portfolios").delete().eq("id", portfolio_id).eq("user_id", st.session_state.user_id).execute()
        
        if response.data:
            # Refresh the portfolio
            fetch_user_portfolio()
            return True
        return False
    except Exception as e:
        st.error(f"Error removing from portfolio: {str(e)}")
        return False

def show_auth_ui():
    """
    Display the authentication UI (login/signup forms).
    
    Returns:
    - bool: True if user is authenticated, False otherwise
    """
    if st.session_state.authenticated:
        return True
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                if login(email, password):
                    st.rerun()
    
    with tab2:
        st.subheader("Sign Up")
        name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
        
        if st.button("Sign Up", key="signup_button"):
            if not name or not email or not password or not password_confirm:
                st.warning("Please fill in all fields.")
            elif password != password_confirm:
                st.warning("Passwords do not match.")
            else:
                if signup(email, password, name):
                    st.success("Sign up successful! You are now logged in.")
                    st.rerun()
    
    return False

def show_portfolio():
    """
    Display the user's portfolio.
    """
    if not st.session_state.authenticated:
        st.warning("Please log in to view your portfolio.")
        return
    
    st.subheader("Your Portfolio")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks to track your investments.")
        return
    
    # Display portfolio in a table
    portfolio_data = []
    for item in st.session_state.portfolio:
        # Get current price using existing stock data functions
        try:
            from data_fetcher import fetch_stock_data
            current_data = fetch_stock_data([item["ticker"]], period="1d")
            current_price = current_data[item["ticker"]].iloc[-1] if item["ticker"] in current_data else item["purchase_price"]
            
            # Calculate current value and profit/loss
            current_value = item["shares"] * current_price
            initial_value = item["shares"] * item["purchase_price"]
            profit_loss = current_value - initial_value
            profit_loss_percent = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
            
            portfolio_data.append({
                "Ticker": item["ticker"],
                "Shares": item["shares"],
                "Purchase Price": f"${item['purchase_price']:.2f}",
                "Current Price": f"${current_price:.2f}",
                "Current Value": f"${current_value:.2f}",
                "Profit/Loss": f"${profit_loss:.2f} ({profit_loss_percent:.2f}%)",
                "Purchase Date": item["purchase_date"],
                "Action": item["id"]  # Store ID for delete action
            })
        except Exception as e:
            st.error(f"Error processing portfolio item {item['ticker']}: {str(e)}")
    
    if portfolio_data:
        # Convert to dataframe for display
        import pandas as pd
        df = pd.DataFrame(portfolio_data)
        
        # Show the table
        for i, row in df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 3, 1])
            
            with col1:
                st.write(f"**{row['Ticker']}**")
                st.write(f"{row['Shares']} shares")
            
            with col2:
                st.write("Purchase:")
                st.write(f"{row['Purchase Price']}")
                st.write(f"{row['Purchase Date']}")
            
            with col3:
                st.write("Current:")
                st.write(f"{row['Current Price']}")
            
            with col4:
                st.write("Value:")
                st.write(f"{row['Current Value']}")
                st.write(f"{row['Profit/Loss']}")
            
            with col5:
                if st.button("Remove", key=f"remove_{row['Action']}"):
                    if remove_from_portfolio(row['Action']):
                        st.success(f"Removed {row['Ticker']} from portfolio.")
                        st.rerun()
            
            st.divider()