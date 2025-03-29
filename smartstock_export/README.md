# AI Stock Advisor

An AI-powered stock recommendation platform that helps investors discover and analyze potential investment opportunities with intelligent, data-driven insights.

## Features

- Goal-based stock recommendation
- Risk tolerance assessment
- Financial insights and analysis
- Portfolio tracking
- Personalized dashboard with recommendations

## Technologies Used

- Python
- Streamlit for interactive web interface
- Yahoo Finance API for real-time stock data
- Machine learning for stock analysis
- Supabase for authentication and database

## Deployment Options

### Deploy on Replit

1. Your app is already configured to run on Replit
2. Make sure your Replit Secrets contain:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase project API key
3. Use the "Run" button to start your application

### Deploy on Render.com

1. Create an account on [Render.com](https://render.com/)
2. Connect your GitHub repository
3. Create a new Web Service and select "Use render.yaml" for configuration
4. Add the following environment secrets in the Render dashboard:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase project API key
5. Deploy your application

## Switching Between Platforms

The project is set up to work seamlessly on both Replit and Render.com:

- **When working on Replit**: Make your changes in the Replit editor. The application will run on port 5000.
- **For Render deployment**: Push your changes to GitHub. Render will automatically deploy your app if you've set up automatic deployments.

## Local Development

1. Clone the repository
2. Create a `.env` file with the following variables:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```
3. Install dependencies: `pip install -r render-requirements.txt`
4. Run the application: `streamlit run app.py`
5. Open your browser to `http://localhost:5000`

## Project Structure

- `app.py`: Main Streamlit application
- `auth.py`: Authentication functionality using Supabase
- `data_fetcher.py`: Functions to retrieve stock data
- `recommendation_engine.py`: AI-powered stock recommendation system
- `stock_analyzer.py`: Stock analysis and metrics calculation
- `utils.py`: Utility functions for the application
- `.streamlit/config.toml`: Streamlit server configuration
- `build.sh`: Build script for deployment
- `render.yaml`: Render.com configuration