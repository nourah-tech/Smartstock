#!/usr/bin/env bash
# Exit on error
set -o errexit

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create Streamlit config for the hosting platform
# This will use PORT environment variable if available (for Render)
# Otherwise, it will default to port 5000 (for local/Replit)
echo "[server]
headless = true
address = \"0.0.0.0\"
port = ${PORT:-5000}" > .streamlit/config.toml

# Handle requirements files flexibly
if [ -f requirements.txt ]; then
  # If requirements.txt exists, use it
  pip install -r requirements.txt
elif [ -f render-requirements.txt ]; then
  # If only render-requirements.txt exists, use it
  pip install -r render-requirements.txt
else
  echo "No requirements file found. Please create requirements.txt or render-requirements.txt"
  exit 1
fi
