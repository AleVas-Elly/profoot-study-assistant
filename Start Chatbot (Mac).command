#!/bin/bash
echo "Starting Profoot..."

# Navigate to the folder where this script is located
cd "$(dirname "$0")" || { echo "Failed to navigate to script directory"; sleep 5; exit 1; }

# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate virtual environment. Did you run the setup script first?"; sleep 5; exit 1; }

# Run Streamlit using explicit python module to ensure paths resolve correctly
python -m streamlit run app.py || { echo "Streamlit closed or crashed."; }

echo "Press any key to close this window..."
read -n 1 -s
