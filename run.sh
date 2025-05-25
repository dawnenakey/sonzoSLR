#!/bin/bash

# Get your local IP address
IP_ADDRESS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}')

echo "Starting Streamlit app..."
echo "Access the app at: http://$IP_ADDRESS:8501"

# Run Streamlit with the following configurations:
# - server.address: 0.0.0.0 (allows external access)
# - server.port: 8501 (default Streamlit port)
# - server.headless: true (runs without opening browser)
streamlit run src/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true 