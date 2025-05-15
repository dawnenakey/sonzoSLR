import os
import sys
from pathlib import Path

# Add the application directory to the Python path
application_path = Path(__file__).parent
sys.path.insert(0, str(application_path))

# Import the Streamlit app
from src.ui.oak_upload_app import app

# Create WSGI application
application = app 