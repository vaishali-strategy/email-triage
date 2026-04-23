#!/usr/bin/env python3
"""
Main entry point for Hugging Face Spaces
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the Streamlit app
if __name__ == "__main__":
    import subprocess
    import streamlit.web.cli as stcli
    
    # Run Streamlit with the correct arguments
    sys.argv = ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())
