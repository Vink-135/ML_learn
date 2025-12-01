#!/usr/bin/env python3
"""
Simple script to run the Pix2Pix Streamlit application
"""
import subprocess
import sys
import os

def main():
    # Change to the pix2pix directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run streamlit with the UI module
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
