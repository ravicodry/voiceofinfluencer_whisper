# src/storage_utils.py
import json
import os
import streamlit as st
from datetime import datetime

# Define the storage directory
STORAGE_DIR = "data"
STORAGE_FILE = os.path.join(STORAGE_DIR, "product_reviews.json")

def ensure_storage_exists():
    """Creates storage directory and file if they don't exist."""
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
    if not os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'w') as f:
            json.dump([], f)

def save_segment_product_analysis(analyzed_segments):
    """Saves the analyzed segments to a JSON file."""
    ensure_storage_exists()
    
    # Read existing data
    try:
        with open(STORAGE_FILE, 'r') as f:
            existing_data = json.load(f)
    except json.JSONDecodeError:
        existing_data = []
    
    # Add timestamp to each segment
    for segment in analyzed_segments:
        segment['timestamp'] = datetime.now().isoformat()
    
    # Append new data
    existing_data.extend(analyzed_segments)
    
    # Save back to file
    with open(STORAGE_FILE, 'w') as f:
        json.dump(existing_data, f, indent=2)

def load_segment_product_analysis():
    """Loads the analyzed segments from the JSON file."""
    ensure_storage_exists()
    
    try:
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def clear_all_data():
    """Clears all stored data."""
    ensure_storage_exists()
    with open(STORAGE_FILE, 'w') as f:
        json.dump([], f)