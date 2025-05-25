# src/ip_utils.py

ALLOWED_IPS = ["127.0.0.1", "localhost", "::1", "0.0.0.0"]

def get_ip():
    import streamlit as st
    return st.experimental_get_query_params().get('ip', [''])[0]

def is_ip_allowed(ip):
    return ip in ALLOWED_IPS
