import streamlit as st

from utils.dashboard import home

st.set_page_config(page_title="COURTSIDE / LAB", page_icon="🏀", layout="wide")
home()
