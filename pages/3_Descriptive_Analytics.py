import streamlit as st

from utils.dashboard import descriptive

st.set_page_config(page_title="COURTSIDE / LAB", page_icon="🏀", layout="wide")
descriptive()
