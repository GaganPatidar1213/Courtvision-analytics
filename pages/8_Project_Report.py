import streamlit as st

from utils.dashboard import report_page

st.set_page_config(page_title="COURTSIDE / LAB", page_icon="🏀", layout="wide")
report_page()
