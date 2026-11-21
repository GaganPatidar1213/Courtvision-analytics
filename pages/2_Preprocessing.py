import streamlit as st

from utils.dashboard import preprocessing

st.set_page_config(page_title="COURTSIDE / LAB", page_icon="🏀", layout="wide")
preprocessing()
