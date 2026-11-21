import streamlit as st

from utils.dashboard import tree_page

st.set_page_config(page_title="COURTSIDE / LAB", page_icon="🏀", layout="wide")
tree_page()
