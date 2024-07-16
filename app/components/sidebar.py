
import streamlit as st

def sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Shot Prediction"])

