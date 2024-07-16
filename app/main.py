
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from pages import data_exploration, model_training, shot_prediction
from components.sidebar import sidebar

st.set_page_config(page_title="NBA Shot Predictor", page_icon="🏀", layout="wide")

def main():
    st.title("NBA Shot Predictor 🏀")
    
    page = sidebar()
    
    if page == "Data Exploration":
        data_exploration.run()
    elif page == "Model Training":
        model_training.run()
    elif page == "Shot Prediction":
        shot_prediction.run()

if __name__ == "__main__":
    main()
