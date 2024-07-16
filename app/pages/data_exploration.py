

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data.data_loader import fetch_shots_data
from visualization.plot_utils import plot_court

def run():
    st.header("Data Exploration")
    
    team_name = st.selectbox("Select Team", ["Boston Celtics", "Los Angeles Lakers", "Golden State Warriors"])
    season = st.selectbox("Select Season", ["2023-24", "2022-23", "2021-22"])
    
    shots = fetch_shots_data(team_name, True, season)
    
    st.subheader("Raw Data")
    st.dataframe(shots.head())
    
    st.subheader("Shot Distribution")
    fig, ax = plt.subplots(figsize=(12, 11))
    plot_court(ax)
    ax.scatter(shots['LOC_X'], shots['LOC_Y'], alpha=0.5)
    st.pyplot(fig)
    
    st.subheader("Shot Success Rate by Distance")
    distance_success = shots.groupby('SHOT_DISTANCE').agg({
        'SHOT_MADE_FLAG': ['count', 'mean']
    })
    distance_success.columns = ['Total Shots', 'Success Rate']
    distance_success = distance_success[distance_success['Total Shots'] > 10]
    st.line_chart(distance_success['Success Rate'])
    
    st.subheader("Top Scorers")
    top_scorers = shots.groupby('PLAYER_NAME').agg({
        'SHOT_MADE_FLAG': ['count', 'sum']
    })
    top_scorers.columns = ['Total Shots', 'Made Shots']
    top_scorers['Points'] = top_scorers['Made Shots'] * 2  # Simplification, not accounting for 3-pointers
    top_scorers = top_scorers.sort_values('Points', ascending=False).head(10)
    st.bar_chart(top_scorers['Points'])
