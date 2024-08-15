
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from nba_api.stats.static import players, teams
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import calculate_efficiency, create_mae_table, save_mae_table, load_mae_table, get_seasons_range

@st.cache_data
def preload_mae_tables(entity_name, season):
    """Preload MAE tables for all teams to speed up future calculations."""
    mae_df_all_path = f'data/shot_chart_data/{entity_name}_mae_table_all_{season}.csv'
    mae_df_all = load_mae_table(mae_df_all_path)
    if mae_df_all is None:
        mae_df_all = create_mae_table(entity_name, season, [team['full_name'] for team in teams.get_teams()])
        save_mae_table(mae_df_all, mae_df_all_path)
    return mae_df_all

@st.cache_data
def create_and_save_mae_table_specific(entity_name, season, opponent_name):
    """Create and save the MAE table for a specific opponent."""
    mae_df_specific_path = f'data/shot_chart_data/{entity_name}_mae_table_specific_{season}.csv'
    mae_df_specific = load_mae_table(mae_df_specific_path)
    
    if mae_df_specific is None or opponent_name not in mae_df_specific['Opponent Team'].values:
        mae_df_specific = create_mae_table(entity_name, season, [opponent_name])
        save_mae_table(mae_df_specific, mae_df_specific_path)
    
    return mae_df_specific

@st.cache_data
def create_and_save_mae_table_all(entity_name, season):
    """Load the preloaded MAE table for all teams."""
    mae_df_all_path = f'data/shot_chart_data/{entity_name}_mae_table_all_{season}.csv'
    mae_df_all = load_mae_table(mae_df_all_path)
    if mae_df_all is None:
        mae_df_all = preload_mae_tables(entity_name, season)
    return mae_df_all

@st.cache_data
def get_teams_list():
    """Get the list of NBA teams."""
    return [team['full_name'] for team in teams.get_teams()]

@st.cache_data
def get_players_list():
    """Get the list of NBA players."""
    return [player['full_name'] for player in players.get_players()]

def main():
    st.title("NBA Shot Analysis")
    
    analysis_type = st.selectbox("Select analysis type", options=["offensive", "defensive", "both"])
    
    entity_type = st.selectbox("Analyze a Team or Player?", options=["team", "player"])
    
    if entity_type == "team":
        entity_name = st.selectbox("Select a Team", options=get_teams_list())
    else:
        entity_name = st.selectbox("Select a Player", options=get_players_list())
    
    season = st.selectbox("Select the season", options=["2023-24", "2022-23", "2021-22", "2020-21"])
    
    opponent_type = st.selectbox("Compare against all teams or a specific team?", options=["all", "specific"])
    
    opponent_name = None
    if opponent_type == "specific":
        opponent_name = st.selectbox("Select an Opponent Team", options=get_teams_list())
    
    if st.button("Run Analysis"):
        # Preload MAE tables for all teams
        mae_df_all = preload_mae_tables(entity_name, season)
        
        # Fetch and display offensive data
        shots = fetch_shots_data(entity_name, entity_type == 'team', season, opponent_name)
        st.write("Shot Data")
        st.dataframe(shots.head())
        
        efficiency = calculate_efficiency(shots)
        st.write(f"Offensive Efficiency for {entity_name}:")
        st.dataframe(efficiency)
        
        # Plot shot chart
        fig = plot_shot_chart_hexbin(shots, f'{entity_name} Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
        st.pyplot(fig)
        
        if opponent_type == 'specific':
            # MAE calculation and saving for specific team
            mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
            st.write(f"MAE Table for {entity_name} against {opponent_name}:")
            st.dataframe(mae_df_specific)
        else:
            # MAE calculation and loading for all teams
            st.write(f"MAE Table for {entity_name} against all teams:")
            st.dataframe(mae_df_all)
        
        min_season, max_season = get_seasons_range(mae_df_all)
        st.write(f"MAE Table available for seasons from {min_season} to {max_season}.")
    
        # If the analysis type is "both", also perform defensive analysis here
        if analysis_type == 'both':
            # Fetch and display defensive data for the specified team
            defensive_shots = fetch_defensive_shots_data(entity_name, True, season, opponent_name)
            defensive_efficiency = calculate_efficiency(defensive_shots)
            st.write(f"Defensive Efficiency for {entity_name}:")
            st.dataframe(defensive_efficiency)
            
            # Plot defensive shot chart
            fig = plot_shot_chart_hexbin(defensive_shots, f'{entity_name} Defensive Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
            st.pyplot(fig)
            
            if opponent_type == 'specific':
                # MAE calculation for defensive analysis against the specific opponent
                mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
                st.write(f"Defensive MAE Table for {entity_name} against {opponent_name}:")
                st.dataframe(mae_df_specific)

if __name__ == "__main__":
    main()
