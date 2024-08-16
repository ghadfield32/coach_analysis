
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nba_api.stats.static import players, teams
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data, fetch_shots_for_multiple_players
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import calculate_efficiency, create_mae_table, save_mae_table, load_mae_table, get_seasons_range, calculate_compatibility_between_players

def preload_mae_tables(entity_name, season):
    """Preload MAE tables for all teams to speed up future calculations."""
    mae_df_all_path = f'data/shot_chart_data/{entity_name}_mae_table_all_{season}.csv'
    mae_df_all = load_mae_table(mae_df_all_path)
    if mae_df_all is None:
        mae_df_all = create_mae_table(entity_name, season, [team['full_name'] for team in teams.get_teams()])
        save_mae_table(mae_df_all, mae_df_all_path)
    return mae_df_all

def create_and_save_mae_table_specific(entity_name, season, opponent_name):
    """Create and save the MAE table for a specific opponent."""
    mae_df_specific_path = f'data/shot_chart_data/{entity_name}_mae_table_specific_{season}.csv'
    mae_df_specific = load_mae_table(mae_df_specific_path)
    
    print(f"Debug: Opponent name before calling create_mae_table: {opponent_name}")
    
    if mae_df_specific is None or opponent_name not in mae_df_specific['Opponent Team'].values:
        mae_df_specific = create_mae_table(entity_name, season, [opponent_name])
        
        print(f"Debug: MAE table head after creation for opponent {opponent_name}:\n{mae_df_specific.head()}")
        
        save_mae_table(mae_df_specific, mae_df_specific_path)
    
    return mae_df_specific

def create_and_save_mae_table_all(entity_name, season):
    """Load the preloaded MAE table for all teams."""
    mae_df_all_path = f'data/shot_chart_data/{entity_name}_mae_table_all_{season}.csv'
    mae_df_all = load_mae_table(mae_df_all_path)
    if mae_df_all is None:
        mae_df_all = preload_mae_tables(entity_name, season)
    return mae_df_all

def run_scenario(entity_name, entity_type, season, opponent_name=None, analysis_type="offensive", compare_players=False, player_names=None, court_areas=None):
    """Run a scenario for a given entity (team or player) against a specific opponent or all teams."""
    if compare_players and player_names:
        st.write("Comparing multiple players...")
        player_shots = fetch_shots_for_multiple_players(player_names, season, court_areas, opponent_name, debug=False)
        
        st.write(f"Player shots data: {player_shots.keys()}")
        for player, shots in player_shots.items():
            st.write(f"{player}: {len(shots['shots'])} shots recorded")
        
        compatibility_df = calculate_compatibility_between_players(player_shots)
        
        st.write("MAE DataFrame after calculation:")
        st.write(compatibility_df)
    else:
        opponent_text = "All Teams" if not opponent_name or (isinstance(opponent_name, str) and opponent_name.lower() == "all") else opponent_name
        st.write(f"Running scenario for {entity_name} ({'Team' if entity_type == 'team' else 'Player'}) vs {opponent_text}")
        
        # Ensure only teams are analyzed for defensive scenarios
        if entity_type == 'player' and analysis_type != 'offensive':
            st.error("Defensive analysis is only applicable for teams.")
            return
        
        # Load the appropriate MAE tables based on whether it's a team or player
        if entity_type == 'team':
            mae_df_all = preload_mae_tables(entity_name, season)
        else:
            mae_df_all = None  # MAE might not be applicable for individual players in this context
        
        # Fetch and display offensive data
        shots = fetch_shots_data(entity_name, entity_type == 'team', season, opponent_name)
        st.write(f"Shots DataFrame head:")
        st.write(shots.head())
        
        efficiency = calculate_efficiency(shots)
        st.write(f"Offensive Efficiency for {entity_name}:")
        st.write(efficiency)
        
        # Plot shot chart and display it
        fig = plot_shot_chart_hexbin(shots, f'{entity_name} Shot Chart', opponent=opponent_text)
        st.pyplot(fig)
        
        if opponent_name and isinstance(opponent_name, str) and opponent_name.lower() != "all" and entity_type == 'team':
            # MAE calculation and saving for specific team (only relevant if we're dealing with teams)
            mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
            
            # Filter the table to include only the specific opponent
            mae_df_specific = mae_df_specific[mae_df_specific['Opponent Team'] == opponent_name]
            
            st.write(f"Defensive MAE Table for {entity_name} against {opponent_name}:")
            st.write(mae_df_specific)
        elif entity_type == 'team':
            # MAE calculation and loading for all teams (only relevant for teams)
            st.write(f"MAE Table for {entity_name} against all teams:")
            st.write(mae_df_all)
        
            min_season, max_season = get_seasons_range(mae_df_all)
            st.write(f"MAE Table available for seasons from {min_season} to {max_season}.")
        
        # Perform defensive analysis only for teams
        if analysis_type == 'both' and entity_type == 'team':
            # Fetch and display defensive data for the specified team
            defensive_shots = fetch_defensive_shots_data(entity_name, entity_type == 'team', season, opponent_name)
            defensive_efficiency = calculate_efficiency(defensive_shots)
            st.write(f"Defensive Efficiency for {entity_name}:")
            st.write(defensive_efficiency)
            
            # Plot defensive shot chart and display it
            fig = plot_shot_chart_hexbin(defensive_shots, f'{entity_name} Defensive Shot Chart', opponent=opponent_text)
            st.pyplot(fig)
            
            if opponent_name and isinstance(opponent_name, str) and opponent_name.lower() != "all" and entity_type == 'team':
                # MAE calculation for defensive analysis against the specific opponent (only for teams)
                mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
                
                # Filter the table to include only the specific opponent
                mae_df_specific = mae_df_specific[mae_df_specific['Opponent Team'] == opponent_name]
                
                st.write(f"Defensive MAE Table for {entity_name} against {opponent_name}:")
                st.write(mae_df_specific)


def main():
    season = "2023-24"  # You can modify this to ask the user for input if needed

    # Specify the court areas of interest, or use 'all' to compare across all areas
    court_areas = 'all'  # Or set to 'all' to include all areas

    # Compare three players in specific areas of the court against a specific opponent or all teams
    player_names = ["Luka Doncic", "Stephen Curry", "Kevin Durant"]
    run_scenario(None, "player", season, opponent_name="all", compare_players=True, player_names=player_names, court_areas=court_areas)

    # Other scenarios remain unchanged
    run_scenario("Boston Celtics", "team", season, "Dallas Mavericks", analysis_type="both")
    run_scenario("Boston Celtics", "team", season, None, analysis_type="both")
    run_scenario("Luka Doncic", "player", season, "Boston Celtics", analysis_type="offensive")

if __name__ == "__main__":
    main()


