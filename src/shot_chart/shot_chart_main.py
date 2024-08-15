
import os
import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.static import players, teams
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import calculate_efficiency, create_mae_table, save_mae_table, load_mae_table, get_seasons_range

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

def main():
    # Step 1: Select analysis type (offensive, defensive, or both)
    analysis_type = input("Select analysis type (offensive/defensive/both): ").strip().lower()
    if analysis_type not in ['offensive', 'defensive', 'both']:
        print("Invalid selection. Please enter 'offensive', 'defensive', or 'both'.")
        return
    
    entity_type = input("Analyze a Team or Player? (team/player): ").strip().lower()
    if entity_type not in ['team', 'player']:
        print("Invalid selection. Please enter 'team' or 'player'.")
        return
    
    entity_name = input(f"Enter the {entity_type} name: ").strip()
    season = input("Enter the season (e.g., 2023-24): ").strip()
    
    opponent_type = input("Compare against all teams or a specific team? (all/specific): ").strip().lower()
    if opponent_type not in ['all', 'specific']:
        print("Invalid selection. Please enter 'all' or 'specific'.")
        return
    
    opponent_name = None
    if opponent_type == 'specific':
        opponent_name = input("Enter the opponent team name: ").strip()
    
    # Preload MAE tables for all teams
    mae_df_all = preload_mae_tables(entity_name, season)
    
    # Fetch and display offensive data
    shots = fetch_shots_data(entity_name, entity_type == 'team', season, opponent_name)
    print(shots.head())
    
    efficiency = calculate_efficiency(shots)
    print(f"Offensive Efficiency for {entity_name}:")
    print(efficiency)
    
    # Update the plot call to include the opponent's name or indicate it's against all teams
    fig = plot_shot_chart_hexbin(shots, f'{entity_name} Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
    plt.show()
    
    if opponent_type == 'specific':
        # MAE calculation and saving for specific team
        mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
        print(f"MAE Table for {entity_name} against {opponent_name}:")
        print(mae_df_specific)
    else:
        # MAE calculation and loading for all teams
        print(f"MAE Table for {entity_name} against all teams:")
        print(mae_df_all)
    
    min_season, max_season = get_seasons_range(mae_df_all)
    print(f"MAE Table available for seasons from {min_season} to {max_season}.")

    # If the analysis type is "both", also perform defensive analysis here
    if analysis_type == 'both':
        # Fetch and display defensive data for the specified team
        defensive_shots = fetch_defensive_shots_data(entity_name, True, season, opponent_name)
        defensive_efficiency = calculate_efficiency(defensive_shots)
        print(f"Defensive Efficiency for {entity_name}:")
        print(defensive_efficiency)
        
        # Update the plot call to include the opponent's name or indicate it's against all teams
        fig = plot_shot_chart_hexbin(defensive_shots, f'{entity_name} Defensive Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
        plt.show()
        
        if opponent_type == 'specific':
            # MAE calculation for defensive analysis against the specific opponent
            mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
            print(f"Defensive MAE Table for {entity_name} against {opponent_name}:")
            print(mae_df_specific)

if __name__ == "__main__":
    main()

