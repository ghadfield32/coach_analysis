
import os
import pandas as pd
from nba_api.stats.static import players, teams
import matplotlib.pyplot as plt
from modular.nba_helpers import fetch_and_save_players_list, load_players_list, get_team_abbreviation, categorize_shot
from modular.nba_shots import fetch_shots_data, fetch_defensive_shots_data
from modular.nba_plotting import plot_shot_chart_hexbin
from modular.nba_efficiency import calculate_efficiency, calculate_team_fit, create_mae_table, save_mae_table, load_mae_table

def main():
    home_team = 'Dallas Mavericks'  # Change this to the team or player name
    season = '2023-24'
    analysis_type = 'both'  # Change to 'offensive', 'defensive', or 'both'
    opponent_team = 'Boston Celtics'  # Set to the opponent team name if needed
    game_date = ''  # Optional: 'YYYY-MM-DD'
    
    all_teams = [team['full_name'] for team in teams.get_teams()]
    
    try:
        if analysis_type in ['offensive', 'both']:
            shots = fetch_shots_data(home_team, True, season, opponent_team, game_date)
            unique_dates = shots['GAME_DATE'].unique()
            print(shots.head())
            
            season_efficiency = calculate_efficiency(shots)
            print("Season Averages:")
            print(season_efficiency)
            
            fig = plot_shot_chart_hexbin(shots, f'{home_team} Shot Chart')
            plt.show()
        
        if analysis_type in ['defensive', 'both']:
            defensive_shots = fetch_defensive_shots_data(home_team, season, opponent_team, game_date)
            fig = plot_shot_chart_hexbin(defensive_shots, f'{home_team} Defensive Shot Chart')
            plt.show()
            
            defensive_efficiency = calculate_efficiency(defensive_shots)
            print("Defensive Efficiency:")
            print(defensive_efficiency)
        
        if analysis_type == 'both':
            mae_df = load_mae_table()
            if mae_df is None:
                mae_df = create_mae_table(home_team, season, all_teams)
                save_mae_table(mae_df)
            print(mae_df)
        
    except Exception as e:
        print(f"Error fetching or plotting data: {e}")

if __name__ == "__main__":
    main()
