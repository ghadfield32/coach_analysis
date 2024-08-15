
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs, commonplayerinfo
from nba_api.stats.static import teams, players
import time
from tabulate import tabulate
import streamlit as st

# Importing the functions from the respective scripts
from percentile_count_trade_impact import (
    fetch_all_player_data,
    calculate_player_stats,
    calculate_player_percentiles,  
    simulate_trade,
    create_comparison_table,
    get_champion_percentiles,
    generate_comparison_tables
)

from overall_team_trade_impact import (
    fetch_and_process_player_data,
    calculate_combined_team_stats,
    trade_impact_analysis
)

from nba_rules_trade_impact import (
    check_salary_matching_rules,
    analyze_trade_scenario
)

# Function to fetch players for a specific team
def get_players_for_team(team_name, season="2023-24"):
    """Fetch players for a given team name."""
    team_id = teams.find_teams_by_full_name(team_name)[0]['id']
    team_players = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    team_players = team_players[team_players['TEAM_ID'] == team_id]
    return sorted(team_players['PLAYER_NAME'].unique())

# Main function to analyze trade impact
def analyze_trade_impact(
    traded_players, 
    trade_date, 
    percentile_seasons=None, 
    debug=False
):
    # Set default percentile seasons if not provided
    if percentile_seasons is None:
        percentile_seasons = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

    # Extract the unique teams involved in the trade
    teams_involved = list(set(traded_players.values()))
    if len(teams_involved) != 2:
        raise ValueError("This function supports trades involving exactly two teams.")

    # Split the players by team for comparison and salary analysis
    team_a_name = teams_involved[0]
    team_b_name = teams_involved[1]
    players_from_team_a = [player for player, team_name in traded_players.items() if team_name == team_a_name]
    players_from_team_b = [player for player, team_name in traded_players.items() if team_name == team_b_name]

    if debug:
        print(f"Team A: {team_a_name}, Players: {players_from_team_a}")
        print(f"Team B: {team_b_name}, Players: {players_from_team_b}")

    # Step 1: Champion Percentile Analysis for Specific Seasons
    print("=== Champion Percentile Analysis ===")
    # Determine the overall trade start and end years based on the percentile seasons
    overall_trade_start_year = int(min(season.split('-')[0] for season in percentile_seasons))
    overall_trade_end_year = int(trade_date.split('-')[0])
    
    average_top_percentiles_df = get_champion_percentiles(percentile_seasons, debug)
    if debug:
        print("\nAverage Champion Percentiles:")
        print(average_top_percentiles_df)

    # Generate comparison tables before and after the trade
    celtics_comparison_table, warriors_comparison_table = generate_comparison_tables(
        percentile_seasons[-1], team_a_name, team_b_name, players_from_team_a, players_from_team_b, average_top_percentiles_df, debug
    )
    
    print(f"\nComparison Table for {team_a_name} vs {team_b_name}:")
    print(celtics_comparison_table)
    print(warriors_comparison_table)

    # Step 2: Overall Trade Impact Analysis
    print("\n=== Overall Trade Impact Analysis ===")
    
    # Use traded_players for the overall trade impact analysis
    comparison_table = trade_impact_analysis(
        overall_trade_start_year, overall_trade_end_year, trade_date, traded_players, 
        champion_filter='Average Champion', debug=debug
    )
    
    # Print the comparison table
    print(comparison_table)

    # Step 3: Trade Scenario Analysis Based on Salary Matching Rules
    print("\n=== Trade Scenario Analysis ===")
    predictions_df = pd.read_csv('../data/processed/predictions_df.csv')
    
    # Set the salary check season based on the trade date
    salary_check_season = overall_trade_end_year
    print(f"Analyzing trade for the {salary_check_season} season:")

    print(f"\nAnalyzing trade between {team_a_name} and {team_b_name}:")
    analyze_trade_scenario(players_from_team_a, players_from_team_b, predictions_df, salary_check_season, debug=False)

if __name__ == "__main__":
    # Example usage with input variables

    traded_players = {
        'Jaylen Brown': 'Boston Celtics',
        'Jayson Tatum': 'Boston Celtics',
        'Stephen Curry': 'Golden State Warriors',
        'Klay Thompson': 'Golden State Warriors'
    }

    trade_date = '2023-12-20'
    percentile_seasons = ["2022-23", "2023-24"]
    
    analyze_trade_impact(traded_players, trade_date, percentile_seasons, debug=False)


# Example usage in a Streamlit app:

# import streamlit as st

# # User inputs for the Streamlit app
# st.title("Trade Impact Analysis")

# # Step 1: Select Teams
# st.header("Select Teams Involved in the Trade")
# team_names = sorted([team['full_name'] for team in teams.get_teams()])
# selected_teams = st.multiselect("Select up to 4 teams", team_names, default=["Boston Celtics", "Golden State Warriors"], max_selections=4)

# # Step 2: Select Players from each Team
# traded_players = {}
# for team in selected_teams:
#     st.subheader(f"Select Players from {team}")
#     players = get_players_for_team(team)
#     selected_players = st.multiselect(f"Select players from {team}", players, max_selections=4)
#     for player in selected_players:
#         traded_players[player] = team

# # Step 3: Select Trade Date
# trade_date = st.date_input("Trade Date", value=pd.to_datetime("2023-12-20"))

# # Step 4: Select Number of Seasons for Percentile Analysis
# num_seasons = st.slider("Number of Seasons for Analysis", 5, 20, 10)
# start_season = st.text_input("Starting Season (e.g., 2014-15)", "2014-15")
# percentile_seasons = [f"{int(start_season.split('-')[0]) + i}-{str(int(start_season.split('-')[0]) + i + 1)[-2:]}" for i in range(num_seasons)]

# # Step 5: Run the Analysis
# if st.button("Run Analysis"):
#     analyze_trade_impact(traded_players, trade_date.strftime('%Y-%m-%d'), percentile_seasons, debug=False)
