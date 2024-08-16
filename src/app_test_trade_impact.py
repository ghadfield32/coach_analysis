
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

def analyze_player_salaries(players, predictions_df):
    """Analyze if the selected players are overpaid or underpaid based on predicted salaries."""
    player_salary_analysis = []

    for player in players:
        player_data = predictions_df[predictions_df['Player'] == player]
        if not player_data.empty:
            actual_salary = player_data['Salary'].values[0]
            salary_cap = player_data['Salary_Cap_Inflated'].values[0]
            predicted_salary = player_data['Predicted_Salary'].values[0] * salary_cap
            difference = actual_salary - predicted_salary
            status = "Overpaid" if difference > 0 else "Underpaid" if difference < 0 else "Fairly Paid"
            player_salary_analysis.append({
                'Player': player,
                'Actual Salary': actual_salary,
                'Predicted Salary': predicted_salary,
                'Difference': difference,
                'Status': status
            })

    return pd.DataFrame(player_salary_analysis)


# Main function to analyze trade impact
def analyze_trade_impact(
    traded_players, 
    trade_date, 
    percentile_seasons=None, 
    debug=False
):
    if percentile_seasons is None:
        percentile_seasons = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

    teams_involved = list(set(traded_players.values()))
    if len(teams_involved) != 2:
        raise ValueError("This function supports trades involving exactly two teams.")

    team_a_name = teams_involved[0]
    team_b_name = teams_involved[1]
    players_from_team_a = [player for player, team_name in traded_players.items() if team_name == team_a_name]
    players_from_team_b = [player for player, team_name in traded_players.items() if team_name == team_b_name]

    overall_trade_start_year = int(min(season.split('-')[0] for season in percentile_seasons))
    overall_trade_end_year = int(trade_date.split('-')[0])

    # Step 1: Champion Percentile Analysis for Specific Seasons
    average_top_percentiles_df = get_champion_percentiles(percentile_seasons, debug)

    # Generate comparison tables before and after the trade
    celtics_comparison_table, warriors_comparison_table = generate_comparison_tables(
        percentile_seasons[-1], team_a_name, team_b_name, players_from_team_a, players_from_team_b, average_top_percentiles_df, debug
    )

    # Step 2: Overall Trade Impact Analysis
    comparison_table = trade_impact_analysis(
        overall_trade_start_year, overall_trade_end_year, trade_date, traded_players, 
        champion_filter='Average Champion', debug=debug
    )

    # Step 3: Trade Scenario Analysis Based on Salary Matching Rules
    predictions_df = pd.read_csv('data/processed/predictions_df.csv')
    salary_check_season = overall_trade_end_year
    trade_scenario_results, trade_scenario_debug = analyze_trade_scenario(players_from_team_a, players_from_team_b, predictions_df, salary_check_season, debug=debug)

    # Overpaid/Underpaid analysis
    predictions_df = pd.read_csv('data/processed/predictions_df.csv')
    all_players = list(traded_players.keys())
    salary_analysis_df = analyze_player_salaries(all_players, predictions_df)

    return {
        'celtics_comparison_table': celtics_comparison_table,
        'warriors_comparison_table': warriors_comparison_table,
        'overall_comparison': comparison_table,
        'trade_scenario_results': trade_scenario_results,
        'trade_scenario_debug': trade_scenario_debug,
        'salary_analysis': salary_analysis_df
    }

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
    
    results = analyze_trade_impact(traded_players, trade_date, percentile_seasons, debug=True)

    # Print the results including the debug output
    print(results['trade_scenario_debug'])
    print("results =", results)


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
