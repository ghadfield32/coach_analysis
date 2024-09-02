
import streamlit as st
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelogs
from datetime import date
from combined_trade_analysis import combined_trade_analysis

# Function to fetch players for a specific team
def get_players_for_team(team_name, season="2023-24"):
    team_id = teams.find_teams_by_full_name(team_name)[0]['id']
    team_players = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    team_players = team_players[team_players['TEAM_ID'] == team_id]
    return sorted(team_players['PLAYER_NAME'].unique())

# Function to fetch unique game dates for a season
def get_unique_game_dates(season):
    gamelogs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    return sorted(gamelogs['GAME_DATE'].unique())

# Function to determine the NBA season based on the trade date
def get_trade_season(trade_date):
    year = trade_date.year
    if trade_date.month in [10, 11, 12]:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"

# Function to generate a list of the last 10 NBA seasons
def get_last_n_seasons(current_season, n=10):
    current_year = int(current_season.split('-')[0])
    seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(current_year - n + 1, current_year + 1)]
    return seasons  # Return in ascending order

# Main function to run the Streamlit app
def main():
    st.title("NBA Trade Impact Analysis")

    st.write("""
    ## About This App
    This application allows you to analyze the impact of a trade between two NBA teams. It includes the following components:
    
    ### 1. Trade Scenario Analysis:
    - Ensure the trade satisfies NBA salary matching rules based on the provided player salaries.

    ### 2. Percentile Counts:
    - The count of top 1, 2, 3, 4, 5, 10, 25, 50 percentiles of the team's performance before and after the trade,
      compared to the last 'n' seasons selected in the champion season filter.

    ### 3. Overall Trade Impact:
    - **Pre-Trade Scenario**:
        * Data Collection: Filter season data to include only games before the trade date.
        * Statistical Calculations: Calculate total points and games played before the trade.
        * Averaging: Calculate average points per game before the trade.
        * Percentile Ranking: Rank teams based on pre-trade performance.
    - **Post-Trade Scenario**:
        * Data Collection: Filter season data for games on or after the trade date.
        * Player Averages: Calculate average points for traded players post-trade.
        * Simulating Game Logs: Simulate additional game logs using calculated player averages.
        * Statistical Calculations: Combine simulated and actual post-trade data for calculations.
        * Averaging: Calculate average points per game post-trade.
        * Percentile Ranking: Rank teams based on post-trade performance.
    - **No-Trade Scenario**:
        * Data Collection: Use full season data assuming no trades occurred.
        * Statistical Calculations: Calculate total points and games played for the entire season.
        * Averaging: Calculate average points per game for the full season.
        * Percentile Ranking: Rank teams based on full-season performance.
    - **Final Comparison**:
        * Aggregation: Organize pre-trade, post-trade, and no-trade results.
        * Metrics Compared: Total points, games played, average points per game, and percentile rankings.

    ### 4. Overpaid/Underpaid Player Analysis:
    - Analyze whether the players involved in the trade are overpaid or underpaid based on predicted salaries.

    ### 5. Player Compatibility Analysis:
    - Calculate the compatibility between the players being traded based on their shooting areas.
    """)

    # Load the predictions data
    predictions_df = pd.read_csv('data/processed/predictions_df.csv')

    # Select teams
    all_teams = [team['full_name'] for team in teams.get_teams()]
    team_a_name = st.selectbox("Select Team A", all_teams, key="team_a")
    team_b_name = st.selectbox("Select Team B", [team for team in all_teams if team != team_a_name], key="team_b")

    # Fetch and display players for each team
    players_from_team_a = get_players_for_team(team_a_name)
    players_from_team_b = get_players_for_team(team_b_name)
    
    selected_players_team_a = st.multiselect(f"Select Players from {team_a_name}", players_from_team_a, key="players_a")
    selected_players_team_b = st.multiselect(f"Select Players from {team_b_name}", players_from_team_b, key="players_b")

    # Select trade season
    trade_season = st.selectbox("Select Trade Season", 
                                ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"], 
                                index=4)

    # Champion seasons selection
    last_10_seasons = get_last_n_seasons(trade_season, n=10)
    champion_seasons = st.multiselect("Select Champion Seasons for Comparison", last_10_seasons, default=last_10_seasons)
    
    # Ensure champion seasons are in ascending order
    champion_seasons = sorted(champion_seasons)

    # Options to select full season or specific date
    analysis_option = st.radio("Select Analysis Period", options=["Full Season", "Specific Date"])

    if analysis_option == "Specific Date":
        unique_dates = get_unique_game_dates(trade_season)
        trade_date = st.selectbox("Select Trade Date", unique_dates)
    else:
        # For full season analysis, use an offseason date to ensure full season is considered.
        trade_date = date(int(trade_season.split('-')[0]), 8, 15)  # August 15th as a sample offseason date

    # Display the criteria used for the analysis at the top of the results section
    st.write(f"""
    ### Analysis Criteria:
    - **Team A:** {team_a_name}
    - **Players from {team_a_name}:** {", ".join(selected_players_team_a)}
    - **Team B:** {team_b_name}
    - **Players from {team_b_name}:** {", ".join(selected_players_team_b)}
    - **Season:** {trade_season}
    """)
    
    if analysis_option == "Specific Date":
        st.write(f"- **Trade Date:** {trade_date}")
    
    st.write(f"- **Champion Seasons for Comparison:** {', '.join(champion_seasons)}")

    # Add a checkbox to include or exclude debug columns
    include_debug_columns = st.checkbox("Include Debug Columns (Games and Totals)", value=False)

    # Validate inputs before running the analysis
    if st.button("Analyze Trade Impact"):
        if not selected_players_team_a or not selected_players_team_b:
            st.error("Please select at least one player from each team to analyze the trade impact.")
        else:
            with st.spinner('Analyzing trade impact...'):
                relevant_stats = ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'FGM', 'FG3M', 'FGA']
                
                try:
                    results = combined_trade_analysis(
                        team_a_name, team_b_name, selected_players_team_a, selected_players_team_b, 
                        trade_date, champion_seasons, trade_season, relevant_stats, predictions_df, debug=include_debug_columns
                    )

                    # Display trade scenario analysis
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        st.write("### Trade Scenario Analysis:")
                        st.text(results['trade_analysis'])

                        st.write("### Average Champion Percentiles:")
                        st.dataframe(results['average_champion_percentiles'])

                        st.write(f"### {team_a_name} Comparison Table:")
                        st.dataframe(results['team_a_comparison_table'])

                        st.write(f"### {team_b_name} Comparison Table:")
                        st.dataframe(results['team_b_comparison_table'])

                        for stat, table in results['comparison_tables'].items():
                            st.write(f"### Comparison Table for {stat}:")
                            st.dataframe(table)

                        # Display overpaid/underpaid player analysis
                        st.write("### Overpaid/Underpaid Player Analysis:")
                        st.dataframe(results['salary_analysis'])

                        # Display player compatibility analysis
                        st.write("### Player Compatibility Analysis:")
                        st.dataframe(results['compatibility_analysis'])
                        
                except Exception as e:
                    st.error(f"An error occurred during the analysis: {e}")

if __name__ == "__main__":
    main()
