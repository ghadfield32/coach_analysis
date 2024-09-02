
# Required imports
import streamlit as st
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelogs
from datetime import date
from trade_impact.combined_trade_analysis import combined_trade_analysis

def convert_season_format(year):
    """
    Converts a single year (e.g., 2023) to the season format (e.g., 2023-24).
    """
    next_year = str(int(year) + 1)[-2:]
    return f"{year}-{next_year}"

def get_players_for_team(team_name, season="2023-24"):
    team_id = teams.find_teams_by_full_name(team_name)[0]['id']
    team_players = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    team_players = team_players[team_players['TEAM_ID'] == team_id]
    return sorted(team_players['PLAYER_NAME'].unique())

def get_unique_game_dates(season):
    gamelogs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    return sorted(gamelogs['GAME_DATE'].unique())

def get_trade_season(trade_date):
    year = trade_date.year
    if trade_date.month in [10, 11, 12]:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def get_last_n_seasons(current_season, n=10):
    current_year = int(current_season.split('-')[0])
    seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(current_year - n + 1, current_year + 1)]
    return seasons  # Return in ascending order


def display_trade_impact_results(results, team_a_name, team_b_name):
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

    st.write("### Overpaid/Underpaid Player Analysis:")
    st.dataframe(results['salary_analysis'])

    st.write("### Player Compatibility Analysis:")
    st.dataframe(results['compatibility_analysis'])


def trade_impact_simulator_app(selected_season="2023"):
    # Convert the selected season to the format "YYYY-YY"
    formatted_season = convert_season_format(selected_season)
    
    st.title(f"NBA Trade Impact Analysis - {formatted_season}")

    # Overview of the app
    st.write("""
    ## About This App
    This application allows you to analyze the impact of a trade between two NBA teams...
    """)

    # Load the predictions data
    predictions_df = pd.read_csv('data/processed/predictions_df.csv')

    # Team and player selection
    all_teams = [team['full_name'] for team in teams.get_teams()]
    team_a_name = st.selectbox("Select Team A", all_teams, key="team_a")
    team_b_name = st.selectbox("Select Team B", [team for team in all_teams if team != team_a_name], key="team_b")
    
    players_from_team_a = st.multiselect(f"Select Players from {team_a_name}", get_players_for_team(team_a_name, formatted_season))
    players_from_team_b = st.multiselect(f"Select Players from {team_b_name}", get_players_for_team(team_b_name, formatted_season))
    
    last_10_seasons = get_last_n_seasons(formatted_season)
    champion_seasons = st.multiselect("Select Champion Seasons for Comparison", last_10_seasons, default=last_10_seasons)
    champion_seasons = sorted(champion_seasons)  # Ensure ascending order

    # Analysis option
    analysis_option = st.radio("Select Analysis Period", options=["Full Season", "Specific Date"])

    if analysis_option == "Specific Date":
        unique_dates = get_unique_game_dates(formatted_season)
        trade_date = st.selectbox("Select Trade Date", unique_dates)
    else:
        trade_date = date(int(formatted_season.split('-')[0]), 8, 15)  # Use an offseason date

    # Display selected analysis criteria
    st.write(f"### Analysis Criteria: \n - **Team A:** {team_a_name} \n - **Team B:** {team_b_name} \n - **Season:** {formatted_season} \n - **Champion Seasons:** {', '.join(champion_seasons)}")
    
    # Checkbox for including debug columns
    include_debug_columns = st.checkbox("Include Debug Columns (Games and Totals)", value=False)

    # Analyze the trade impact
    if st.button("Analyze Trade Impact"):
        if not players_from_team_a or not players_from_team_b:
            st.error("Please select at least one player from each team.")
        else:
            with st.spinner('Analyzing trade impact...'):
                try:
                    results = combined_trade_analysis(
                        team_a_name, team_b_name, players_from_team_a, players_from_team_b, 
                        trade_date, champion_seasons, formatted_season, 
                        ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'FGM', 'FG3M', 'FGA'], 
                        predictions_df, debug=include_debug_columns
                    )
                    display_trade_impact_results(results, team_a_name, team_b_name)
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    trade_impact_simulator_app()
