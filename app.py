import sys
import os

# Add the notebooks directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'notebooks'))

import streamlit as st
import pandas as pd
from datetime import datetime
from nba_api.stats.static import teams
from notebooks.modular.nba_helpers import load_players_list, get_team_abbreviation
from notebooks.modular.nba_shots import fetch_shots_data, fetch_defensive_shots_data
from notebooks.modular.nba_plotting import plot_shot_chart_hexbin
from notebooks.modular.nba_efficiency import calculate_efficiency, save_mae_table, load_mae_table

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("NBA Shot Chart Analysis")

    # Select season
    season = st.selectbox("Select Season:", ["2023-24", "2022-23", "2021-22"])  # Add more seasons as needed

    # Choice between player or team
    analysis_type = st.radio("Analyze a Player or a Team", ["Player", "Team"])

    name = None
    is_team = False

    # Select player or team name
    if analysis_type == "Player":
        try:
            players_list = load_players_list(season)
            player_list = sorted(players_list['full_name'].unique())
            name = st.selectbox("Select Player Name:", player_list)
        except Exception as e:
            st.write(f"Error fetching current season players: {e}")
            return
    else:
        team_dictionary = teams.get_teams()
        team_list = sorted([team['full_name'] for team in team_dictionary])
        name = st.selectbox("Select Team Name:", team_list)
        is_team = True

    # Fetch the shot data to get available dates
    try:
        shots = fetch_shots_data(name, is_team, season)
        unique_dates = shots['GAME_DATE'].unique()
        unique_dates = pd.to_datetime(unique_dates, format='%Y%m%d').strftime('%Y-%m-%d').tolist()
    except Exception as e:
        st.write(f"Error fetching data for available dates: {e}")
        return

    # Display season averages
    try:
        season_efficiency = calculate_efficiency(shots)
        st.write("Season Averages:")
        st.write(season_efficiency)
    except Exception as e:
        st.write(f"Error calculating season averages: {e}")
        return

    # Dial choice between date or opponent team
    filter_choice = st.radio("Filter by Game Date or Opponent Team", ["Game Date", "Opponent Team"])

    game_date_str = ""
    opponent_team = ""

    if filter_choice == "Game Date":
        game_date_str = st.selectbox("Select Game Date (Optional):", [""] + unique_dates)
    else:
        team_dictionary = teams.get_teams()
        team_list = sorted([team['full_name'] for team in team_dictionary])
        opponent_team = st.selectbox("Select Opponent Team Name (Optional):", [""] + team_list)

    if st.button("Analyze"):
        try:
            st.write("Fetching data...")
            # Re-fetch the shots data with the appropriate filters
            shots = fetch_shots_data(name, is_team, season, opponent_team, game_date_str)

            if shots.empty:
                st.write("No shots found for this game.")
                return

            # Print some of the data for debugging
            st.write("Filtered shots data:")
            st.write(shots.head())

            efficiency = calculate_efficiency(shots)
            st.write("Filtered Efficiency:")
            st.write(efficiency)

            fig = plot_shot_chart_hexbin(shots, f'{name} Shot Chart' + (f' on {game_date_str}' if game_date_str else f' against {opponent_team}'))
            st.pyplot(fig)
        except Exception as e:
            st.write(f"Error fetching or plotting data: {e}")

    # Defensive shot chart analysis
    if analysis_type == "Team" and st.button("Analyze Defensive Data"):
        try:
            st.write("Fetching defensive data...")
            defensive_shots = fetch_defensive_shots_data(name, season, opponent_team, game_date_str)

            if defensive_shots.empty:
                st.write("No defensive shots found for this game.")
                return

            # Print some of the data for debugging
            st.write("Filtered defensive shots data:")
            st.write(defensive_shots.head())

            defensive_efficiency = calculate_efficiency(defensive_shots)
            st.write("Defensive Efficiency:")
            st.write(defensive_efficiency)

            fig = plot_shot_chart_hexbin(defensive_shots, f'{name} Defensive Shot Chart' + (f' on {game_date_str}' if game_date_str else f' against {opponent_team}'))
            st.pyplot(fig)
        except Exception as e:
            st.write(f"Error fetching or plotting defensive data: {e}")

if __name__ == "__main__":
    main()
