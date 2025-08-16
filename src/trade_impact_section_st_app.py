
# Required imports
import streamlit as st
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelogs
from datetime import date
from trade_impact.combined_trade_analysis import combined_trade_analysis

# --- REPLACE this helper; keep the name used by the app ---
def convert_season_format(year):
    """
    Converts an input year (e.g., 2023 or '2023') to the season format (e.g., '2023-24').
    If the input is already 'YYYY-YY', returns it unchanged.
    """
    from trade_impact.utils.nba_api_utils import normalize_season
    return normalize_season(year)



def get_players_for_team(team_name, season="2023-24", *, use_live: bool = True, debug: bool = False):
    """
    Prefer CommonTeamRoster (light). If it fails and no roster cache exists,
    derive roster from league PlayerGameLogs cache for the season.
    """
    from trade_impact.utils.nba_api_utils import (
        get_team_id_by_full_name,
        get_commonteamroster_df,
        get_playergamelogs_df,
        normalize_season,
    )

    season_norm = normalize_season(season)
    team_id = get_team_id_by_full_name(team_name)
    if team_id is None:
        return []

    # 1) Try roster endpoint (with fallback to cache inside)
    try:
        roster = get_commonteamroster_df(team_id, season_norm, use_live=use_live, debug=debug)
        if "PLAYER" in roster.columns and roster["PLAYER"].notna().any():
            return sorted(roster["PLAYER"].dropna().astype(str).unique().tolist())
        if debug:
            print(f"[get_players_for_team] No players in roster dataframe; will try logs fallback")
    except Exception as e:
        if debug:
            print(f"[get_players_for_team] roster fetch error: {e}")

    # 2) Derive from season logs (works if logs cache was prewarmed or fetched once)
    try:
        logs = get_playergamelogs_df(season_norm, use_live=use_live, debug=debug)
        if "TEAM_NAME" in logs.columns and "PLAYER_NAME" in logs.columns:
            derived = logs.loc[logs["TEAM_NAME"] == team_name, "PLAYER_NAME"].dropna().astype(str).unique().tolist()
            if derived:
                return sorted(derived)
    except Exception as e:
        if debug:
            print(f"[get_players_for_team] logs fallback error: {e}")

    # 3) Final attempt: cache-only logs (don’t touch network)
    try:
        logs = get_playergamelogs_df(season_norm, use_live=False, debug=debug)
        if "TEAM_NAME" in logs.columns and "PLAYER_NAME" in logs.columns:
            derived = logs.loc[logs["TEAM_NAME"] == team_name, "PLAYER_NAME"].dropna().astype(str).unique().tolist()
            if derived:
                return sorted(derived)
    except Exception as e:
        if debug:
            print(f"[get_players_for_team] cache-only logs fallback error: {e}")

    return []



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


def get_unique_game_dates(season, *, use_live: bool = True, debug: bool = False):
    """
    Return sorted unique game dates (date objects) for the given season.
    Uses league-wide PlayerGameLogs via utils to leverage retries/cache.
    """
    import pandas as pd
    from trade_impact.utils.nba_api_utils import get_playergamelogs_df, normalize_season

    season_norm = normalize_season(season)
    if debug:
        print(f"[get_unique_game_dates] season={season_norm} use_live={use_live}")

    logs = get_playergamelogs_df(season_norm, timeout=90, retries=3, use_live=use_live, debug=debug)
    if "GAME_DATE" not in logs.columns:
        if debug:
            print(f"[get_unique_game_dates] 'GAME_DATE' missing. Columns={list(logs.columns)}")
        return []

    dates = pd.to_datetime(logs["GAME_DATE"], errors="coerce").dt.date.dropna().unique().tolist()
    dates_sorted = sorted(dates)

    if debug:
        print(f"[get_unique_game_dates] unique_dates={len(dates_sorted)} sample={dates_sorted[:5]}")
    return dates_sorted



def trade_impact_simulator_app(selected_season="2023"):
    from trade_impact.utils.nba_api_utils import normalize_season
    formatted_season = normalize_season(selected_season)

    st.title(f"NBA Trade Impact Analysis - {formatted_season}")

    st.sidebar.subheader("Data Source")
    use_live_api = st.sidebar.checkbox("Use live NBA API", value=True,
        help="Uncheck to use cached data only. If live calls fail, cached data will be used when available.")

    st.markdown("""
    ## About This App

    This application allows you to analyze the impact of a trade between two NBA teams. It includes the following components:

    ### 1. Trade Scenario Analysis:
    - Ensure the trade satisfies NBA salary matching rules based on the provided player salaries.

    ### 2. Percentile Counts:
    - The count of top 1, 2, 3, 4, 5, 10, 25, 50 percentiles of the team's performance before and after the trade, compared to the last 'n' seasons selected in the champion season filter.

    ### 3. Overall Trade Impact:
    - **Pre-Trade Scenario**:
        * **Data Collection:** Filter season data to include only games before the trade date.
        * **Statistical Calculations:** Calculate total points and games played before the trade.
        * **Averaging:** Calculate average points per game before the trade.
        * **Percentile Ranking:** Rank teams based on pre-trade performance.

    - **Post-Trade Scenario**:
        * **Data Collection:** Filter season data for games on or after the trade date.
        * **Player Averages:** Calculate average points for traded players post-trade.
        * **Simulating Game Logs:** Simulate additional game logs using calculated player averages.
        * **Statistical Calculations:** Combine simulated and actual post-trade data for calculations.
        * **Averaging:** Calculate average points per game post-trade.
        * **Percentile Ranking:** Rank teams based on post-trade performance.

    - **No-Trade Scenario**:
        * **Data Collection:** Use full season data assuming no trades occurred.
        * **Statistical Calculations:** Calculate total points and games played for the entire season.
        * **Averaging:** Calculate average points per game for the full season.
        * **Percentile Ranking:** Rank teams based on full-season performance.

    - **Final Comparison**:
        * **Aggregation:** Organize pre-trade, post-trade, and no-trade results.
        * **Metrics Compared:** Total points, games played, average points per game, and percentile rankings.

    ### 4. Overpaid/Underpaid Player Analysis:
    - Analyze whether the players involved in the trade are overpaid or underpaid based on predicted salaries.

    ### 5. Player Compatibility Analysis:
    - Calculate the compatibility between the players being traded based on their shooting areas.
    """)

    # Load predictions (unchanged)
    predictions_df = pd.read_csv('data/processed/predictions_df.csv')

    # Team and player selectors
    all_teams = [team['full_name'] for team in teams.get_teams()]
    team_a_name = st.selectbox("Select Team A", all_teams, key="team_a")
    team_b_name = st.selectbox("Select Team B", [t for t in all_teams if t != team_a_name], key="team_b")

    # Safely populate player lists using cached, lightweight calls
    try:
        players_a_options = get_players_for_team(team_a_name, formatted_season, use_live=use_live_api, debug=True)
    except Exception as e:
        players_a_options = []
        st.warning(f"Could not load roster for {team_a_name}: {e}")

    try:
        players_b_options = get_players_for_team(team_b_name, formatted_season, use_live=use_live_api, debug=True)
    except Exception as e:
        players_b_options = []
        st.warning(f"Could not load roster for {team_b_name}: {e}")

    players_from_team_a = st.multiselect(f"Select Players from {team_a_name}", players_a_options)
    players_from_team_b = st.multiselect(f"Select Players from {team_b_name}", players_b_options)

    # Champion seasons
    def get_last_n_seasons(current_season, n=10):
        y = int(str(current_season)[:4])
        return [f"{yr}-{str(yr+1)[-2:]}" for yr in range(y - n + 1, y + 1)]

    last_10_seasons = get_last_n_seasons(formatted_season)
    champion_seasons = st.multiselect("Select Champion Seasons for Comparison", last_10_seasons, default=last_10_seasons)
    champion_seasons = sorted(champion_seasons)

    analysis_option = st.radio("Select Analysis Period", options=["Full Season", "Specific Date"])

    if analysis_option == "Specific Date":
        try:
            unique_dates = get_unique_game_dates(formatted_season, use_live=use_live_api, debug=True)
            trade_date = st.selectbox("Select Trade Date", unique_dates)
        except Exception as e:
            st.error(f"Could not load game dates for {formatted_season}: {e}")
            return
    else:
        y = int(str(formatted_season)[:4])
        from datetime import date
        trade_date = date(y, 8, 15)  # offseason default

    st.write(f"### Analysis Criteria:\n- **Team A:** {team_a_name}\n- **Team B:** {team_b_name}\n- **Season:** {formatted_season}\n- **Champion Seasons:** {', '.join(champion_seasons)}")

    include_debug_columns = st.checkbox("Include Debug Columns (Games and Totals)", value=False)

    if st.button("Analyze Trade Impact"):
        if not players_from_team_a or not players_from_team_b:
            st.error("Please select at least one player from each team.")
            return

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
