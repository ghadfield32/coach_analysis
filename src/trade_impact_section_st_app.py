
# Required imports
import streamlit as st
import pandas as pd
from nba_api.stats.static import teams
from datetime import date
from trade_impact.combined_trade_analysis import combined_trade_analysis
from streamlit_app_helpers import get_players_for_season_fast, check_network_connectivity

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
    Fast + robust players list for a specific team in a season.
    Priority:
      1) Season index (local parquet): [Player, PlayerID, Team, TeamID]
      2) Authoritative fallback: CommonTeamRoster(team_id, season)
    No filling/masking. Heavily instrumented for diagnostics.
    """
    import pandas as pd
    from nba_api.stats.static import teams as _static_teams
    from trade_impact.utils.nba_api_utils import normalize_season
    from streamlit_app_helpers import get_players_for_season_fast
    from salary_nba_data_pull.fetch_utils import fetch_team_roster

    season_norm = normalize_season(season)

    # Resolve team metadata
    all_teams = _static_teams.get_teams()
    by_full = {t["full_name"].casefold(): t for t in all_teams}
    by_abbr = {t["abbreviation"].casefold(): t for t in all_teams}
    meta = by_full.get(team_name.casefold()) or by_abbr.get(team_name.casefold())
    if meta is None:
        if debug:
            print(f"[get_players_for_team] cannot resolve team metadata for '{team_name}'")
        return []
    team_id = int(meta["id"]); abbr = meta["abbreviation"]; full = meta["full_name"]

    # Try the season index first (fast path)
    idx = get_players_for_season_fast(season_norm, debug=debug).copy()
    if not idx.empty:
        if debug:
            print(f"[get_players_for_team] source=index  rows={len(idx)}  "
                  f"cols={list(idx.columns)}  season={season_norm} team={full}({team_id})")
        players = []
        try:
            mask = pd.Series(False, index=idx.index)
            if "TeamID" in idx.columns and idx["TeamID"].notna().any():
                mask |= (idx["TeamID"].astype("Int64") == team_id)
            if (not mask.any()) and ("Team" in idx.columns):
                tser = idx["Team"].astype(str)
                mask |= tser.str.casefold().eq(full.casefold()) | tser.str.upper().eq(abbr.upper())
            players = (idx.loc[mask, ["Player", "PlayerID"]]
                          .dropna(subset=["Player"])
                          .drop_duplicates()
                          .sort_values("Player")["Player"].tolist())
            if debug:
                via_id = int((idx.get("TeamID", pd.Series(dtype="Int64")).astype("Int64") == team_id).sum()) if "TeamID" in idx.columns else -1
                via_lbl = 0
                if "Team" in idx.columns:
                    tser = idx["Team"].astype(str)
                    via_lbl = int((tser.str.casefold().eq(full.casefold()) | tser.str.upper().eq(abbr.upper())).sum())
                print(f"[get_players_for_team] index-match via TeamID={via_id}, via label≈{via_lbl}, final={len(players)}")
        except Exception as e:
            if debug:
                print(f"[get_players_for_team][index-path][ERROR] {e}")
            players = []
        # Check if index returned a reasonable team size
        MIN_TEAM_SIZE = 8  # Reasonable minimum for a full roster
        if len(players) >= MIN_TEAM_SIZE:
            if debug:
                print(f"[get_players_for_team] using index result ({len(players)} players)")
            return players
        elif len(players) > 0:
            if debug:
                print(f"[get_players_for_team] index returned only {len(players)} players, trying roster fallback")

    # Authoritative fallback: CommonTeamRoster
    if debug:
        print(f"[get_players_for_team] source=CommonTeamRoster fallback  season={season_norm} team={full}({team_id})")
    roster = fetch_team_roster(team_id=team_id, season=season_norm, debug=debug)
    if roster.empty:
        if debug:
            print(f"[get_players_for_team] roster fallback returned 0 rows")
        # Return whatever the index had as last resort
        return players if 'players' in locals() else []
    
    name_col = "PLAYER" if "PLAYER" in roster.columns else None
    if name_col is None:
        if debug:
            print(f"[get_players_for_team] roster columns unexpected: {list(roster.columns)}")
        # Return whatever the index had as last resort
        return players if 'players' in locals() else []
    
    roster_players = roster[[name_col]].dropna().drop_duplicates().sort_values(name_col)[name_col].tolist()
    if debug:
        print(f"[get_players_for_team] roster fallback returned {len(roster_players)} players")
    
    # Prefer roster result if it's substantial, otherwise fall back to index
    if len(roster_players) >= MIN_TEAM_SIZE:
        return roster_players
    else:
        if debug:
            print(f"[get_players_for_team] roster fallback also sparse ({len(roster_players)}), using index result")
        return players if 'players' in locals() else roster_players


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
    logs = get_playergamelogs_df(season_norm, use_live=use_live, debug=debug)
    
    if "GAME_DATE" not in logs.columns or logs.empty:
        if debug:
            print(f"[get_unique_game_dates] No game dates found for {season_norm}")
        return []
    
    # Convert to date objects and get unique sorted dates
    dates = pd.to_datetime(logs["GAME_DATE"]).dt.date.unique()
    return sorted(dates)



def trade_impact_simulator_app(selected_season="2023"):
    from trade_impact.utils.nba_api_utils import normalize_season
    formatted_season = normalize_season(selected_season)

    st.title(f"NBA Trade Impact Analysis - {formatted_season}")

    st.sidebar.subheader("Data Source")
    use_live_api = st.sidebar.checkbox("Use live NBA API", value=True,
        help="Uncheck to use cached data only. If live calls fail, cached data will be used when available.")
    
    # Debug option for troubleshooting
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False,
        help="Show detailed debug information in the console/logs for troubleshooting team player loading issues.")

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
        players_a_options = get_players_for_team(team_a_name, formatted_season, use_live=use_live_api, debug=debug_mode)
    except Exception as e:
        players_a_options = []
        st.warning(f"Could not load roster for {team_a_name}: {e}")

    try:
        players_b_options = get_players_for_team(team_b_name, formatted_season, use_live=use_live_api, debug=debug_mode)
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
