
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
    Fast, robust player list for a specific team in a given season.
    - Prefers TeamID-based filtering (authoritative).
    - Falls back to full-name / abbreviation match if TeamID missing.
    - Adds explicit debugs; no filling or silent coercion.
    """
    from trade_impact.utils.nba_api_utils import normalize_season
    from nba_api.stats.static import teams as _static_teams
    import pandas as pd

    season_norm = normalize_season(season)

    # 1) Load season players (index or roster fallback)
    players_df = get_players_for_season_fast(season_norm, debug=debug).copy()

    # Hard guard: nothing to do
    if players_df is None or players_df.empty:
        if debug:
            print(f"[get_players_for_team] season={season_norm}: players_df is empty or None")
        return []

    # 2) Resolve TeamID + abbreviation from static teams
    all_teams = _static_teams.get_teams()
    by_full = {t["full_name"].casefold(): t for t in all_teams}
    by_abbr = {t["abbreviation"].casefold(): t for t in all_teams}
    meta = by_full.get(team_name.casefold()) or by_abbr.get(team_name.casefold())

    if meta is None:
        if debug:
            print(f"[get_players_for_team] cannot resolve team metadata for '{team_name}'")
        return []

    team_id = int(meta["id"])
    abbr    = meta["abbreviation"]
    full    = meta["full_name"]

    # 3) Dump quick diagnostics (transparent, non-destructive)
    if debug:
        cols = players_df.columns.tolist()
        print(f"[get_players_for_team] season={season_norm} team={full} (id={team_id}, abbr={abbr})")
        print(f"[get_players_for_team] players_df rows={len(players_df)}, cols={cols}")
        if "Team" in players_df.columns:
            uniq = players_df["Team"].dropna().astype(str).unique().tolist()[:12]
            print(f"[get_players_for_team] sample Team values (first ~12): {uniq}")
        if "TeamID" in players_df.columns:
            print(f"[get_players_for_team] TeamID non-null rows: {int(players_df['TeamID'].notna().sum())}")

    # 4) Prefer TeamID-based filtering (authoritative)
    mask = pd.Series(False, index=players_df.index)
    if "TeamID" in players_df.columns and players_df["TeamID"].notna().any():
        # handle dtype safely
        mask = mask | (players_df["TeamID"].astype("Int64") == team_id)

    # 5) Fallbacks if TeamID path yields nothing
    if not mask.any() and "Team" in players_df.columns:
        tser = players_df["Team"].astype(str)

        # direct full-name or abbr match
        mask = mask | tser.str.casefold().eq(full.casefold()) | tser.str.upper().eq(abbr.upper())

        # handle common historical aliases that appear in older datasets
        aliases = {
            "NOP": ["NOH", "NOK"],
            "BKN": ["NJN"],
            "CHA": ["CHO"],     # modern Charlotte used in some sources as CHO
            "WAS": ["WSB"],     # old Bullets
            "OKC": ["SEA"],     # SuperSonics to Thunder
            "LAC": ["SDC"],     # historical Clippers alias
            "SAC": ["KCK", "ROC", "CIN", "KCO"],  # Kings franchise history codes in some sources
        }
        for alt in aliases.get(abbr.upper(), []):
            mask = mask | tser.str.upper().eq(alt)

    # 6) Final extraction
    team_players = players_df.loc[mask, "Player"].dropna().unique().tolist()
    team_players.sort()

    if debug:
        count_id = int((players_df["TeamID"].astype("Int64") == team_id).sum()) if "TeamID" in players_df.columns else -1
        count_lbl = 0
        if "Team" in players_df.columns:
            tser = players_df["Team"].astype(str)
            count_lbl = int((tser.str.casefold().eq(full.casefold()) | tser.str.upper().eq(abbr.upper())).sum())
        print(f"[get_players_for_team] matched via TeamID={count_id}, via label/abbr≈{count_lbl}, final={len(team_players)}")

        if len(team_players) == 0:
            # show first few rows for triage — still no filling, just visibility
            sample = players_df.head(12).to_string(index=False)
            print(f"[get_players_for_team] ZERO players after matching. Head of df:\n{sample}")

    return team_players


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
