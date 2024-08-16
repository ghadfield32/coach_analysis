
import streamlit as st
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot, get_all_court_areas
from shot_chart.nba_shots import fetch_shots_for_multiple_players
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import create_mae_table, save_mae_table, load_mae_table, get_seasons_range, calculate_compatibility_between_players
from shot_chart.shot_chart_main import run_scenario
from nba_api.stats.static import players, teams

@st.cache_data
def get_teams_list():
    """Get the list of NBA teams."""
    return [team['full_name'] for team in teams.get_teams()]

@st.cache_data
def get_players_list():
    """Get the list of NBA players."""
    return [player['full_name'] for player in players.get_players()]

def main():
    st.title("NBA Shot Analysis")
    
    # Add guidelines and purpose explanation at the top
    st.markdown("""
    ### Welcome to the NBA Shot Analysis App!
    
    This app allows you to analyze the offensive and defensive efficiency of NBA teams and players. 
    You can compare players or teams to identify the most efficient spots on the court, 
    analyze player compatibility based on shot area efficiency, and much more.
    
    **Options and Guidelines:**
    - **Analysis Type**: Choose between offensive, defensive, or both types of analysis.
    - **Team or Player**: Analyze a team or an individual player.
    - **Court Areas**: Select specific court areas or analyze all areas.
    - **Comparison**: Compare multiple players to see how their offensive efficiencies align or differ.
    
    ### How to Find the Most Efficient Spots:
    - The app allows you to explore shot efficiency across different court areas.
    - You can see how players perform against other teams and how well they play together.
    - The MAE (Mean Absolute Error) metric helps identify the compatibility between players based on their shooting efficiency in various areas.
    """)
    
    analysis_type = st.selectbox("Select analysis type", options=["offensive", "defensive", "both"])
    
    entity_type = st.selectbox("Analyze a Team or Player?", options=["team", "player"])
    
    if entity_type == "team":
        st.markdown("_**Team option is able to analyze both offense and defense by looking into the defense by shot detail from other teams' shot charts against the Opposing Team.**_")
        entity_name = st.selectbox("Select a Team", options=get_teams_list())
    else:
        st.markdown("_**Player Option is only able to look at offense.**_")
        player_names = st.multiselect("Select Players to Analyze", options=get_players_list())
    
    season = st.selectbox("Select the season", options=["2023-24", "2022-23", "2021-22", "2020-21"])
    
    opponent_type = st.selectbox("Compare against all teams or a specific team?", options=["all", "specific"])
    
    opponent_name = None
    if opponent_type == "specific":
        opponent_name = st.selectbox("Select an Opponent Team", options=get_teams_list())
    
    court_areas = st.selectbox("Select court areas to analyze", options=["all", "specific"], index=0)
    
    if court_areas == "specific":
        court_areas = st.multiselect("Select specific court areas", options=get_all_court_areas())
    else:
        court_areas = "all"
    
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    
    if st.button("Run Analysis"):
        if entity_type == "player" and (not player_names or len(player_names) < 1):
            st.error("Please select at least one player.")
        else:
            if entity_type == "player":
                if len(player_names) == 1:
                    # Single player analysis
                    run_scenario(
                        entity_name=player_names[0],
                        entity_type=entity_type,
                        season=season,
                        opponent_name=opponent_name,
                        analysis_type=analysis_type,
                        compare_players=False,
                        player_names=None,
                        court_areas=court_areas
                    )
                else:
                    # Multiple players comparison
                    player_shots = fetch_shots_for_multiple_players(player_names, season, court_areas, opponent_name, debug=debug_mode)
                    
                    for player, shots in player_shots.items():
                        st.pyplot(plot_shot_chart_hexbin(shots['shots'], f'{player} Shot Chart', opponent=opponent_name if opponent_name else "all teams"))
                        st.write(f"Efficiency for {player}:")
                        st.write(shots['efficiency'])
                    
                    compatibility_df = calculate_compatibility_between_players(player_shots)
                    st.write("Player Shooting Area Compatibility:")
                    st.write(compatibility_df)
            else:
                # Team analysis
                run_scenario(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    season=season,
                    opponent_name=opponent_name,
                    analysis_type=analysis_type,
                    compare_players=False,
                    court_areas=court_areas
                )

    # Add explanation for shot chart MAE analysis
    with st.expander("Understanding MAE in Player Analysis with context from their Shooting"):
        st.markdown("""
        **MAE** is a metric that measures the average magnitude of errors between predicted values and actual values, without considering their direction.
        
        In our context, MAE is used to measure the difference between the shooting efficiencies of two players across various areas on the court.
        
        **Steps to Analyze MAE:**
        1. **Define Common Areas**: The court is divided into areas like "Left Corner 3", "Top of Key", "Paint", etc.
        2. **Calculate Individual Efficiencies**: Fetch shot data for each player and calculate their shooting efficiency in these areas.
        3. **Identify Common Areas**: When comparing players, identify the areas where both players have taken shots.
        4. **Calculate MAE**: Compute the absolute difference between efficiencies in each common area and average them.
        5. **Interpret Compatibility**:
            - **High MAE**: Indicates players excel in different areas (more compatible).
            - **Low MAE**: Indicates similar efficiencies in the same areas (less compatible).
        
        **Use this metric to assess player compatibility based on where they excel on the court!**
        """)
        
    with st.expander("Understanding MAE in Team (offensive or defensive) in comparison to other Teams"):
        st.markdown("""
        **MAE** is a metric that measures the average magnitude of errors between predicted values and actual values, without considering their direction.
        
        In the context of team analysis, MAE is used to measure the difference between the shooting efficiencies of one team's offense and the defensive efficiencies of other teams.
        
        **Steps to Analyze MAE for Team Comparison:**
        1. **Calculate Offensive Efficiency**: Fetch shot data for the team of interest and calculate their shooting efficiency across various areas on the court.
        2. **Calculate Defensive Efficiency of Opponents**: For each opponent team, calculate their defensive efficiency by analyzing how well they defend these same areas on the court.
        3. **Calculate MAE**: Compute the MAE between the offensive efficiency of the team of interest and the defensive efficiencies of each opponent team across the defined court areas.
        4. **Interpret the Results**:
            - **Low MAE**: Indicates that the opponent team is effective at defending the areas where the team of interest typically excels. This suggests that the opponent is a "bad fit" for the team of interest, as they defend well against their strengths.
            - **High MAE**: Indicates that the opponent team struggles to defend the areas where the team of interest typically excels. This suggests that the opponent is a "good fit" for the team of interest, as their defense is less effective against the team's offensive strengths.
        
        **Use this analysis to identify which teams are tough matchups (bad fits) versus easier matchups (good fits) based on how well they can defend your team's key offensive areas!**
        """)

if __name__ == "__main__":
    main()

