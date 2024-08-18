
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import plotly.graph_objects as go
from datetime import datetime
from nba_api.stats.static import teams, players

# Import functions from other modules
from data_loader_preprocessor import load_data, format_season, clean_data, engineer_features, encode_data
from model_trainer import train_and_save_models, evaluate_models
from model_predictor import predict

# Import functions from app_test_trade_impact.py
from app_test_trade_impact import analyze_trade_impact, get_players_for_team

# Importing Shot Chart Analysis functions
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot, get_all_court_areas
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data, fetch_shots_for_multiple_players
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import create_mae_table, save_mae_table, load_mae_table, get_seasons_range, calculate_compatibility_between_players
from shot_chart.shot_chart_main import run_scenario, preload_mae_tables, create_and_save_mae_table_specific, create_and_save_mae_table_all

@st.cache_data
def get_teams_list():
    """Get the list of NBA teams."""
    return [team['full_name'] for team in teams.get_teams()]

@st.cache_data
def get_players_list():
    """Get the list of NBA players."""
    return [player['full_name'] for player in players.get_players()]

@st.cache_data
def load_team_data():
    nba_teams = teams.get_teams()
    team_df = pd.DataFrame(nba_teams)
    return team_df[['id', 'full_name', 'abbreviation']]

@st.cache_data
def load_player_data(start_year, end_year):
    player_data = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        data = fetch_season_data_by_year(year)
        if data is not None:
            player_data = pd.concat([player_data, data], ignore_index=True)
    return player_data

def identify_overpaid_underpaid(predictions_df):
    # Adjust Predicted_Salary calculation
    predictions_df['Predicted_Salary'] = predictions_df['Predicted_Salary'] * predictions_df['Salary_Cap_Inflated']
    
    predictions_df['Salary_Difference'] = predictions_df['Salary'] - predictions_df['Predicted_Salary']
    predictions_df['Overpaid'] = predictions_df['Salary_Difference'] > 0
    predictions_df['Underpaid'] = predictions_df['Salary_Difference'] < 0
    
    overpaid = predictions_df[predictions_df['Overpaid']].sort_values('Salary_Difference', ascending=False)
    underpaid = predictions_df[predictions_df['Underpaid']].sort_values('Salary_Difference')
    
    return overpaid.head(10), underpaid.head(10)

# Utility functions
def load_processed_data(file_path):
    data = load_data(file_path)
    data = format_season(data)
    data = clean_data(data)
    data = engineer_features(data)
    return data

def filter_data_by_season(data, season):
    return data[data['Season'] == season]

# Data visualization functions
def plot_feature_distribution(data, feature):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[feature], kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    return fig

def plot_correlation_heatmap(data):
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

# Model metrics function
def display_model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Squared Error", f"{mse:.4f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    col3.metric("Mean Absolute Error", f"{mae:.4f}")
    col4.metric("R-squared", f"{r2:.4f}")

def display_overpaid_underpaid(predictions_df):
    st.subheader("Top 10 Overpaid and Underpaid Players")

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        team_filter = st.multiselect("Filter by Team", options=sorted(predictions_df['Team'].unique()))
    with col2:
        position_filter = st.multiselect("Filter by Position", options=sorted(predictions_df['Position'].unique()))

    # Apply filters
    filtered_df = predictions_df
    if team_filter:
        filtered_df = filtered_df[filtered_df['Team'].isin(team_filter)]
    if position_filter:
        filtered_df = filtered_df[filtered_df['Position'].isin(position_filter)]

    # Identify overpaid and underpaid players
    overpaid, underpaid = identify_overpaid_underpaid(filtered_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Overpaid Players")
        st.dataframe(overpaid[['Player', 'Team', 'Position', 'Salary', 'Predicted_Salary', 'Salary_Difference']])

    with col2:
        st.subheader("Top 10 Underpaid Players")
        st.dataframe(underpaid[['Player', 'Team', 'Position', 'Salary', 'Predicted_Salary', 'Salary_Difference']])


# Trade Impact Simulator function
def display_trade_impact(results, team_a_name, team_b_name):
    st.subheader(f"Percentile Counts Impact on {team_a_name} Compared to Average Champion Percentiles")
    st.table(results['celtics_comparison_table'])

    st.subheader(f"Percentile Counts Impact on {team_b_name} Compared to Average Champion Percentiles")
    st.table(results['warriors_comparison_table'])

    st.subheader("Pre vs Post Trade Impact")
    st.table(results['overall_comparison'])

    st.subheader("Salary Analysis: Overpaid vs Underpaid")
    st.table(results['salary_analysis'])

    st.subheader("Salary Cap Clearance: (debug shows Salary cap vs Salary Help)")
    st.write(results['trade_scenario_results'])



def trade_impact_simulator():
    st.header("Trade Impact Simulator")

    teams = load_team_data()
    team_a_name = st.selectbox("Select Team A", teams['full_name'])
    team_b_name = st.selectbox("Select Team B", teams['full_name'][teams['full_name'] != team_a_name].tolist())

    players_from_team_a = st.multiselect(f"Select Players from {team_a_name}", get_players_for_team(team_a_name))
    players_from_team_b = st.multiselect(f"Select Players from {team_b_name}", get_players_for_team(team_b_name))

    trade_date = st.date_input("Trade Date", value=pd.to_datetime("2023-12-20"))

    percentile_seasons = st.multiselect("Select Average Champion Comparison Percentiles Seasons (default: 10):", 
                                        ["2014-15", "2015-16", "2016-17", "2017-18", 
                                         "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
                                        default=["2014-15", "2015-16", "2016-17", "2017-18", 
                                         "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"])

    if st.button("Simulate Trade Impact"):
        if players_from_team_a and players_from_team_b:
            traded_players = {player: team_a_name for player in players_from_team_a}
            traded_players.update({player: team_b_name for player in players_from_team_b})

            # Call the analysis function and capture the results
            results = analyze_trade_impact(traded_players, trade_date.strftime('%Y-%m-%d'), percentile_seasons, debug=True)

            if results:
                # Display all results in the app
                display_trade_impact(results, team_a_name, team_b_name)

                # MAE Analysis section
                st.subheader("Player Shooting Area Compatibility (MAE Analysis)")

                # Combine the players from both teams
                all_players = players_from_team_a + players_from_team_b

                # Fetch shots for the selected players
                player_shots = fetch_shots_for_multiple_players(all_players, season='2023-24', court_areas='all')

                # Calculate compatibility
                compatibility_df = calculate_compatibility_between_players(player_shots)

                # Display the MAE table
                st.write(compatibility_df)

                # Salary Cap Clearance section
                st.subheader("Salary Cap Clearance: (debug shows Salary cap vs Salary Help)")
                st.write(results['trade_scenario_results'])

                # Display debug output
                st.subheader("Debug Information")
                st.text_area("Detailed Debug Output", results['trade_scenario_debug'], height=300)

                st.success("Trade Impact Simulation Completed")
            else:
                st.error("Trade impact analysis failed. Please check the inputs.")
        else:
            st.error("Please select players from both teams to simulate the trade impact.")


# Shot Chart Analysis function
def shot_chart_analysis():
    st.header("Shot Chart Analysis")

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

# Main Streamlit app
def main():
    st.set_page_config(page_title="NBA Salary Prediction, Trade Analysis, and Shot Chart Analysis", layout="wide")
    st.title("NBA Salary Prediction, Trade Analysis, and Shot Chart Analysis")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Model Results", "Salary Evaluation", "Trade Impact Simulator", "Shot Chart Analysis"])

    # Load base data
    print(os.getcwd())
    data = load_processed_data('data/processed/nba_player_data_final_inflated.csv')

    # Load existing predictions for 2023
    initial_predictions_df = pd.read_csv('data/processed/predictions_df.csv')

    # Season selection
    seasons = sorted(data['Season'].unique(), reverse=True)
    selected_season = st.selectbox("Select Season", seasons)

    # Load models at the beginning of main()
    model_save_path = 'data/models'
    rf_model = joblib.load(f"{model_save_path}/best_rf_model.pkl")
    xgb_model = joblib.load(f"{model_save_path}/best_xgb_model.pkl")

    # Use initial predictions if 2023 is selected, otherwise retrain
    if selected_season == 2023:
        predictions_df = initial_predictions_df
    else:
        # Train model and make predictions
        train_data = data[data['Season'] < selected_season]
        test_data = data[data['Season'] == selected_season]

        # Prepare the data for training
        X_train = train_data.drop(['SalaryPct', 'Salary', 'Player'], axis=1)
        y_train = train_data['SalaryPct']

        # Encode the training data
        X_train_encoded, _, encoders, scaler, numeric_cols, player_encoder = encode_data(X_train)

        # Train and save models
        train_and_save_models(X_train_encoded, y_train, model_save_path, scaler, X_train_encoded.columns, encoders, player_encoder, numeric_cols)

        # Make predictions on the test data
        predictions_df = predict(test_data, model_save_path)

    if page == "Data Analysis":
        st.header("Data Analysis")

        # Filter data by selected season
        season_data = filter_data_by_season(data, selected_season)

        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(season_data.describe())

        # Feature distribution
        st.subheader("Feature Distribution")
        feature = st.selectbox("Select Feature", season_data.columns)
        fig = plot_feature_distribution(season_data, feature)
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig = plot_correlation_heatmap(season_data)
        st.pyplot(fig)

        # Data handling explanation
        st.subheader("Data Handling")
        st.write("""
        We preprocessed the data to ensure it's suitable for our models:
        1. Cleaned missing values and outliers
        2. Engineered new features like PPG, APG, etc.
        3. Encoded categorical variables (Position, Team, Injury Risk)
        4. Scaled numerical features
        """)

    elif page == "Model Results":
        st.header("Model Results")

        # Model selection
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])

        if model_choice == "Random Forest":
            model = rf_model
            y_pred = predictions_df['RF_Predictions']
        else:
            model = xgb_model
            y_pred = predictions_df['XGB_Predictions']

        # Display model metrics
        display_model_metrics(predictions_df['SalaryPct'], y_pred)

        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Features used in the model:", model.feature_names_in_)  # Debug statement
        print("Feature importance data:", feature_importance.head())  # Debug statement
        # Filter out categorical variables (e.g., Position_ and Team_ columns)
        filtered_feature_importance = feature_importance[
            ~feature_importance['feature'].str.startswith('Team_') &
            ~feature_importance['feature'].str.startswith('Position_') &
            ~feature_importance['feature'].str.startswith('Injury_Risk')
        ]

        # Before plotting feature importance
        print("Filtered features for plotting:", filtered_feature_importance['feature'].tolist())  # Debug statement
        st.bar_chart(filtered_feature_importance.set_index('feature'))


        # Model explanation
        st.subheader("Model Explanation")
        st.write(f"""
        The {model_choice} model was trained on historical NBA player data to predict salary percentages.
        We used the following techniques to improve model performance:
        1. Feature engineering to create relevant statistics
        2. Proper encoding of categorical variables
        3. Scaling of numerical features
        4. Hyperparameter tuning using GridSearchCV
        """)
        
    elif page == "Salary Evaluation":
        st.header("Salary Evaluation")
        display_overpaid_underpaid(predictions_df)
        st.write("""
        Using the Predicted Salary and Salary based on that season's stats, we can identify 
        overpaid and underpaid players. We find the difference to uncover potential value opportunities for different teams.
        """)
        
    elif page == "Trade Impact Simulator":
        trade_impact_simulator()
        
        # Trade analysis explanation
        st.subheader("Trade Analysis Explanation")
        st.write("""
        Our trade analysis compares team statistics before and after the proposed trade.
        We consider:
        1. Changes in key performance metrics (PPG, RPG, APG, etc.)
        2. Salary implications and cap space impact
        3. Comparison to league averages and recent championship teams
        4. Distribution of top performers in various statistical categories
        5. Overpaid/Underpaid player analysis
        """)

    elif page == "Shot Chart Analysis":
        shot_chart_analysis()

if __name__ == "__main__":
    main()
