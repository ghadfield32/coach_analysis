
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
from nba_api.stats.static import teams

# Import functions from other modules
from data_loader_preprocessor import load_data, format_season, clean_data, engineer_features, encode_data
from model_trainer import train_and_save_models, evaluate_models
from model_predictor import predict

# Import functions from app_test_trade_impact.py
from app_test_trade_impact import analyze_trade_impact, get_players_for_team

#importing shot chart app functions
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import calculate_efficiency, create_mae_table, save_mae_table, load_mae_table, get_seasons_range
from shot_chart.shot_chart_main import preload_mae_tables, create_and_save_mae_table_specific, create_and_save_mae_table_all

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
    st.subheader(f"Percentile Counts Impact on {team_a_name}Compared to Average Champion Percentiles")
    st.table(results['celtics_comparison_table'])

    st.subheader(f"Percentile Counts Impact on {team_b_name} Compared to Average Champion Percentiles")
    st.table(results['warriors_comparison_table'])

    st.subheader("Pre vs Post Trade Impact")
    st.table(results['overall_comparison'])

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
                
                # Display debug output
                st.subheader("Debug Information")
                st.text_area("Detailed Debug Output", results['trade_scenario_debug'], height=300)

                st.success("Trade Impact Simulation Completed")
            else:
                st.error("Trade impact analysis failed. Please check the inputs.")
        else:
            st.error("Please select players from both teams to simulate the trade impact.")




# Main Streamlit app
def main():
    st.set_page_config(page_title="NBA Salary Prediction and Trade Analysis", layout="wide")
    st.title("NBA Salary Prediction and Trade Analysis")

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
        Using the Predicted Salary and Salary based on that seasons stats we can get 
        overpaid/underpaid players. We find the difference to find the biggest disparities to
        hopefully uncover a couple finds that would be great for different teams.
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
        st.header("Shot Analysis to find MAE against other teams or defense against other teams")
        
        analysis_type = st.selectbox("Select analysis type", options=["offensive", "defensive", "both"])
        
        entity_type = st.selectbox("Analyze a Team or Player?", options=["team", "player"])
        
        if entity_type == "team":
            entity_name = st.selectbox("Select a Team", options=get_teams_list())
        else:
            entity_name = st.selectbox("Select a Player", options=get_players_list())
        
        season = st.selectbox("Select the season", options=["2023-24", "2022-23", "2021-22", "2020-21"])
        
        opponent_type = st.selectbox("Compare against all teams or a specific team?", options=["all", "specific"])
        
        opponent_name = None
        if opponent_type == "specific":
            opponent_name = st.selectbox("Select an Opponent Team", options=get_teams_list())
        
        if st.button("Run Analysis"):
            # Preload MAE tables for all teams
            mae_df_all = preload_mae_tables(entity_name, season)
            
            # Fetch and display offensive data
            shots = fetch_shots_data(entity_name, entity_type == 'team', season, opponent_name)
            st.write("Shot Data")
            st.dataframe(shots.head())
            
            efficiency = calculate_efficiency(shots)
            st.write(f"Offensive Efficiency for {entity_name}:")
            st.dataframe(efficiency)
            
            # Plot shot chart
            fig = plot_shot_chart_hexbin(shots, f'{entity_name} Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
            st.pyplot(fig)
            
            if opponent_type == 'specific':
                # MAE calculation and saving for specific team
                mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
                st.write(f"MAE Table for {entity_name} against {opponent_name}:")
                st.dataframe(mae_df_specific)
            else:
                # MAE calculation and loading for all teams
                st.write(f"MAE Table for {entity_name} against all teams:")
                st.dataframe(mae_df_all)
            
            min_season, max_season = get_seasons_range(mae_df_all)
            st.write(f"MAE Table available for seasons from {min_season} to {max_season}.")
        
            # If the analysis type is "both", also perform defensive analysis here
            if analysis_type == 'both':
                # Fetch and display defensive data for the specified team
                defensive_shots = fetch_defensive_shots_data(entity_name, True, season, opponent_name)
                defensive_efficiency = calculate_efficiency(defensive_shots)
                st.write(f"Defensive Efficiency for {entity_name}:")
                st.dataframe(defensive_efficiency)
                
                # Plot defensive shot chart
                fig = plot_shot_chart_hexbin(defensive_shots, f'{entity_name} Defensive Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
                st.pyplot(fig)
                
                if opponent_type == 'specific':
                    # MAE calculation for defensive analysis against the specific opponent
                    mae_df_specific = create_and_save_mae_table_specific(entity_name, season, opponent_name)
                    st.write(f"Defensive MAE Table for {entity_name} against {opponent_name}:")
                    st.dataframe(mae_df_specific)

        
if __name__ == "__main__":
    main()
