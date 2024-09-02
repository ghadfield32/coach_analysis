
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

# Importing Shot Chart Analysis functions
from shot_chart.nba_helpers import get_team_abbreviation, categorize_shot, get_all_court_areas
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data, fetch_shots_for_multiple_players
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_efficiency import create_mae_table, save_mae_table, load_mae_table, get_seasons_range, calculate_compatibility_between_players
from shot_chart.shot_chart_main import run_scenario, preload_mae_tables, create_and_save_mae_table_specific, create_and_save_mae_table_all

# Import functions from the small example app
from advanced_metrics import plot_career_clusters, plot_injury_risk_vs_salary, plot_availability_vs_salary, plot_vorp_vs_salary, table_metric_salary, display_top_10_salary_per_metric, cluster_players_specialized, display_top_10_salary_per_metric_with_ws

# Import New and improved Trade functions
from trade_impact_section_st_app import trade_impact_simulator_app

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


# Advanced Metrics Analysis Function
def advanced_metrics_analysis():
    st.header("NBA Advanced Metrics and Salary Analysis")
    
    # Load the data
    data = pd.read_csv('data/processed/nba_player_data_final_inflated.csv')
    
    # Add a dropdown to select the season
    seasons = sorted(data['Season'].unique(), reverse=True)
    selected_season = st.selectbox("Select a Season", seasons)
    
    # Filter the data by the selected season
    data_season = data[data['Season'] == selected_season]
    
    # Cluster players based on the filtered data
    data_season = cluster_players_specialized(data_season, n_clusters=7)
    
    st.header("Plots")
    
    # Dropdown to select the plot
    plot_choice = st.selectbox("Select a plot to view:", 
                               ["Career Clusters: Age vs Salary", 
                                "Injury Risk vs Salary", 
                                "Availability vs Salary", 
                                "VORP vs Salary"])
    
    if plot_choice == "Career Clusters: Age vs Salary":
        fig = plot_career_clusters(data_season)
        st.pyplot(fig)
    elif plot_choice == "Injury Risk vs Salary":
        fig = plot_injury_risk_vs_salary(data_season)
        st.pyplot(fig)
    elif plot_choice == "Availability vs Salary":
        fig = plot_availability_vs_salary(data_season)
        st.pyplot(fig)
    elif plot_choice == "VORP vs Salary":
        fig = plot_vorp_vs_salary(data_season)
        st.pyplot(fig)
    
    st.header("Top 10 Salary per Metric Tables")
    
    # Calculate metrics table
    metric_salary_table = table_metric_salary(data_season)
    
    # Dropdown to select the metric table
    metric_choice = st.selectbox("Select a metric to view top 10:", 
                                 ["Salary_per_WS", 
                                  "Salary_per_VORP", 
                                  "Salary_per_OWS", 
                                  "Salary_per_DWS"])
    
    # Display the selected top 10 table with WS included
    top_10_table = display_top_10_salary_per_metric_with_ws(metric_salary_table, metric_choice)
    st.write(f"Top 10 {metric_choice}:")
    st.dataframe(top_10_table)

# Main Streamlit app
def main():
    st.set_page_config(page_title="NBA Salary Prediction, Trade Analysis, and Shot Chart Analysis", layout="wide")
    st.title("NBA Salary Prediction, Trade Analysis, and Shot Chart Analysis")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Introduction",
            "Data Analysis",
            "Model Results",
            "Salary Evaluation",
            "Shot Chart Analysis",
            "Advanced Metrics Analysis",
            "Trade Impact Simulator"
        ]
    )

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


    if page == "Introduction":
        st.title("Enhanced NBA Player Salary Analysis")
        st.write("Welcome to the NBA Salary Analysis and Prediction App! This project aims to provide comprehensive insights into NBA player salaries, advanced metrics, and future salary predictions based on historical data. Here's a detailed breakdown of the steps involved in creating this app:")

        st.subheader("Data Collection")
        
        st.write("### Salary Data")
        st.write("- **Sources**:")
        st.write("  - [Basketball Reference Salary Cap History](https://www.basketball-reference.com/contracts/salary-cap-history.html)")
        st.write("- **Description**: Data on the NBA salary cap from various seasons, along with maximum salary details for players based on years of service.")

        st.write("### Add Injury Data (source will need to be updated**):")
        st.write("- **Source**: [Kaggle NBA Injury Stats 1951-2023](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023/data)")
        st.write("- **Description**: This dataset provides detailed statistics on NBA injuries from 1951 to 2023, allowing for analysis of player availability and its impact on performance and salaries.")

        st.write("### Advanced Metrics")
        st.write("- **Source**: [Basketball Reference](https://www.basketball-reference.com)")
        st.write("- **Description**: Advanced player metrics such as Player Efficiency Rating (PER), True Shooting Percentage (TS%), and Value Over Replacement Player (VORP) were scraped using BeautifulSoup.")

        st.write("### Player Salaries and Team Data")
        st.write("- **Source**: [Hoopshype](https://hoopshype.com)")
        st.write("- **Description**: Player salary data was scraped for multiple seasons, with detailed information on individual player earnings and team salaries.")

        st.subheader("Data Processing")

        st.write("### Inflation Adjustment")
        st.write("- **Source**: [Adjusting for Inflation in Python](https://medium.com/analytics-vidhya/adjusting-for-inflation-when-analysing-historical-data-with-python-9d69a8dcbc27)")
        st.write("- **Description**: Adjusted historical salary data for inflation to provide a consistent basis for comparison.")

        st.write("### Data Aggregation")
        st.write("- Steps:")
        st.write("  1. Loaded salary data and combined it with team standings and advanced metrics.")
        st.write("  2. Merged multiple data sources to create a comprehensive dataset containing player performance, salaries, and advanced metrics.")

        st.subheader("Model Training and Prediction")

        st.write("### Data Preprocessing")
        st.write("- Implemented functions to handle missing values, perform feature engineering, and calculate key metrics such as points per game (PPG), assists per game (APG), and salary growth.")

        st.write("### Model Selection")
        st.write("- Utilized various machine learning models including Random Forest, Gradient Boosting, Ridge Regression, and others to predict future player salaries.")
        st.write("- Employed grid search for hyperparameter tuning and selected the best-performing models based on evaluation metrics like Mean Squared Error (MSE) and R² score.")

        st.write("### Feature Importance and Clustering")
        st.write("- Analyzed feature importance to understand the key factors influencing player salaries.")
        st.write("- Clustered players into categories based on career trajectories, providing insights into player development and value.")

        st.subheader("App Development")

        st.write("### Streamlit App")
        st.write("- Built an interactive app using Streamlit to visualize data, perform exploratory data analysis, and make salary predictions.")
        st.write("- **Features**:")
        st.write("  - **Data Overview**: Display raw and processed data.")
        st.write("  - **Exploratory Data Analysis**: Visualize salary distributions, age vs. salary, and other key metrics.")
        st.write("  - **Advanced Analytics**: Analyze VORP to salary ratio, career trajectory clusters, and other advanced metrics.")
        st.write("  - **Salary Predictions**: Predict future salaries and compare actual vs. predicted values.")
        st.write("  - **Player Comparisons**: Compare selected players based on predicted salaries and performance metrics.")
        st.write("  - **Model Evaluation**: Evaluate different models and display their performance metrics and feature importance.")

        st.write("### Data Files")
        st.write("- Stored processed data and model files in a structured format to facilitate easy loading and analysis within the app.")

        st.subheader("Improvements:")
        
        st.subheader("Conclusion")

        st.write("This app provides a robust platform for analyzing NBA player salaries, understanding the factors influencing earnings, and predicting future salaries based on historical data and advanced metrics. Explore the app to gain insights into player performance, salary trends, and much more.")


    elif page == "Data Analysis":
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
        

    elif page == "Shot Chart Analysis":
        shot_chart_analysis()

    elif page == "Advanced Metrics Analysis":
        advanced_metrics_analysis()

    elif page == "Trade Impact Simulator":
        st.header("Trade Impact Simulator")
        
        # Assume selected_season is in YYYY format (e.g., 2023)
        trade_impact_simulator_app(selected_season)
        

if __name__ == "__main__":
    main()
