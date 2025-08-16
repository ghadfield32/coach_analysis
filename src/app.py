
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import joblib
import matplotlib.pyplot as plt
from nba_api.stats.static import teams, players

#importing model utils
from salary_model_training.data_loader_preprocessor import format_season, engineer_features, label_encode_injury_risk, build_pipeline, filter_seasons
from salary_model_training.util_functions import check_or_train_model, display_feature_importance, display_model_metrics, identify_overpaid_underpaid, plot_feature_importance
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

def data_analysis():
    st.header("Data Analysis")

    # Load the data
    file_path = 'data/processed/nba_player_data_final_inflated.csv'
    data = pd.read_csv(file_path)

    # Add a dropdown to select the season
    seasons = sorted(data['Season'].unique(), reverse=True)
    selected_season = st.selectbox("Select a Season", seasons)

    # Filter data by selected season
    season_data = data[data['Season'] == selected_season]

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

    # Data preprocessing explanation
    st.subheader("Data Preprocessing")
    st.write("""
    We preprocess the data to ensure it's suitable for modeling. Here are the key steps involved:
    1. **Cleaning Data**: Handle missing values, clean advanced statistics columns, and remove unnecessary columns.
    2. **Feature Engineering**: Create new features like Points Per Game (PPG), Availability, Salary Percentages, and Efficiency.
    3. **Label Encoding**: Encode categorical features like 'Injury Risk' and 'Position'.
    4. **Scaling**: Scale numerical features to normalize the data.
    5. **Season Filtering**: Filter the data by seasons to prepare train and test datasets.
    """)

    # Add preprocessing steps breakdown
    preprocessing_step = st.selectbox("Select a Preprocessing Step", [
        "Clean Data",
        "Feature Engineering",
        "Label Encoding"
    ])

    if preprocessing_step == "Clean Data":
        st.write("""
        In this step, we remove unnecessary columns such as 'Wins', 'Losses', and '2nd Apron', and handle missing data in percentage-based columns (e.g., 3P%, FT%, 2P%). 
        The columns are dropped based on the assumption that they do not contribute significantly to salary prediction.
        """)
        st.write("Cleaned Data Columns: ", data.columns.tolist())

    elif preprocessing_step == "Feature Engineering":
        st.write("""
        We derive new features such as:
        - **PPG (Points Per Game)**: Points scored per game.
        - **Availability**: Games played / total games in a season.
        - **SalaryPct**: Salary as a percentage of the inflated salary cap.
        - **Efficiency**: A custom efficiency metric based on offensive and defensive stats.
        """)
        st.write("Engineered Feature Example: Availability and Efficiency")
        engineered_data, pipeline_data, _ = engineer_features(season_data)
        st.write(engineered_data[['Availability', 'Efficiency']].head())

    elif preprocessing_step == "Label Encoding":
        st.write("""
        We encode categorical features like 'Injury Risk' into numeric values to feed into the machine learning model. For example:
        - **Low Risk**: 1
        - **Moderate Risk**: 2
        - **High Risk**: 3
        """)
        label_encoded_data = label_encode_injury_risk(season_data)
        st.write(label_encoded_data[['Injury_Risk']].head())


def convert_season_format(season_str):
    try:
        # Ensure we are splitting the season string correctly
        if isinstance(season_str, str):
            print(f"Original season string: {season_str}")  # Debug: Print original season string

            # Split the season by '-' (e.g., '2023-24' -> ['2023', '24'])
            year = season_str.split('-')[0]  # Get '2023'

            print(f"Formatted season string (year only): {year}")  # Debug: Print year only

            return year  # Return only the starting year
        else:
            raise TypeError(f"Expected a string, but got {type(season_str)}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return season_str  # Fallback to original season if there's an issue
    except Exception as e:
        print(f"Error formatting season: {e}")
        raise

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


# Main app logic
def main():
    st.set_page_config(page_title="NBA Salary Prediction, Analysis, and Simulator", layout="wide")
    st.title("NBA Salary Prediction, Data Analysis, and Trade Impact Simulator")

    # Load the data
    file_path = 'data/processed/nba_player_data_final_inflated.csv'
    data = pd.read_csv(file_path)

    # Get the unique seasons and exclude the earliest one
    seasons = sorted(data['Season'].unique(), reverse=True)  # Sort in descending order
    if len(seasons) > 1:
        seasons = seasons[:-1]  # Remove the earliest season (the last element in the sorted list)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Introduction", 
        "Data Analysis", 
        "Model Results", 
        "Salary Evaluation", 
        "Shot Chart Analysis", 
        "Advanced Metrics Analysis", 
        "Trade Impact Simulator"
    ])

    # Season Selection (format to integer year)
    selected_season = st.selectbox("Select Season", seasons)
    season_year = int(selected_season.split('-')[0])

    # File Paths
    model_save_path = f'data/models/season_{season_year}'

    # Load or train model and get predictions
    predictions_df = check_or_train_model(file_path, model_save_path, season_year)

    # Load models (to be reused across pages)
    rf_model_path = f'{model_save_path}/best_rf_model.pkl'
    xgb_model_path = f'{model_save_path}/best_xgb_model.pkl'

    try:
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)
        feature_names = joblib.load(f'{model_save_path}/feature_names.pkl')
    except FileNotFoundError:
        st.error("Models or feature names not found for the selected season. Please ensure the models are trained.")
        return

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
        st.subheader("Original Data")
        original_df = pd.read_csv(file_path)
        st.dataframe(original_df)

        st.subheader("Predicted Data")
        st.write("Here are the salary predictions generated based on Random Forest and XGBoost models.")
        st.dataframe(predictions_df)

    elif page == "Data Analysis":
        data_analysis()


    elif page == "Model Results":
        st.header("Model Results")

        # 1) Metrics table (loaded from evaluation_results.pkl)
        metrics_df = display_model_metrics(model_save_path)
        if metrics_df.empty:
            st.warning("No evaluation metrics found for this season. Train the models first.")
        else:
            st.subheader("Model Performance Metrics")
            st.dataframe(metrics_df)

        # 2) Choose model to inspect feature importance
        model_choice = st.selectbox("Select model for feature importance", ["Random Forest", "XGBoost"])

        if model_choice == "Random Forest":
            model = rf_model
        else:
            model = xgb_model

        # 3) Feature count
        from salary_model_training.util_functions import get_feature_count
        n_features = get_feature_count(model)
        if n_features > 0:
            st.write(f"**Number of features in model:** {n_features}")
        else:
            st.write("**Number of features in model:** (not available for this estimator)")

        # 4) Feature importance: filtered DF + chart
        if model_choice == "Random Forest":
            st.subheader("Random Forest Feature Importance")
            feature_importances_df = display_feature_importance(model, feature_names, ['Position_', 'Team_'])
        else:
            st.subheader("XGBoost Feature Importance")
            feature_importances_df = display_feature_importance(model, feature_names, ['Position_', 'Team_'])

        if feature_importances_df is None or feature_importances_df.empty:
            st.info("This model does not expose feature importances or no importances were computed.")
        else:
            st.dataframe(feature_importances_df)

            # Bar chart
            fig = plot_feature_importance(feature_importances_df, model_choice)
            st.pyplot(fig)

    elif page == "Salary Evaluation":
        st.header("Salary Evaluation")
        num_players = st.slider("Select number of players to display", min_value=5, max_value=20, value=10)
        overpaid, underpaid = identify_overpaid_underpaid(predictions_df, top_n=num_players)
        st.subheader(f"Top {num_players} Overpaid Players")
        st.dataframe(overpaid[['Player', 'Team', 'Salary', 'Predicted_Salary', 'Salary_Difference']])
        st.subheader(f"Top {num_players} Underpaid Players")
        st.dataframe(underpaid[['Player', 'Team', 'Salary', 'Predicted_Salary', 'Salary_Difference']])


    elif page == "Shot Chart Analysis":
        shot_chart_analysis()

    elif page == "Advanced Metrics Analysis":
        advanced_metrics_analysis()

    elif page == "Trade Impact Simulator":
        st.header("Trade Impact Simulator")
        formatted_season = convert_season_format(selected_season)
        trade_impact_simulator_app(formatted_season) #2023 or XXXX format is needed

if __name__ == "__main__":
    main()


