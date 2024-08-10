
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Import functions from other modules
from data_loader_preprocessor import load_data, format_season, clean_data, engineer_features, encode_data
from model_trainer import train_and_save_models, evaluate_models
from model_predictor import predict
from trade_utils import analyze_two_team_trade, get_champions


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

# Trade impact display function
def display_trade_impact(result, team1, team2):
    for team_abbr in [team1, team2]:
        st.subheader(f"{team_abbr} Trade Impact")
        
        team_data = result[team_abbr]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Salary", f"${team_data['current_salary']:,.2f}")
        col2.metric("Salary After Trade", f"${team_data['new_salary']:,.2f}")
        col3.metric("Salary Difference", f"${team_data['new_salary'] - team_data['current_salary']:,.2f}")
        
        st.subheader("Stat Comparisons")
        
        # Create a DataFrame for the main stat comparisons
        comparison_data = []
        for stat, values in team_data['comparison'].items():
            comparison_data.append({
                'Stat': stat,
                'Current': f"{values['Current']:.2f} ({values['Current Percentile']:.1f}%ile)",
                'After Trade': f"{values['After Trade']:.2f} ({values['After Trade Percentile']:.1f}%ile)",
                'Champion Average': f"{values['Champ Average']:.2f}",
                'League Average': f"{values['League Average']:.2f}",
                'Change vs League': f"{values['After Trade vs League'] - values['Current vs League']:.2f}",
                'Change vs Champ': f"{values['After Trade vs Champ'] - values['Current vs Champ']:.2f}"
            })
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        st.subheader("Percentile Counts")
        percentile_data = []
        for stat, values in team_data['comparison'].items():
            stat_data = {'Stat': stat}
            for percentile in [99, 98, 97, 96, 95, 90, 75, 50]:
                percentile_key = f"Top {100-percentile}%"
                stat_data[f"Current {percentile_key}"] = values['Current Percentile Counts'][percentile_key]
                stat_data[f"After Trade {percentile_key}"] = values['After Trade Percentile Counts'][percentile_key]
                stat_data[f"Champion {percentile_key}"] = values['Champ Percentile Counts'][percentile_key]
            percentile_data.append(stat_data)
        
        percentile_df = pd.DataFrame(percentile_data)
        st.table(percentile_df)
        
        st.markdown("---")

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


# Main Streamlit app
def main():
    st.set_page_config(page_title="NBA Salary Prediction and Trade Analysis", layout="wide")
    st.title("NBA Salary Prediction and Trade Analysis")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Model Results", "Salary Evaluation", "Trade Analysis"])

    # Load base data
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
        st.bar_chart(feature_importance.set_index('feature'))


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

    elif page == "Trade Analysis":
        st.header("Trade Analysis")
        st.write("""
        Analyze potential trades and their impact on team statistics and salary cap.
        For more information on trade rules, visit: [NBA Trade Rules](https://www.hoopsrumors.com/2023/09/salary-matching-rules-for-trades-during-2023-24-season.html)
        """)

        # Team selection
        teams = sorted(predictions_df['Team'].unique())
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Team 1", teams)
        with col2:
            team2 = st.selectbox("Select Team 2", teams, index=1)

        # Player selection
        team1_players = predictions_df[predictions_df['Team'] == team1]['Player'].tolist()
        team2_players = predictions_df[predictions_df['Team'] == team2]['Player'].tolist()

        col1, col2 = st.columns(2)
        with col1:
            players_leaving_team1 = st.multiselect(f"Select players leaving {team1}", team1_players)
        with col2:
            players_leaving_team2 = st.multiselect(f"Select players leaving {team2}", team2_players)

        if st.button("Analyze Trade"):
            champions = get_champions(selected_season - 10, selected_season - 1)
            result = analyze_two_team_trade(team1, team2, players_leaving_team1, players_leaving_team2, predictions_df, champions)
            
            if result:
                display_trade_impact(result, team1, team2)
            else:
                st.error("Trade analysis failed. Please check your selections.")

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

if __name__ == "__main__":
    main()
