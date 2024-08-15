
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, load_predictions, get_project_root
from data_preprocessor import handle_missing_values, feature_engineering, calculate_vorp_salary_ratio, cluster_career_trajectories
from predictor import load_model_and_scaler, make_predictions
from model_trainer import retrain_and_save_models
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from champ_percentile_ranks import calculate_percentiles, analyze_team_percentiles, get_champions
from data_loader import load_predictions, get_project_root
from trade_utils import analyze_trade, plot_trade_impact, analyze_trade_impact

def filter_by_position(df, selected_positions):
    if not selected_positions:
        return df
    return df[df['Position'].apply(lambda x: any(pos in x.split('-') for pos in selected_positions))]

def format_salary_df(df):
    formatted_df = df.copy()
    salary_columns = ['Salary', 'Predicted_Salary', 'Salary_Change']
    
    for col in salary_columns:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x/1e6:.2f}M")
    
    return formatted_df[['Player', 'Position', 'Age', 'Salary', 'Predicted_Salary', 'Salary_Change']]

def load_selected_model(model_name, use_inflated_data):
    try:
        model, scaler, selected_features = load_model_and_scaler(model_name, use_inflated_data)
        df = load_data(use_inflated_data)
        df = feature_engineering(df, use_inflated_data)
        df = handle_missing_values(df)
        
        X = df[selected_features]
        y = df['SalaryPct']
        X_scaled = scaler.transform(X)
        
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        salary_cap_column = 'Salary_Cap_Inflated' if use_inflated_data else 'Salary Cap'
        max_salary_cap = df[salary_cap_column].max()
        
        return model_name, model, mse, r2, selected_features, scaler, max_salary_cap
    except Exception as e:
        st.error(f"Error in load_selected_model: {str(e)}")
        raise

def find_best_model(use_inflated_data):
    root_dir = get_project_root()
    suffix = '_inflated' if use_inflated_data else ''
    
    with open(os.path.join(root_dir, 'data', 'models', f'best_model_name{suffix}.txt'), 'r') as f:
        best_model_name = f.read().strip()
    
    return load_selected_model(best_model_name, use_inflated_data)



def load_champions_data():
    root_dir = get_project_root()
    champions_file = os.path.join(root_dir, 'data', 'processed', 'nba_champions.csv')
    return pd.read_csv(champions_file)

RELEVANT_STATS = ['PTS', 'TRB', 'AST', 'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP']

def calculate_team_percentiles(team_players):
    team_percentiles = {}
    for stat in RELEVANT_STATS:
        if stat in team_players.columns:
            values = team_players[stat].values
            team_percentiles[stat] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'above_average': np.sum(values > np.mean(values)),
                'total_players': len(values)
            }
    return team_percentiles

def analyze_trade(players1, players2, predictions_df):
    group1_data = predictions_df[predictions_df['Player'].isin(players1)]
    group2_data = predictions_df[predictions_df['Player'].isin(players2)]
    
    group1_percentiles = calculate_team_percentiles(group1_data)
    group2_percentiles = calculate_team_percentiles(group2_data)
    
    return {
        'group1': {
            'players': group1_data,
            'percentiles': group1_percentiles,
            'salary_before': group1_data['Previous_Season_Salary'].sum(),
            'salary_after': group1_data['Predicted_Salary'].sum(),
        },
        'group2': {
            'players': group2_data,
            'percentiles': group2_percentiles,
            'salary_before': group2_data['Previous_Season_Salary'].sum(),
            'salary_after': group2_data['Predicted_Salary'].sum(),
        }
    }


def plot_salary_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df['Salary_M'], bins=30, kde=True, ax=ax1)
    ax1.set_title('Distribution of NBA Player Salaries (in Millions)')
    ax1.set_xlabel('Salary (in Millions)')
    sns.boxplot(y='Salary_M', x='Position', data=df, ax=ax2)
    ax2.set_title('NBA Player Salaries by Position (in Millions)')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Salary (in Millions)')
    plt.xticks(rotation=45)
    return fig

def plot_age_vs_salary(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Salary_M', hue='Position', data=df, ax=ax)
    ax.set_title('Age vs Salary (in Millions)')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def plot_vorp_vs_salary(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='VORP', y='Salary_M', hue='Position', size='Age', data=df, ax=ax)
    ax.set_title('VORP vs Salary')
    ax.set_xlabel('VORP')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def plot_career_clusters(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Age', y='Salary_M', hue='Cluster_Definition', style='Position', data=df, ax=ax)
    ax.set_title('Career Clusters: Age vs Salary')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def plot_salary_change_distribution(filtered_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(filtered_df['Salary_Change'] / 1e6, bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Predicted Salary Changes')
    ax.set_xlabel('Salary Change (in Millions)')
    ax.set_ylabel('Count')
    return fig

def plot_player_comparison(comparison_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df['Salary_M'] = comparison_df['Predicted_Salary'] / 1e6
    sns.barplot(x='Player', y='Salary_M', data=comparison_df, ax=ax)
    ax.set_title('Predicted Salaries for Selected Players')
    ax.set_xlabel('Player')
    ax.set_ylabel('Predicted Salary (in Millions)')
    plt.xticks(rotation=45, ha='right')
    return fig

def plot_performance_metrics_comparison(df, selected_players):
    metrics = ['PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP']
    metrics_df = df[df['Player'].isin(selected_players)][['Player'] + metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, metric in enumerate(metrics):
        sns.barplot(x='Player', y=metric, data=metrics_df, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f'{metric} Comparison')
        axes[i//3, i%3].set_xticklabels(axes[i//3, i%3].get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_salary_difference_distribution(filtered_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(filtered_df['Salary_Difference'] / 1e6, bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Salary Differences')
    ax.set_xlabel('Salary Difference (in Millions)')
    ax.set_ylabel('Count')
    return fig

def plot_category_analysis(avg_predictions, category):
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_predictions[['Salary', 'Predicted_Salary']].plot(kind='bar', ax=ax)
    ax.set_title(f'Average Actual vs Predicted Salary by {category}')
    ax.set_ylabel('Salary')
    plt.xticks(rotation=45)
    return fig

def plot_model_evaluation(df, y_pred, model_choice):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['SalaryPct'], y_pred, alpha=0.5)
    ax.plot([df['SalaryPct'].min(), df['SalaryPct'].max()], [df['SalaryPct'].min(), df['SalaryPct'].max()], 'r--', lw=2)
    ax.set_xlabel("Actual Salary Percentage")
    ax.set_ylabel("Predicted Salary Percentage")
    ax.set_title(f"Actual vs Predicted Salary Percentage - {model_choice}")
    return fig

def plot_feature_importance(feature_importance, model_choice):
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.plot(x='feature', y='importance', kind='bar', ax=ax)
    ax.set_title(f"Feature Importances - {model_choice}")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45, ha='right')
    return fig


def main():
    st.sidebar.title("Navigation")
    sections = ["Introduction", "Data Overview", "Exploratory Data Analysis", 
                "Advanced Analytics", "Salary Predictions", "Player Comparisons", 
                "Salary Comparison", "Analysis by Categories", "Model Selection and Evaluation",
                "Model Retraining", "Trade Analysis"]
    choice = st.sidebar.radio("Go to", sections)
    
    # Update model selection dropdown
    model_options = ['Random_Forest', 'Gradient_Boosting', 'Ridge_Regression', 'ElasticNet', 'SVR', 'Decision_Tree']
    selected_model = st.sidebar.selectbox("Select Model", model_options)

    use_inflated_data = st.sidebar.checkbox("Use Inflation Adjusted Salary Cap Data")
    st.sidebar.markdown("### All Salaries in Millions")

    # Load the selected model
    model_name, model, mse, r2, selected_features, scaler, max_salary_cap = load_selected_model(selected_model, use_inflated_data)

    # Display model info in sidebar
    st.sidebar.markdown(f"### Selected Model: {model_name}")
    st.sidebar.write(f"MSE: {mse:.4f}")
    st.sidebar.write(f"R²: {r2:.4f}")

    df = load_data(use_inflated_data)
    df = feature_engineering(df)
    df = handle_missing_values(df)

    seasons = df['Season'].unique()
    selected_season = st.sidebar.selectbox("Select Season", seasons)
    
    df = calculate_vorp_salary_ratio(df)
    df = cluster_career_trajectories(df)

    if model and selected_features and scaler:
        predictions = make_predictions(df, model, scaler, selected_features, selected_season, use_inflated_data, max_salary_cap)
    else:
        predictions = None
    
    
    if choice == "Introduction":
        st.title("Enhanced NBA Player Salary Analysis")
        st.write("Welcome to the NBA Salary Analysis and Prediction App! This project aims to provide comprehensive insights into NBA player salaries, advanced metrics, and future salary predictions based on historical data. Here's a detailed breakdown of the steps involved in creating this app:")

        st.subheader("Data Collection")
        
        st.write("### Salary Data")
        st.write("- **Sources**:")
        st.write("  - [Basketball Reference Salary Cap History](https://www.basketball-reference.com/contracts/salary-cap-history.html)")
        st.write("- **Description**: Data on the NBA salary cap from various seasons, along with maximum salary details for players based on years of service.")

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
        
        st.write("### Add Injury Data:")
        st.write("- **Source**: [Kaggle NBA Injury Stats 1951-2023](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023/data)")
        st.write("- **Description**: This dataset provides detailed statistics on NBA injuries from 1951 to 2023, allowing for analysis of player availability and its impact on performance and salaries.")

        st.subheader("Conclusion")

        st.write("This app provides a robust platform for analyzing NBA player salaries, understanding the factors influencing earnings, and predicting future salaries based on historical data and advanced metrics. Explore the app to gain insights into player performance, salary trends, and much more.")


    elif choice == "Data Overview":
        st.header("Data Overview")
        st.write("First few rows of the current season's dataset:")
        st.write(df[['Player', 'Season', 'Salary', 'GP', 'PTS', 'TRB', 'AST', 'Injured', 'Injury_Periods', 'Position', 'Age', 'Team', 'Years of Service', 'PER', 'WS', 'VORP', 'Salary Cap', 'Salary_Cap_Inflated']].head())
        st.write("\nFirst few rows of the predictions dataset:")
        st.write(predictions.head())
        
        if use_inflated_data:
            st.write("\nNote: This data uses inflated salary cap projections.")
        else:
            st.write("\nNote: This data uses the standard salary cap.")

    elif choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Salary Distribution")
        df['Salary_M'] = df['Salary'] / 1e6
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(df['Salary_M'], bins=30, kde=True, ax=ax1)
        ax1.set_title('Distribution of NBA Player Salaries (in Millions)')
        ax1.set_xlabel('Salary (in Millions)')
        sns.boxplot(y='Salary_M', x='Position', data=df, ax=ax2)
        ax2.set_title('NBA Player Salaries by Position (in Millions)')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Salary (in Millions)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Age vs Salary")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Age', y='Salary_M', hue='Position', data=df, ax=ax)
        ax.set_title('Age vs Salary (in Millions)')
        ax.set_xlabel('Age')
        ax.set_ylabel('Salary (in Millions)')
        st.pyplot(fig)

    elif choice == "Advanced Analytics":
        st.header("Advanced Analytics")

        st.subheader("VORP to Salary Ratio")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='VORP', y='Salary_M', hue='Position', size='Age', data=df, ax=ax)
        ax.set_title('VORP vs Salary')
        ax.set_xlabel('VORP')
        ax.set_ylabel('Salary (in Millions)')
        st.pyplot(fig)

        top_value_players = df.nlargest(10, 'VORP_Salary_Ratio')
        st.write("Top 10 Value Players (Highest VORP to Salary Ratio):")
        st.write(top_value_players[['Player', 'Position', 'Age', 'Salary_M', 'VORP', 'VORP_Salary_Ratio']])

        st.subheader("Career Trajectory Clusters")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='Age', y='Salary_M', hue='Cluster_Definition', style='Position', data=df, ax=ax)
        ax.set_title('Career Clusters: Age vs Salary')
        ax.set_xlabel('Age')
        ax.set_ylabel('Salary (in Millions)')
        st.pyplot(fig)

        st.write("Average Metrics by Cluster:")
        cluster_averages = df.groupby('Cluster_Definition')[['Age', 'Salary_M', 'PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP']].mean()
        st.write(cluster_averages)


    elif choice == "Salary Predictions":
        st.header("Salary Predictions")
        
        if model:
            predictions = make_predictions(df, model, scaler, selected_features, selected_season, use_inflated_data, max_salary_cap)
            
            st.sidebar.subheader("Filter by Position")
            unique_positions = sorted(set([pos for sublist in predictions['Position'].str.split('-') for pos in sublist]))
            selected_positions = st.sidebar.multiselect("Select positions", unique_positions, default=unique_positions)
            filtered_df = filter_by_position(predictions, selected_positions)
            
            st.write("### Top 10 Highest Predicted Salaries")
            st.write(format_salary_df(filtered_df.nlargest(10, 'Predicted_Salary')))
            
            st.subheader("Salary Change Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(filtered_df['Salary_Change'] / 1e6, bins=30, kde=True, ax=ax)
            ax.set_title('Distribution of Predicted Salary Changes')
            ax.set_xlabel('Salary Change (in Millions)')
            ax.set_ylabel('Count')
            st.pyplot(fig)

            if use_inflated_data:
                st.write("\nNote: These predictions are based on inflated salary cap projections.")
            else:
                st.write("\nNote: These predictions are based on the standard salary cap.")
        else:
            st.warning("No model found. Please select a valid model or retrain the models.")


            
    elif choice == "Player Comparisons":
        st.header("Player Comparisons")
        
        players = sorted(predictions['Player'].unique())
        selected_players = st.multiselect("Select players to compare", players)
        
        if selected_players:
            comparison_df = predictions[predictions['Player'].isin(selected_players)]
            st.write(format_salary_df(comparison_df))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            comparison_df['Salary_M'] = comparison_df['Predicted_Salary'] / 1e6
            sns.barplot(x='Player', y='Salary_M', data=comparison_df, ax=ax)
            ax.set_title('Predicted Salaries for Selected Players')
            ax.set_xlabel('Player')
            ax.set_ylabel('Predicted Salary (in Millions)')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            st.subheader("Performance Metrics Comparison")
            metrics = ['PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP']
            metrics_df = df[df['Player'].isin(selected_players)][['Player'] + metrics]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            for i, metric in enumerate(metrics):
                sns.barplot(x='Player', y=metric, data=metrics_df, ax=axes[i//3, i%3])
                axes[i//3, i%3].set_title(f'{metric} Comparison')
                axes[i//3, i%3].set_xticklabels(axes[i//3, i%3].get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

    elif choice == "Salary Comparison":
        st.header("Salary Comparison")

        st.sidebar.subheader("Filter by Position")
        unique_positions = sorted(set([pos for sublist in predictions['Position'].str.split('-') for pos in sublist]))
        selected_positions = st.sidebar.multiselect("Select positions", unique_positions, default=unique_positions)
        filtered_df = filter_by_position(predictions, selected_positions)

        filtered_df['Salary_Difference'] = filtered_df['Salary'] - filtered_df['Predicted_Salary']
        
        top_overpaid_count = st.sidebar.slider("Number of Top Overpaid Players to Display", min_value=1, max_value=50, value=10)
        top_underpaid_count = st.sidebar.slider("Number of Top Underpaid Players to Display", min_value=1, max_value=50, value=10)
        
        st.subheader("Overpaid vs Underpaid Players")
        st.write("### Top Overpaid Players")
        st.write(format_salary_df(filtered_df.nlargest(top_overpaid_count, 'Salary_Difference')))
        
        st.write("### Top Underpaid Players")
        st.write(format_salary_df(filtered_df.nsmallest(top_underpaid_count, 'Salary_Difference')))
        
        st.subheader("Salary Difference Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(filtered_df['Salary_Difference'] / 1e6, bins=30, kde=True, ax=ax)
        ax.set_title('Distribution of Salary Differences')
        ax.set_xlabel('Salary Difference (in Millions)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
    elif choice == "Analysis by Categories":
        st.header("Analysis by Categories")
        
        category = st.selectbox("Select Category", ['Position', 'Age', 'Team'])
        
        if category == 'Age':
            predictions['Age_Group'] = pd.cut(predictions['Age'], bins=[0, 25, 30, 35, 100], labels=['Under 25', '25-30', '30-35', 'Over 35'])
            category = 'Age_Group'
        
        avg_predictions = predictions.groupby(category)[['Salary', 'Predicted_Salary', 'Salary_Change']].mean()
        
        st.write(f"Average Salaries by {category}")
        st.write(avg_predictions)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        avg_predictions[['Salary', 'Predicted_Salary']].plot(kind='bar', ax=ax)
        ax.set_title(f'Average Actual vs Predicted Salary by {category}')
        ax.set_ylabel('Salary')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif choice == "Model Selection and Evaluation":
        st.header("Model Selection and Evaluation")
        
        models = ['Random_Forest', 'Gradient_Boosting', 'Ridge_Regression', 'ElasticNet', 'SVR', 'Decision_Tree']
        model_choice = st.selectbox("Select Model to Evaluate", models)
        
        if model_choice:
            try:
                # Load the model, scaler, and selected features
                model, scaler, selected_features = load_model_and_scaler(model_choice, use_inflated_data)
                
                # Ensure we have the correct features in our dataframe
                df_features = df[selected_features]
                
                # Scale the features
                X_scaled = scaler.transform(df_features)
                
                # Make predictions
                y_pred = model.predict(X_scaled)
                
                # Calculate evaluation metrics
                mse = mean_squared_error(df['SalaryPct'], y_pred)
                r2 = r2_score(df['SalaryPct'], y_pred)
                
                st.write(f"### Evaluation Metrics for {model_choice}")
                st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.write(f"R² Score: {r2:.4f}")
                
                # Create a scatter plot of actual vs predicted values
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df['SalaryPct'], y_pred, alpha=0.5)
                ax.plot([df['SalaryPct'].min(), df['SalaryPct'].max()], [df['SalaryPct'].min(), df['SalaryPct'].max()], 'r--', lw=2)
                ax.set_xlabel("Actual Salary Percentage")
                ax.set_ylabel("Predicted Salary Percentage")
                ax.set_title(f"Actual vs Predicted Salary Percentage - {model_choice}")
                st.pyplot(fig)
                
                # Display feature importances for tree-based models
                if model_choice in ['Random_Forest', 'Gradient_Boosting', 'Decision_Tree']:
                    feature_importance = pd.DataFrame({
                        'feature': selected_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.write("### Feature Importances")
                    st.write(feature_importance)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    feature_importance.plot(x='feature', y='importance', kind='bar', ax=ax)
                    ax.set_title(f"Feature Importances - {model_choice}")
                    ax.set_xlabel("Features")
                    ax.set_ylabel("Importance")
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                
            except FileNotFoundError as e:
                st.error("Error: Model file not found")
                st.error(str(e))
                st.error("Please make sure the model file exists and the name is correct.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error("Please check the logs for more details and ensure all required files are present.")

    # Update the Model Retraining section in your main() function
    elif choice == "Model Retraining":
        st.header("Model Retraining")
        
        if st.button("Retrain Models"):
            try:
                with st.spinner("Retraining models... This may take a while."):
                    best_model_name, best_model, evaluations, selected_features, scaler, max_salary_cap = retrain_and_save_models(use_inflated_data)
                
                st.success("Retraining completed successfully!")
                st.write(f"Best model: {best_model_name}")
                st.write("Model performance:")
                for model, metrics in evaluations.items():
                    st.write(f"{model}:")
                    st.write(f"  MSE: {metrics['MSE']:.4f}")
                    st.write(f"  R²: {metrics['R²']:.4f}")
                
                st.write("All models have been retrained and saved. The best model will be used for future predictions.")
                
                # Refresh the app to use the new models
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during model retraining: {str(e)}")
                st.error("Please check the logs for more details.")

    elif choice == "Trade Analysis":
        st.header("Trade Analysis")
        
        try:
            # Load the necessary data
            use_inflated_data_trade = st.checkbox("Use Inflation Adjusted Salary Cap Data", key="trade_analysis_inflated_data")
            
            predictions = load_predictions(use_inflated_data_trade)
            
            st.write("Debug: Predictions DataFrame shape:", predictions.shape)
            st.write("Debug: Predictions DataFrame columns:", predictions.columns)
            
            if 'Team' not in predictions.columns:
                st.error("The 'Team' column is missing from the predictions data. Please check your data loading process.")
                st.write("Debug: Available columns:", predictions.columns)
            else:
                st.write("Debug: 'Team' column exists in predictions DataFrame")
                st.write("Debug: Unique teams:", predictions['Team'].unique())
                st.write("Debug: Number of unique teams:", predictions['Team'].nunique())
                
                # Team filter
                all_teams = sorted(predictions['Team'].unique())
                st.write("Debug: all_teams list:", all_teams)
                
                if len(all_teams) > 0:
                    team1 = st.selectbox("Select Team 1", all_teams, key="trade_analysis_team1")
                    team2 = st.selectbox("Select Team 2", all_teams, index=1, key="trade_analysis_team2")
                    
                    predictions1 = predictions[predictions['Team'] == team1]
                    predictions2 = predictions[predictions['Team'] == team2]
                    
                    st.write(f"Debug: Number of players in {team1}:", len(predictions1))
                    st.write(f"Debug: Number of players in {team2}:", len(predictions2))
                    
                    st.subheader(f"Available Players for {team1}")
                    st.write(predictions1[['Player', 'Age', 'Previous_Season_Salary', 'Predicted_Salary', 'PTS', 'TRB', 'AST']])
                    
                    st.subheader(f"Available Players for {team2}")
                    st.write(predictions2[['Player', 'Age', 'Previous_Season_Salary', 'Predicted_Salary', 'PTS', 'TRB', 'AST']])
                    
                    # Player selection
                    players1 = st.multiselect(f"Select players from {team1}", predictions1['Player'].unique(), key="trade_analysis_players1")
                    players2 = st.multiselect(f"Select players from {team2}", predictions2['Player'].unique(), key="trade_analysis_players2")
                    
                    if st.button("Analyze Trade", key="trade_analysis_button"):
                        if not players1 or not players2:
                            st.warning("Please select players from both teams.")
                        else:
                            combined_predictions = pd.concat([predictions1, predictions2])
                            trade_analysis = analyze_trade(players1, players2, combined_predictions)
                            
                        
                        st.subheader("Trade Impact")
                        
                        for group, data in trade_analysis.items():
                            st.write(f"\n{group.upper()} Analysis:")
                            st.write(f"Total Salary Before: ${data['salary_before']/1e6:.2f}M")
                            st.write(f"Total Salary After: ${data['salary_after']/1e6:.2f}M")
                            st.write(f"Salary Change: ${(data['salary_after'] - data['salary_before'])/1e6:.2f}M")
                            
                            st.write("\nPlayer Details:")
                            st.write(data['players'][['Player', 'Age', 'Previous_Season_Salary', 'Predicted_Salary', 'Salary_Change', 'PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP']])
                            
                            st.write("\nTeam Percentiles:")
                            for stat in RELEVANT_STATS:
                                if stat in data['percentiles']:
                                    st.write(f"{stat}: {data['percentiles'][stat]['mean']:.2f}")
                        
                        st.subheader("Salary Comparison")
                        group1_trade_salary = trade_analysis['group1']['salary_after']
                        group2_trade_salary = trade_analysis['group2']['salary_after']
                        salary_difference = abs(group1_trade_salary - group2_trade_salary)
                        
                        st.write(f"{team1} is trading ${group1_trade_salary/1e6:.2f}M in salary")
                        st.write(f"{team2} is trading ${group2_trade_salary/1e6:.2f}M in salary")
                        st.write(f"Salary difference: ${salary_difference/1e6:.2f}M")
                        
                        if salary_difference > 5e6:  # Assuming a 5 million threshold for salary matching
                            st.warning("The salaries in this trade are not well-matched. This may not be a valid trade under NBA rules.")
                        else:
                            st.success("The salaries in this trade are well-matched.")
                        
                        # Visualize the trade impact
                        fig = plot_trade_impact(trade_analysis, team1, team2)
                        st.pyplot(fig)

            else:
                st.error("No teams found in the predictions data. Please check your data loading process.")

    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}")
        st.error("Please make sure the predictions file exists in the correct location.")
    except KeyError as e:
        st.error(f"Error: {str(e)}")
        st.error("Please check your data files and ensure they contain all required columns.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the data and try again.")
        st.write("Debug: Exception details:", str(e))

if __name__ == "__main__":
    main()
