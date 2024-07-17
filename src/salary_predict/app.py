
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
        model, scaler, selected_features = load_model_and_scaler(model_name, use_inflated_data)
        df = load_data(use_inflated_data)
        df = feature_engineering(df)
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

# Add this function to find the best model
def find_best_model(use_inflated_data):
    root_dir = get_project_root()
    suffix = '_inflated' if use_inflated_data else ''
    
    with open(os.path.join(root_dir, 'data', 'models', f'best_model_name{suffix}.txt'), 'r') as f:
        best_model_name = f.read().strip()
    
    return load_selected_model(best_model_name, use_inflated_data)

def main():
    st.sidebar.title("Navigation")
    sections = ["Introduction", "Data Overview", "Exploratory Data Analysis", 
                "Advanced Analytics", "Salary Predictions", "Player Comparisons", 
                "Salary Comparison", "Analysis by Categories", "Model Selection and Evaluation",
                "Model Retraining"]
    choice = st.sidebar.radio("Go to", sections)
    
    # Add model selection dropdown
    model_options = ['Random_Forest', 'Gradient_Boosting', 'Ridge_Regression', 'ElasticNet', 'SVR', 'Decision_Tree', 'Best_Model']
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=model_options.index('Best_Model'))

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
        st.write(df.head())
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


if __name__ == "__main__":
    main()
