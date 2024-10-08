
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from .data_loader_preprocessor import preprocessed_datasets
from .model_trainer import load_and_preprocess_data, train_and_save_models, evaluate_models  # Add this line
from .model_predictor import make_predictions
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ['Position_', 'Team_']

def load_evaluation_metrics(model_save_path):
    """Load saved evaluation metrics."""
    eval_save_path = f"{model_save_path}/evaluation_results.pkl"
    if os.path.exists(eval_save_path):
        eval_results = joblib.load(eval_save_path)
        return eval_results
    else:
        print("Evaluation results not found.")
        return None

def check_or_train_model(file_path, model_save_path, season_year):
    logger.debug(f"Received season_year: {season_year}, Type: {type(season_year)}")

    # Convert season_year to an integer if it's a string
    if isinstance(season_year, str):
        season_year = int(season_year)
        logger.debug(f"Converted season_year to int: {season_year}, Type: {type(season_year)}")

    predictions_file_path = f'{model_save_path}/rf_predictions.csv'
    if os.path.exists(predictions_file_path):
        logger.debug(f"Predictions file found for {season_year}.")
        predictions_df = pd.read_csv(predictions_file_path)
    else:
        logger.debug(f"Predictions not available for {season_year}. Training the model now...")
        
        # Train and predict
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, season_year, model_save_path)
        train_and_save_models(X_train, y_train, model_save_path)
        evaluate_models(X_test, y_test, model_save_path)

        # Generate predictions
        rf_final_df, xgb_final_df = make_predictions(file_path, season_year, model_save_path)
        predictions_df = pd.concat([rf_final_df, xgb_final_df], axis=1)

    return predictions_df



def display_model_metrics(model_save_path):
    """Display saved model performance metrics for both Random Forest and XGBoost."""
    eval_results = load_evaluation_metrics(model_save_path)

    if eval_results:
        print("\nModel Performance Metrics:")
        print(f"Random Forest RMSE: {eval_results['rf_rmse']:.4f}")
        print(f"Random Forest MAE: {eval_results['rf_mae']:.4f}")
        print(f"Random Forest R²: {eval_results['rf_r2']:.4f}")
        print(f"Random Forest MSE: {eval_results['rf_mse']:.4f}")
        
        print(f"\nXGBoost RMSE: {eval_results['xgb_rmse']:.4f}")
        print(f"XGBoost MAE: {eval_results['xgb_mae']:.4f}")
        print(f"XGBoost R²: {eval_results['xgb_r2']:.4f}")
        print(f"XGBoost MSE: {eval_results['xgb_mse']:.4f}")
    else:
        print("No evaluation metrics found.")

def filter_categorical_features(importance_df, categorical_features):
    """Filter out categorical features from the importance dataframe."""
    filtered_df = importance_df[~importance_df['Feature'].str.startswith(tuple(categorical_features))]
    return filtered_df

def display_feature_importance(model, feature_names, categorical_features):
    """Displays feature importance for the selected model, filtering out categorical features."""
    if hasattr(model, "feature_importances_"):
        n_features = len(model.feature_importances_)
        print(f"Number of features in model: {n_features}")
        
        # Create the DataFrame of feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names[:n_features],  # Adjust if feature names mismatch
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        # Filter out categorical features
        filtered_importance_df = filter_categorical_features(importance_df, categorical_features)
        return filtered_importance_df
    else:
        print("This model does not support feature importance visualization.")
        return None

def plot_feature_importance(feature_importances_df, model_name):
    """Function to plot the feature importance as a bar chart."""
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'{model_name} Feature Importance')
    plt.gca().invert_yaxis()  # Most important feature at the top
    return plt

def identify_overpaid_underpaid(predictions_df, top_n=10):
    logger.debug(f"Checking Salary and Predicted_Salary columns")
    
    # Check for duplicate columns and remove them
    predictions_df = predictions_df.loc[:, ~predictions_df.columns.duplicated()]
    
    logger.debug(f"Salary type: {type(predictions_df['Salary'].iloc[0])}, Predicted_Salary type: {type(predictions_df['Predicted_Salary'].iloc[0])}")
    logger.debug(f"First few Salary values: {predictions_df['Salary'].head()}")
    logger.debug(f"First few Predicted_Salary values: {predictions_df['Predicted_Salary'].head()}")
    
    # Calculate salary differences
    predictions_df['Salary_Difference'] = predictions_df['Salary'] - predictions_df['Predicted_Salary']
    
    # Identify overpaid and underpaid players
    overpaid = predictions_df[predictions_df['Salary_Difference'] > 0].sort_values('Salary_Difference', ascending=False).head(top_n)
    underpaid = predictions_df[predictions_df['Salary_Difference'] < 0].sort_values('Salary_Difference').head(top_n)
    
    logger.debug(f"Top overpaid: {overpaid[['Player', 'Salary', 'Predicted_Salary', 'Salary_Difference']].head()}")
    logger.debug(f"Top underpaid: {underpaid[['Player', 'Salary', 'Predicted_Salary', 'Salary_Difference']].head()}")
    
    return overpaid, underpaid



def display_overpaid_underpaid(predictions_df, top_n=10):
    """Display top overpaid and underpaid players."""
    overpaid, underpaid = identify_overpaid_underpaid(predictions_df, top_n)

    print(f"\nTop {top_n} Overpaid Players:")
    print(overpaid[['Player', 'Team', 'Salary', 'Predicted_Salary', 'Salary_Difference']])

    print(f"\nTop {top_n} Underpaid Players:")
    print(underpaid[['Player', 'Team', 'Salary', 'Predicted_Salary', 'Salary_Difference']])




import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distribution(data, feature):
    """Plot the distribution of a selected feature."""
    logger.debug(f"Plotting distribution for feature: {feature}")
    fig, ax = plt.subplots()
    data[feature].hist(ax=ax, bins=20)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    return fig


def plot_correlation_heatmap(data):
    """Plot a correlation heatmap of the numerical features in the dataset."""
    logger.debug("Plotting correlation heatmap for numeric features.")
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig

def test_data_analysis_functions():
    """Test the data analysis utility functions."""
    # Load some test data
    file_path = '../data/processed/nba_player_data_final_inflated.csv'
    cleaned_data, engineered_data, pipeline_data, columns_to_re_add = preprocessed_datasets(file_path)

    # Select a feature to test the distribution plot
    feature = 'SalaryPct'  # Choose a numerical feature available in your dataset
    logger.debug(f"Testing feature distribution for: {feature}")
    
    # Test the feature distribution function
    fig = plot_feature_distribution(pipeline_data, feature)
    fig.show()  # Show the plot to ensure it's working correctly

    # Test the correlation heatmap function
    logger.debug("Testing correlation heatmap plot.")
    fig = plot_correlation_heatmap(pipeline_data)
    fig.show()  # Show the heatmap plot



def main_test_function():
    """Main function to test all utility functions."""
    file_path = '../data/processed/nba_player_data_final_inflated.csv'
    season_year = '2021'  # initially a string
    logger.debug(f"Original season_year: {season_year}, Type: {type(season_year)}")

    # Convert to integer if necessary
    if isinstance(season_year, str):
        season_year = int(season_year)
        logger.debug(f"Converted season_year to int: {season_year}, Type: {season_year}")

    model_save_path = f'../data/models/season_{season_year}'

    # Test check_or_train_model
    predictions_df = check_or_train_model(file_path, model_save_path, season_year)
    logger.debug(f"Predictions DataFrame:\n{predictions_df.head()}")

    # Test display_model_metrics
    display_model_metrics(model_save_path)

    # Load a model for testing feature importance
    rf_model_path = f'{model_save_path}/best_rf_model.pkl'
    rf_model = joblib.load(rf_model_path)
    feature_names_path = f'{model_save_path}/feature_names.pkl'
    feature_names = joblib.load(feature_names_path)

    # Test display_feature_importance with filtering categorical features
    feature_importances_df = display_feature_importance(rf_model, feature_names, CATEGORICAL_FEATURES)
    
    # Test plot_feature_importance
    plot = plot_feature_importance(feature_importances_df, "Random Forest")
    plot.show()

    # Test display_overpaid_underpaid
    display_overpaid_underpaid(predictions_df)

    # Test the data analysis functions
    test_data_analysis_functions()

if __name__ == "__main__":
    main_test_function()


