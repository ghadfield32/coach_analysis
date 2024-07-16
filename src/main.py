import numpy as np
import pandas as pd
from src.data.data_preprocessing import load_data, preprocess_data
from src.models.ridge_regression import train_ridge_model, evaluate_ridge_model
from src.models.svr import train_svr_model, evaluate_svr_model
from src.models.random_forest import train_random_forest_model, evaluate_random_forest_model
from src.models.stacking_ensemble import train_stacking_model, evaluate_stacking_model
from src.visualization.visualizations import (
    plot_correlation_matrix, plot_feature_importance, 
    plot_actual_vs_predicted, plot_residuals
)

def main():
    # Load and preprocess data
    raw_data = load_data('data/raw/nba_2022-23_all_stats_with_salary.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(raw_data)

    # Train and evaluate Ridge Regression model
    ridge_model = train_ridge_model(X_train, y_train)
    ridge_mse, ridge_r2 = evaluate_ridge_model(ridge_model, X_test, y_test)
    print(f"Ridge Regression - MSE: {ridge_mse:.2e}, R2: {ridge_r2:.4f}")

    # Train and evaluate SVR model
    svr_model = train_svr_model(X_train, y_train)
    svr_mse, svr_r2 = evaluate_svr_model(svr_model, X_test, y_test)
    print(f"SVR - MSE: {svr_mse:.2e}, R2: {svr_r2:.4f}")

    # Train and evaluate Random Forest model
    rf_model = train_random_forest_model(X_train, y_train)
    rf_mse, rf_r2 = evaluate_random_forest_model(rf_model, X_test, y_test)
    print(f"Random Forest - MSE: {rf_mse:.2e}, R2: {rf_r2:.4f}")

    # Train and evaluate Stacking Ensemble model
    stacking_model = train_stacking_model(X_train, y_train)
    stacking_mse, stacking_r2 = evaluate_stacking_model(stacking_model, X_test, y_test)
    print(f"Stacking Ensemble - MSE: {stacking_mse:.2e}, R2: {stacking_r2:.4f}")

    # Visualization
    plot_correlation_matrix(raw_data, ['PTS', 'VORP', 'GS', 'AST', 'TRB', '3P', 'STL', 'Age', 'BPM'])
    
    # Assuming random forest for feature importance and residual plots
    plot_feature_importance(rf_model, ['PTS', 'VORP', 'GS', 'AST', 'TRB', '3P', 'STL', 'Age', 'BPM'])
    plot_actual_vs_predicted(y_test, rf_model.predict(X_test), "Random Forest")
    plot_residuals(y_test, rf_model.predict(X_test), "Random Forest")

if __name__ == "__main__":
    main()
