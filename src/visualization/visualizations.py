import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_matrix(df, features):
    corr = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of Selected Features')
    plt.savefig('notebooks/correlation_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('notebooks/feature_importance.png')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    plt.title(f'{model_name} - Actual vs Predicted Salaries')
    plt.tight_layout()
    plt.savefig(f'notebooks/{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png')
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Salary')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residual Plot')
    plt.tight_layout()
    plt.savefig(f'notebooks/{model_name.lower().replace(" ", "_")}_residuals.png')
    plt.close()
