import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, important_features=None):
    """
    Plots a correlation heatmap of the given DataFrame.
    
    Parameters:
    - df: DataFrame containing the data.
    - important_features: List of important features to include in the heatmap. 
                          If None, all features are used.
                          
    Returns:
    - fig: The matplotlib figure object containing the heatmap.
    """
    if important_features is not None:
        df = df[important_features]
    
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Important Features")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

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
