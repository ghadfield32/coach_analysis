
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Advanced Analytics functions
def plot_career_clusters(df):
    if 'Cluster_Definition' not in df.columns:
        raise ValueError("The 'Cluster_Definition' column is missing from the DataFrame.")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Age', y='Salary', hue='Cluster_Definition', style='Position', data=df, ax=ax)
    ax.set_title('Career Clusters: Age vs Salary')
    ax.set_xlabel('Age')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def plot_injury_risk_vs_salary(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Injury_Risk', y='Salary', data=df, ax=ax)
    ax.set_title('Injury Risk vs Salary')
    ax.set_xlabel('Injury Risk')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def plot_availability_vs_salary(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='GP', y='Salary', hue='Injury_Risk', data=df, ax=ax)
    ax.set_title('Availability (Games Played) vs Salary')
    ax.set_xlabel('Games Played')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def plot_vorp_vs_salary(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='VORP', y='Salary', hue='Position', data=df, ax=ax)
    ax.set_title('VORP vs Salary')
    ax.set_xlabel('VORP')
    ax.set_ylabel('Salary (in Millions)')
    return fig

def table_metric_salary(df, min_ws_threshold=0.1):
    metrics = ['OWS', 'DWS', 'WS', 'VORP']
    table = df[metrics + ['Salary', 'Player']].copy()
    
    # Apply the minimum threshold for WS to avoid large numbers
    table['WS'] = table['WS'].apply(lambda x: max(x, min_ws_threshold))
    table['OWS'] = table['OWS'].apply(lambda x: max(x, min_ws_threshold))
    table['DWS'] = table['DWS'].apply(lambda x: max(x, min_ws_threshold))
    
    # Calculate salary per metric
    table['Salary_per_WS'] = table['Salary'] / table['WS']
    table['Salary_per_VORP'] = table['Salary'] / (table['VORP'] + 1e-5)  # Avoid division by zero
    table['Salary_per_OWS'] = table['Salary'] / table['OWS']
    table['Salary_per_DWS'] = table['Salary'] / table['DWS']
    
    return table


def display_top_10_salary_per_metric(df, metric_col):
    # Sort by the specified metric and display the top 10 with Player names
    return df.sort_values(by=metric_col, ascending=False).head(10)[['Player', metric_col, 'Salary']]

# Function to calculate percentiles for relevant metrics
def calculate_percentiles(df, metrics):
    for metric in metrics:
        df[f'{metric}_percentile'] = df[metric].rank(pct=True)
    return df

# Function to cluster players with specialization based on percentiles
def cluster_players_specialized(df, n_clusters=7):
    df = df.copy()
    
    # Metrics to calculate percentiles for
    relevant_metrics = [
        '3P%', '2P%', 'MP', 'STL', 'BLK', 'AST', 'TRB', 'PTS', 'Salary'
    ]
    
    # Calculate percentiles within the season
    df = calculate_percentiles(df, relevant_metrics)
    
    # Calculate salary per minute
    df['Salary_per_min'] = df['Salary'] / df['MP']
    
    # Drop rows with missing values in these columns
    df_numeric = df[[f'{metric}_percentile' for metric in relevant_metrics] + ['Salary_per_min']].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    print(f"Clustering players using columns: {df_numeric.columns}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    
    # Fit the model and predict clusters
    clusters = kmeans.fit_predict(df_scaled)
    
    # Create a new DataFrame with the relevant metrics to maintain the same index as the original DataFrame
    df_clustered = df.loc[df_numeric.index]
    df_clustered['Cluster'] = clusters
    
    # Mapping cluster numbers to descriptive names with conditions
    cluster_names = {
        0: "3-Point Specialist",
        1: "Inside Scorer",
        2: "Bench Player",
        3: "Defensive Specialist",
        4: "Playmaker",
        5: "Rebounder",
        6: "All-Rounder"
    }
    
    # Apply conditions to enhance cluster definitions based on percentiles
    df_clustered['Cluster_Definition'] = df_clustered['Cluster'].map(cluster_names)
    
    # High 3P% percentile and low 2P% percentile for 3-Point Specialist
    df_clustered.loc[(df_clustered['3P%_percentile'] > 0.75) & (df_clustered['2P%_percentile'] < 0.50), 'Cluster_Definition'] = "3-Point Specialist"
    
    # High 2P% percentile and low 3P% percentile for Inside Scorer
    df_clustered.loc[(df_clustered['2P%_percentile'] > 0.75) & (df_clustered['3P%_percentile'] < 0.50), 'Cluster_Definition'] = "Inside Scorer"
    
    # Low MP percentile for Bench Player, adjusted by salary per minute
    df_clustered.loc[(df_clustered['MP_percentile'] < 0.25) & (df_clustered['Salary_per_min'] > df_clustered['Salary_per_min'].median()), 'Cluster_Definition'] = "Bench Player"
    
    # High STL or BLK percentiles for Defensive Specialist
    df_clustered.loc[(df_clustered['STL_percentile'] > 0.75) | (df_clustered['BLK_percentile'] > 0.75), 'Cluster_Definition'] = "Defensive Specialist"
    
    # High AST percentile for Playmaker
    df_clustered.loc[(df_clustered['AST_percentile'] > 0.75), 'Cluster_Definition'] = "Playmaker"
    
    # High TRB percentile and below-average PTS percentile for Rebounder
    average_pts_percentile = df_clustered['PTS_percentile'].mean()
    df_clustered.loc[(df_clustered['TRB_percentile'] > 0.75) & (df_clustered['PTS_percentile'] < average_pts_percentile), 'Cluster_Definition'] = "Rebounder"
    
    # Check the results
    if 'Cluster_Definition' not in df_clustered.columns:
        raise ValueError("Failed to create 'Cluster_Definition' during clustering.")
    
    # Merging the cluster information back to the original DataFrame
    df = df.merge(df_clustered[['Cluster', 'Cluster_Definition']], left_index=True, right_index=True, how='left')
    
    return df


def display_top_10_salary_per_metric_with_ws(df, metric_col):
    # Sort by the specified metric and display the top 10 with Player names and WS
    return df.sort_values(by=metric_col, ascending=False).head(10)[['Player', metric_col, 'WS', 'Salary']]


# Main function to test advanced metrics
def main():
    # Sample data loading
    data = pd.read_csv('../data/processed/nba_player_data_final_inflated.csv')

    # Clustering players with specialized logic based on percentiles
    data = cluster_players_specialized(data, n_clusters=7)

    # Check if Cluster_Definition exists
    if 'Cluster_Definition' not in data.columns:
        print("Cluster_Definition was not added correctly to the DataFrame.")
        return
    
    # Testing Career Clusters plot
    fig1 = plot_career_clusters(data)
    plt.show(fig1)

    # Plot Injury Risk vs Salary
    fig2 = plot_injury_risk_vs_salary(data)
    plt.show(fig2)

    # Plot Availability vs Salary
    fig3 = plot_availability_vs_salary(data)
    plt.show(fig3)

    # Plot VORP vs Salary
    fig4 = plot_vorp_vs_salary(data)
    plt.show(fig4)

    # Display metric/salary table
    metric_salary_table = table_metric_salary(data)
    print("Metric/Salary Table:")
    print(metric_salary_table.head())

    # Display top 10 by Salary per WS
    top_10_salary_per_ws = display_top_10_salary_per_metric(metric_salary_table, 'Salary_per_WS')
    print("Top 10 Salary per WS:")
    print(top_10_salary_per_ws)

    # Display top 10 by Salary per VORP
    top_10_salary_per_vorp = display_top_10_salary_per_metric(metric_salary_table, 'Salary_per_VORP')
    print("Top 10 Salary per VORP:")
    print(top_10_salary_per_vorp)

    # Display top 10 by Salary per OWS
    top_10_salary_per_ows = display_top_10_salary_per_metric(metric_salary_table, 'Salary_per_OWS')
    print("Top 10 Salary per OWS:")
    print(top_10_salary_per_ows)

    # Display top 10 by Salary per DWS
    top_10_salary_per_dws = display_top_10_salary_per_metric(metric_salary_table, 'Salary_per_DWS')
    print("Top 10 Salary per DWS:")
    print(top_10_salary_per_dws)

if __name__ == "__main__":
    main()
