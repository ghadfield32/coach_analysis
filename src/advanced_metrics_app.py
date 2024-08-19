
import streamlit as st
import pandas as pd
from advanced_metrics import plot_career_clusters, plot_injury_risk_vs_salary, plot_availability_vs_salary, plot_vorp_vs_salary, table_metric_salary, display_top_10_salary_per_metric, cluster_players_specialized

# Function to display top 10 Salary per metric including WS
def display_top_10_salary_per_metric_with_ws(df, metric_col):
    # Sort by the specified metric and display the top 10 with Player names and WS
    return df.sort_values(by=metric_col, ascending=False).head(10)[['Player', metric_col, 'WS', 'Salary']]

# Streamlit app
def main():
    st.title("NBA Advanced Metrics and Salary Analysis")
    
    # Load the data
    data = pd.read_csv('data/processed/nba_player_data_final_inflated.csv')
    
    # Add a dropdown to select the season
    seasons = sorted(data['Season'].unique())
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

if __name__ == "__main__":
    main()
