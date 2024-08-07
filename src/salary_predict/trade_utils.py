import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from data_loader import load_predictions, get_project_root, load_data
# from data_preprocessor import calculate_percentiles

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

def plot_trade_impact(trade_analysis, team1, team2):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(RELEVANT_STATS))
    width = 0.35
    
    group1_stats = [trade_analysis['group1']['percentiles'].get(stat, {}).get('mean', 0) for stat in RELEVANT_STATS]
    group2_stats = [trade_analysis['group2']['percentiles'].get(stat, {}).get('mean', 0) for stat in RELEVANT_STATS]
    
    ax.bar([i - width/2 for i in x], group1_stats, width, label=team1)
    ax.bar([i + width/2 for i in x], group2_stats, width, label=team2)
    
    ax.set_ylabel('Value')
    ax.set_title('Trade Impact on Team Stats')
    ax.set_xticks(x)
    ax.set_xticklabels(RELEVANT_STATS, rotation=45, ha='right')
    ax.legend()
    
    return fig

def simulate_trade(team_players, new_player_stats):
    # Remove a player (e.g., the lowest-ranked player) to make room for the new player
    team_players = team_players.sort_values('PTS', ascending=True).iloc[1:]
    
    # Add the new player to the team
    new_team = pd.concat([team_players, new_player_stats], ignore_index=True)
    
    return new_team

def compare_percentiles(current_percentiles, simulated_percentiles, champ_percentiles):
    comparison = {}
    for stat in RELEVANT_STATS:
        comparison[stat] = {
            'Current': current_percentiles[stat]['mean'],
            'With New Player': simulated_percentiles[stat]['mean'],
            'Champ Average': champ_percentiles[stat]['mean'],
            'Current Diff': current_percentiles[stat]['mean'] - champ_percentiles[stat]['mean'],
            'Simulated Diff': simulated_percentiles[stat]['mean'] - champ_percentiles[stat]['mean']
        }
    return comparison

def analyze_trade_impact(team_abbr, new_player_name, season, predictions_df, min_minutes_per_game=10, min_games=20):
    # Get team players
    team_players = predictions_df[predictions_df['Team'] == team_abbr]

    # Get new player's stats
    new_player_stats = predictions_df[predictions_df['Player'] == new_player_name]

    if new_player_stats.empty:
        print(f"Could not find {new_player_name}'s stats.")
        return

    # Calculate current team percentiles
    current_team_percentiles = calculate_team_percentiles(team_players)

    # Simulate trade
    team_with_new_player = simulate_trade(team_players, new_player_stats)

    # Calculate simulated team percentiles
    simulated_team_percentiles = calculate_team_percentiles(team_with_new_player)

    # Get championship team percentiles (you may need to adapt this part)
    champ_percentiles = calculate_team_percentiles(predictions_df)

    # Compare percentiles
    comparison = compare_percentiles(current_team_percentiles, simulated_team_percentiles, champ_percentiles)

    return comparison

def debug_trade_analysis():
    print("Debugging Trade Analysis Functionality")
    
    # Load predictions data
    use_inflated_data = False
    predictions = load_predictions(use_inflated_data)
    
    print("\nPredictions DataFrame Info:")
    print(predictions.info())
    
    print("\nUnique Teams:")
    print(predictions['Team'].unique())
    
    if len(predictions['Team'].unique()) < 2:
        print("\nWARNING: Less than 2 unique teams found in the dataset.")
        return
    
    # Select two teams for testing
    team1, team2 = predictions['Team'].unique()[:2]
    
    print(f"\nAnalyzing trade between {team1} and {team2}")
    
    # Select players from each team
    players1 = predictions[predictions['Team'] == team1]['Player'].head(2).tolist()
    players2 = predictions[predictions['Team'] == team2]['Player'].head(2).tolist()
    
    print(f"Players from {team1}: {players1}")
    print(f"Players from {team2}: {players2}")
    
    # Perform trade analysis
    trade_analysis = analyze_trade(players1, players2, predictions)
    
    print("\nTrade Analysis Results:")
    for group, data in trade_analysis.items():
        print(f"\n{group.upper()}:")
        print(f"Salary Before: ${data['salary_before']/1e6:.2f}M")
        print(f"Salary After: ${data['salary_after']/1e6:.2f}M")
        print(f"Salary Change: ${(data['salary_after'] - data['salary_before'])/1e6:.2f}M")
        
        print("\nPercentiles:")
        for stat, values in data['percentiles'].items():
            print(f"{stat}: Mean = {values['mean']:.2f}, Std = {values['std']:.2f}")
    
    # Plot trade impact
    fig = plot_trade_impact(trade_analysis, team1, team2)
    plt.show()

    # Test trade impact analysis
    print("\nTesting Trade Impact Analysis")
    new_player_name = predictions[predictions['Team'] != team1]['Player'].iloc[0]
    season = predictions['Season'].max()
    
    impact_comparison = analyze_trade_impact(team1, new_player_name, season, predictions)
    
    print(f"\nTrade Impact Analysis for {team1} acquiring {new_player_name}:")
    print("{:<10} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
        "Stat", "Current", f"With {new_player_name}", "Champ Average", "Current Diff", "Simulated Diff"))
    print("-" * 90)
    for stat, values in impact_comparison.items():
        print("{:<10} {:<15.2f} {:<15.2f} {:<20.2f} {:<15.2f} {:<15.2f}".format(
            stat, 
            values['Current'], 
            values['With New Player'], 
            values['Champ Average'],
            values['Current Diff'],
            values['Simulated Diff']
        ))

if __name__ == "__main__":
    debug_trade_analysis()
