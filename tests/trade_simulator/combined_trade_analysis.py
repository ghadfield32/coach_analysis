
import pandas as pd
from percentile_count_trade_impact import get_champion_percentiles, generate_comparison_tables
from overall_team_trade_impact import trade_impact_analysis
from nba_rules_trade_impact import analyze_trade_scenario
from shot_chart.nba_efficiency import calculate_compatibility_between_players
from shot_chart.nba_shots import fetch_shots_for_multiple_players

def analyze_player_salaries(players, predictions_df):
    """Analyze if the selected players are overpaid or underpaid based on predicted salaries."""
    player_salary_analysis = []

    for player in players:
        player_data = predictions_df[predictions_df['Player'] == player]
        if not player_data.empty:
            actual_salary = player_data['Salary'].values[0]
            salary_cap = player_data['Salary_Cap_Inflated'].values[0]
            predicted_salary = player_data['Predicted_Salary'].values[0] * salary_cap
            difference = actual_salary - predicted_salary
            status = "Overpaid" if difference > 0 else "Underpaid" if difference < 0 else "Fairly Paid"
            player_salary_analysis.append({
                'Player': player,
                'Actual Salary': actual_salary,
                'Predicted Salary': predicted_salary,
                'Difference': difference,
                'Status': status
            })

    return pd.DataFrame(player_salary_analysis)

def combined_trade_analysis(team_a_name, team_b_name, selected_players_team_a, selected_players_team_b, 
                            trade_date, champion_seasons, trade_season, relevant_stats, predictions_df, debug=False):
    """
    Perform a comprehensive analysis of the trade impact on the involved teams.
    """
    # Step 1: Analyze the trade scenario using NBA salary matching rules
    trade_valid, trade_analysis_debug = analyze_trade_scenario(
        selected_players_team_a, selected_players_team_b, predictions_df=predictions_df, season=trade_season, debug=debug
    )

    # Step 2: Fetch champion percentiles and calculate averages
    average_top_percentiles_df = get_champion_percentiles(champion_seasons, debug)

    # Step 3: Generate comparison tables before and after the trade for the trade season
    team_a_comparison_table, team_b_comparison_table = generate_comparison_tables(
        trade_season, team_a_name, team_b_name, selected_players_team_a, selected_players_team_b, 
        average_top_percentiles_df, debug
    )

    # Step 4: Perform the trade impact analysis for the trade season
    traded_players = {player: team_b_name for player in selected_players_team_a}
    traded_players.update({player: team_a_name for player in selected_players_team_b})

    comparison_tables = trade_impact_analysis(
        start_season=trade_season, end_season=trade_season, trade_date=trade_date, 
        traded_players=traded_players, 
        team_a_name=team_a_name, team_b_name=team_b_name, 
        champion_seasons=champion_seasons, relevant_stats=relevant_stats, debug=debug
    )

    # Step 5: Analyze player salaries to determine if they are overpaid or underpaid
    all_players = selected_players_team_a + selected_players_team_b
    salary_analysis_df = analyze_player_salaries(all_players, predictions_df)

    # Step 6: Calculate compatibility between the players being traded based on their shooting areas
    player_shots = fetch_shots_for_multiple_players(all_players, season=trade_season, court_areas='all')
    compatibility_df = calculate_compatibility_between_players(player_shots)

    return {
        'average_champion_percentiles': average_top_percentiles_df,
        'team_a_comparison_table': team_a_comparison_table,
        'team_b_comparison_table': team_b_comparison_table,
        'comparison_tables': comparison_tables,
        'trade_analysis': trade_analysis_debug,  # Include the trade scenario analysis output
        'trade_valid': trade_valid,  # Include the trade validity status
        'salary_analysis': salary_analysis_df,  # Include player salary analysis output
        'compatibility_analysis': compatibility_df  # Include player compatibility analysis output
    }

def main():
    # Load the predictions data
    predictions_df = pd.read_csv('../data/processed/predictions_df.csv')

    # Define parameters for the test
    team_a_name = "Los Angeles Lakers"
    team_b_name = "Atlanta Hawks"
    selected_players_team_a = ["LeBron James", "Anthony Davis"]
    selected_players_team_b = ["Dejounte Murray"]
    trade_date = "2023-09-15"  # Example trade date
    champion_seasons = ["2020-21", "2021-22", "2022-23"]
    trade_season = "2023-24"
    relevant_stats = ["PTS", "AST", "REB", "STL", "BLK"]
    debug = True  # Set to True to see debug information

    # Call the combined_trade_analysis function
    results = combined_trade_analysis(
        team_a_name, team_b_name, selected_players_team_a, selected_players_team_b, 
        trade_date, champion_seasons, trade_season, relevant_stats, predictions_df, debug
    )

    # Print the trade scenario analysis result
    print("Trade Analysis Debug Info:\n", results['trade_analysis'])
    
    # Check if the trade is valid
    if results['trade_valid']:
        print("The trade satisfies NBA salary matching rules.")
    else:
        print("The trade does NOT satisfy NBA salary matching rules.")

    # Print the other results
    print("Average Champion Percentiles:\n", results['average_champion_percentiles'])
    print(f"{team_a_name} Comparison Table:\n", results['team_a_comparison_table'])
    print(f"{team_b_name} Comparison Table:\n", results['team_b_comparison_table'])
    for stat, table in results['comparison_tables'].items():
        print(f"Comparison Table for {stat}:\n", table)
    
    # Print the salary analysis results
    print("\nPlayer Salary Analysis:")
    print(results['salary_analysis'])

    # Print the compatibility analysis results
    print("\nPlayer Compatibility Analysis:")
    print(results['compatibility_analysis'])

if __name__ == "__main__":
    main()
