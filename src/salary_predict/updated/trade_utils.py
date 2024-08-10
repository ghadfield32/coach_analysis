

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time
from scipy import stats

RELEVANT_STATS = ['PPG', 'APG', 'TPG', 'SPG', 'BPG', 'ORPG', 'DRPG', 'eFG%']
PERCENTILE_THRESHOLDS = [99, 98, 97, 96, 95, 90, 75, 50]

def get_champion(season):
    games = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable='Playoffs').get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    last_game = games.sort_values('GAME_DATE').iloc[-2:]
    winner = last_game[last_game['WL'] == 'W'].iloc[0]
    return winner['TEAM_ID'], winner['TEAM_NAME']

def get_champions(start_year, end_year):
    champions = {}
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        champ_id, champ_name = get_champion(season)
        champions[season] = {'ChampionTeamID': champ_id, 'ChampionTeamName': champ_name}
        time.sleep(1)  # To avoid overwhelming the API
    return champions

def calculate_team_stats(team_players, all_players):
    team_stats = {}
    for stat in RELEVANT_STATS:
        if stat in team_players.columns:
            values = team_players[stat].values
            all_values = all_players[stat].values
            percentiles = np.percentile(all_values, PERCENTILE_THRESHOLDS)
            
            team_stats[stat] = {
                'mean': np.mean(values) if len(values) > 0 else 0,
                'median': np.median(values) if len(values) > 0 else 0,
                'max': np.max(values) if len(values) > 0 else 0,
                'total': values.tolist(),
                'percentile_counts': {
                    f'Top {100-p}%': np.sum(values >= percentiles[i])
                    for i, p in enumerate(PERCENTILE_THRESHOLDS)
                }
            }
    return team_stats

def calculate_champ_stats(champions, num_years=10):
    current_year = max(int(season.split('-')[0]) for season in champions.keys())
    start_year = current_year - num_years + 1
    recent_champions = {k: v for k, v in champions.items() if int(k.split('-')[0]) >= start_year}
    
    champ_stats = pd.DataFrame()
    num_champ_seasons = len(recent_champions)

    for season, champ_info in recent_champions.items():
        games = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            team_id_nullable=champ_info['ChampionTeamID'],
            season_type_nullable='Regular Season'
        ).get_data_frames()[0]
        
        season_stats = pd.DataFrame({
            'PPG': [games['PTS'].mean()],
            'APG': [games['AST'].mean()],
            'TPG': [games['TOV'].mean()],
            'SPG': [games['STL'].mean()],
            'BPG': [games['BLK'].mean()],
            'ORPG': [games['OREB'].mean()],
            'DRPG': [games['DREB'].mean()],
            'eFG%': [(games['FGM'].sum() + 0.5 * games['FG3M'].sum()) / games['FGA'].sum()]
        })
        
        champ_stats = pd.concat([champ_stats, season_stats], ignore_index=True)
    
    if champ_stats.empty:
        return {stat: {'mean': 0, 'median': 0, 'max': 0, 'percentile_counts': {f'Top {100-p}%': 0 for p in PERCENTILE_THRESHOLDS}} for stat in RELEVANT_STATS}
    
    champ_percentiles = {}
    for stat in RELEVANT_STATS:
        if stat in champ_stats.columns:
            values = champ_stats[stat]
            champ_percentiles[stat] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'max': np.max(values),
                'percentile_counts': {
                    f'Top {100-p}%': np.sum(values >= np.percentile(values, p)) / num_champ_seasons
                    for p in PERCENTILE_THRESHOLDS
                }
            }
    
    return champ_percentiles

def compare_stats(current_stats, simulated_stats, league_stats, champ_stats):
    comparison = {}
    for stat in RELEVANT_STATS:
        if stat in current_stats and stat in league_stats:
            current_value = current_stats[stat]['mean']
            after_trade_value = simulated_stats[stat]['mean']
            league_average = league_stats[stat]['mean']
            champ_average = champ_stats[stat]['mean']
            
            all_league_values = league_stats[stat]['total']
            current_percentile = stats.percentileofscore(all_league_values, current_value)
            after_trade_percentile = stats.percentileofscore(all_league_values, after_trade_value)
            
            comparison[stat] = {
                'Current': current_value,
                'Current Percentile': current_percentile,
                'After Trade': after_trade_value,
                'After Trade Percentile': after_trade_percentile,
                'League Average': league_average,
                'Champ Average': champ_average,
                'Current vs League': current_value - league_average,
                'After Trade vs League': after_trade_value - league_average,
                'Current vs Champ': current_value - champ_average,
                'After Trade vs Champ': after_trade_value - champ_average,
                'Current Percentile Counts': current_stats[stat]['percentile_counts'],
                'After Trade Percentile Counts': simulated_stats[stat]['percentile_counts'],
                'Champ Percentile Counts': champ_stats[stat]['percentile_counts']
            }
    return comparison

def simulate_trade(team_players, players_leaving, players_joining):
    team_after_trade = team_players[~team_players['Player'].isin(players_leaving)].copy()
    return pd.concat([team_after_trade, players_joining], ignore_index=True)

FIRST_TAX_APRON = 172_346_000

def check_salary_matching_rules(outgoing_salary, incoming_salary, team_salary_before_trade):
    if team_salary_before_trade < FIRST_TAX_APRON:
        if outgoing_salary <= 7_500_000:
            max_incoming_salary = 2 * outgoing_salary + 250_000
        elif outgoing_salary <= 29_000_000:
            max_incoming_salary = outgoing_salary + 7_500_000
        else:
            max_incoming_salary = 1.25 * outgoing_salary + 250_000
    else:
        max_incoming_salary = 1.10 * outgoing_salary

    return incoming_salary <= max_incoming_salary

def analyze_two_team_trade(team1_abbr, team2_abbr, players_leaving_team1, players_leaving_team2, predictions_df, champions):
    try:
        team1_players = predictions_df[predictions_df['Team'] == team1_abbr]
        team2_players = predictions_df[predictions_df['Team'] == team2_abbr]

        players_joining_team1 = team2_players[team2_players['Player'].isin(players_leaving_team2)]
        players_joining_team2 = team1_players[team1_players['Player'].isin(players_leaving_team1)]

        if players_joining_team1.empty or players_joining_team2.empty:
            print("Could not find one or more of the specified players' stats.")
            return

        current_team1_stats = calculate_team_stats(team1_players, predictions_df)
        current_team2_stats = calculate_team_stats(team2_players, predictions_df)

        team1_after_trade = simulate_trade(team1_players, players_leaving_team1, players_joining_team1)
        team2_after_trade = simulate_trade(team2_players, players_leaving_team2, players_joining_team2)

        simulated_team1_stats = calculate_team_stats(team1_after_trade, predictions_df)
        simulated_team2_stats = calculate_team_stats(team2_after_trade, predictions_df)

        league_stats = calculate_team_stats(predictions_df, predictions_df)
        champ_stats = calculate_champ_stats(champions)

        team1_current_salary = team1_players['Salary'].sum()
        team2_current_salary = team2_players['Salary'].sum()
        team1_new_salary = team1_after_trade['Salary'].sum()
        team2_new_salary = team2_after_trade['Salary'].sum()

        outgoing_salary_team1 = team1_players[team1_players['Player'].isin(players_leaving_team1)]['Salary'].sum()
        incoming_salary_team1 = players_joining_team1['Salary'].sum()
        outgoing_salary_team2 = team2_players[team2_players['Player'].isin(players_leaving_team2)]['Salary'].sum()
        incoming_salary_team2 = players_joining_team2['Salary'].sum()

        salary_match_team1 = check_salary_matching_rules(outgoing_salary_team1, incoming_salary_team1, team1_current_salary)
        salary_match_team2 = check_salary_matching_rules(outgoing_salary_team2, incoming_salary_team2, team2_current_salary)

        team1_comparison = compare_stats(current_team1_stats, simulated_team1_stats, league_stats, champ_stats)
        team2_comparison = compare_stats(current_team2_stats, simulated_team2_stats, league_stats, champ_stats)

        return {
            team1_abbr: {
                'comparison': team1_comparison,
                'current_salary': team1_current_salary,
                'new_salary': team1_new_salary,
                'salary_match': salary_match_team1
            },
            team2_abbr: {
                'comparison': team2_comparison,
                'current_salary': team2_current_salary,
                'new_salary': team2_new_salary,
                'salary_match': salary_match_team2
            }
        }

    except Exception as e:
        print(f"Error in analyze_two_team_trade: {str(e)}")
        return None

def calculate_league_stats_from_api(start_year, end_year):
    league_stats = {stat: [] for stat in RELEVANT_STATS}
    
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        games = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Regular Season'
        ).get_data_frames()[0]
        
        team_stats = games.groupby('TEAM_ID').agg({
            'PTS': 'mean',
            'AST': 'mean',
            'TOV': 'mean',
            'STL': 'mean',
            'BLK': 'mean',
            'OREB': 'mean',
            'DREB': 'mean',
            'FGM': 'sum',
            'FG3M': 'sum',
            'FGA': 'sum'
        })
        
        team_stats['eFG%'] = (team_stats['FGM'] + 0.5 * team_stats['FG3M']) / team_stats['FGA']
        
        for stat, api_stat in zip(RELEVANT_STATS, ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'eFG%']):
            league_stats[stat].extend(team_stats[api_stat].tolist())
        
        time.sleep(1)  # To avoid overwhelming the API
    
    league_percentiles = {}
    for stat in RELEVANT_STATS:
        values = league_stats[stat]
        percentiles = np.percentile(values, PERCENTILE_THRESHOLDS)
        league_percentiles[stat] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'max': np.max(values),
            'total': values,
            'percentile_counts': {
                f'Top {100-p}%': np.sum(np.array(values) >= percentiles[i])
                for i, p in enumerate(PERCENTILE_THRESHOLDS)
            }
        }
    
    return league_percentiles

def identify_overpaid_underpaid(predictions_df):
    predictions_df['Salary_Difference'] = predictions_df['Salary'] - predictions_df['Predicted_Salary']
    predictions_df['Overpaid'] = predictions_df['Salary_Difference'] > 0
    predictions_df['Underpaid'] = predictions_df['Salary_Difference'] < 0
    
    overpaid = predictions_df[predictions_df['Overpaid']].sort_values('Salary_Difference', ascending=False)
    underpaid = predictions_df[predictions_df['Underpaid']].sort_values('Salary_Difference')
    
    return overpaid.head(10), underpaid.head(10)


def identify_overpaid_underpaid(predictions_df):
    # Adjust Predicted_Salary calculation
    predictions_df['Predicted_Salary'] = predictions_df['Predicted_Salary'] * predictions_df['Salary_Cap_Inflated']
    
    predictions_df['Salary_Difference'] = predictions_df['Salary'] - predictions_df['Predicted_Salary']
    predictions_df['Overpaid'] = predictions_df['Salary_Difference'] > 0
    predictions_df['Underpaid'] = predictions_df['Salary_Difference'] < 0
    
    overpaid = predictions_df[predictions_df['Overpaid']].sort_values('Salary_Difference', ascending=False)
    underpaid = predictions_df[predictions_df['Underpaid']].sort_values('Salary_Difference')
    
    return overpaid.head(10), underpaid.head(10)

if __name__ == "__main__":
    predictions_df = pd.read_csv('../data/processed/predictions_df.csv')
    predictions_df = predictions_df[['Season', 'Position', 'Age', 'Team', 'TeamID', 'Years of Service', '3P%', '2P%', 'eFG%', 'FT%', 'PER', 'VORP', 'Salary', 'Total_Days_Injured', 'Injury_Risk', 'Salary Cap', 'Salary_Cap_Inflated', 'PPG', 'APG', 'TPG', 'SPG', 'BPG', 'Availability', 'SalaryPct', 'Efficiency', 'ValueOverReplacement', 'ExperienceSquared', 'Days_Injured_Percentage', 'WSPG', 'DWSPG', 'OWSPG', 'PFPG', 'ORPG', 'DRPG', 'RF_Predictions', 'XGB_Predictions', 'Predicted_Salary', 'Player']]
    
    current_year = 2023
    start_year = current_year - 10
    
    # Calculate league stats from API
    league_stats = calculate_league_stats_from_api(start_year, current_year - 1)
    
    print("League Stats:")
    for stat, values in league_stats.items():
        print(f"\n{stat}:")
        print(f"  Mean: {values['mean']:.2f}")
        print(f"  Median: {values['median']:.2f}")
        print(f"  Max: {values['max']:.2f}")
        print("  Percentile Counts:")
        for percentile, count in values['percentile_counts'].items():
            print(f"    {percentile}: {count}")

    champions = get_champions(start_year, current_year - 1)
    champ_stats = calculate_champ_stats(champions)
    
    print("\nChampion Stats:")
    for stat, values in champ_stats.items():
        print(f"\n{stat}:")
        print(f"  Mean: {values['mean']:.2f}")
        print(f"  Median: {values['median']:.2f}")
        print(f"  Max: {values['max']:.2f}")
        print("  Percentile Counts:")
        for percentile, count in values['percentile_counts'].items():
            print(f"    {percentile}: {count:.2f}")
    
    # Identify overpaid and underpaid players with corrected Predicted_Salary
    overpaid, underpaid = identify_overpaid_underpaid(predictions_df)
    
    print("\nTop 10 Overpaid Players:")
    print(overpaid[['Player', 'Team', 'Salary', 'Predicted_Salary', 'Salary_Difference']])
    
    print("\nTop 10 Underpaid Players:")
    print(underpaid[['Player', 'Team', 'Salary', 'Predicted_Salary', 'Salary_Difference']])
    
    # Example trade analysis
    team1_abbr = 'LAL'
    team2_abbr = 'BOS'
    players_leaving_team1 = ['Anthony Davis', 'D\'Angelo Russell']
    players_leaving_team2 = ['Jayson Tatum', 'Jaylen Brown']
    
    result = analyze_two_team_trade(team1_abbr, team2_abbr, players_leaving_team1, players_leaving_team2, predictions_df, champions)
    
    if result:
        for team_abbr, team_data in result.items():
            print(f"\n{team_abbr} Trade Analysis:")
            print(f"Current Salary: ${team_data['current_salary']:,.2f}")
            print(f"Salary After Trade: ${team_data['new_salary']:,.2f}")
            print(f"Salary Difference: ${team_data['new_salary'] - team_data['current_salary']:,.2f}")
            print(f"Salary Match: {'Yes' if team_data['salary_match'] else 'No'}")
            
            print("\nStat Comparisons:")
            for stat in RELEVANT_STATS:
                values = team_data['comparison'][stat]
                print(f"{stat}:")
                print(f"  Current: {values['Current']:.2f} ({values['Current Percentile']:.1f}%ile)")
                print(f"  After Trade: {values['After Trade']:.2f} ({values['After Trade Percentile']:.1f}%ile)")
                print(f"  Change vs League: {values['After Trade vs League'] - values['Current vs League']:.2f}")
                print(f"  Change vs Champ: {values['After Trade vs Champ'] - values['Current vs Champ']:.2f}")
                print("  Percentile Counts (Current / After Trade / Champ Average):")
                for percentile in PERCENTILE_THRESHOLDS:
                    current_count = values['Current Percentile Counts'][f'Top {100-percentile}%']
                    after_trade_count = values['After Trade Percentile Counts'][f'Top {100-percentile}%']
                    champ_count = values['Champ Percentile Counts'][f'Top {100-percentile}%']
                    print(f"    Top {100-percentile}%: {current_count:.1f} / {after_trade_count:.1f} / {champ_count:.1f}")
                    
                    

