import argparse
import pandas as pd
import logging
import time
import glob
import os
from datetime import datetime, timedelta
from fetch_utils import fetch_all_players
from process_utils import process_player_data, inflate_value, calculate_percentages
from scrape_utils import scrape_salary_cap_history, merge_injury_data, scrape_player_salary_data, scrape_team_salary_data, load_injury_data
from data_utils import clean_dataframe, merge_salary_cap_data, validate_data


def update_data(existing_data, start_year, end_year, player_filter=None, min_avg_minutes=None, debug=False):
    all_data = existing_data.copy() if existing_data is not None else pd.DataFrame()

    # Load injury data
    injury_data = load_injury_data()

    salary_data = scrape_player_salary_data(start_year, end_year, player_filter=player_filter, debug=debug)

    new_data = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        
        if debug:
            print(f"Processing season: {season}")
        
        team_salary_data = scrape_team_salary_data(season, debug=debug)
        all_players = fetch_all_players(season=season, debug=debug)
        
        season_salary_data = salary_data[salary_data['Season'] == season]
        
        if player_filter and player_filter.lower() != 'all':
            season_salary_data = season_salary_data[season_salary_data['Player'].str.lower() == player_filter.lower()]

        additional_stats = []

        for _, salary_row in season_salary_data.iterrows():
            player_name = salary_row['Player']
            player_name_lower = player_name.lower()
            
            if player_name_lower in all_players:
                player_stats = process_player_data(player_name, season, all_players, debug=debug)
                if player_stats:
                    player_stats['Salary'] = salary_row['Salary']
                    additional_stats.append(player_stats)
            elif debug:
                print(f"Player not found in all_players: {player_name}")

        additional_stats_df = pd.DataFrame(additional_stats)

        if additional_stats_df.empty or 'Team' not in additional_stats_df.columns:
            if debug:
                print(f"Warning: No valid player stats data for season {season}")
            continue

        # Merge team salary data
        merged_data = pd.merge(additional_stats_df, team_salary_data, on=['Team', 'Season'], how='left', suffixes=('', '_team'))

        # Apply minimum average minutes filter if specified
        if min_avg_minutes is not None:
            before_filter = len(merged_data)
            merged_data = merged_data[merged_data['MP'] >= min_avg_minutes]
            if debug:
                print(f"Filtered {before_filter - len(merged_data)} players based on minimum average minutes")

        # Merge injury data
        merged_data = merge_injury_data(merged_data, injury_data)

        new_data = pd.concat([new_data, merged_data], ignore_index=True, sort=False)

    # Remove existing data for the players and seasons we just updated
    if not all_data.empty and not new_data.empty:
        all_data = all_data[~((all_data['Season'].isin(new_data['Season'])) & 
                              (all_data['Player'].isin(new_data['Player'])))]

    # Combine existing data with new data
    all_data = pd.concat([all_data, new_data], ignore_index=True, sort=False)

    # Sort the final data by season and player
    all_data.sort_values(by=['Season', 'Player'], inplace=True)

    # Calculate percentages
    all_data = calculate_percentages(all_data)

    # Clean the dataframe
    all_data = clean_dataframe(all_data)

    if debug:
        print(f"Final data shape: {all_data.shape}")
        print(f"Columns: {all_data.columns.tolist()}")

    return all_data

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def remove_old_logs(log_dir, days_to_keep=7):
    current_time = datetime.now()
    for log_file in glob.glob(os.path.join(log_dir, 'stat_pull_log_*.txt')):
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        if current_time - file_modified_time > timedelta(days=days_to_keep):
            os.remove(log_file)

def main(start_year, end_year, player_filter=None, min_avg_minutes=None, debug=False):
    start_time = time.time()
    
    # Create the output directory if it doesn't exist
    output_dir = '../../data/stat_pull_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove old log files
    remove_old_logs(output_dir)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'stat_pull_log_{get_timestamp()}.txt')
    logging.basicConfig(filename=log_file, level=logging.DEBUG if debug else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        logging.info(f"Starting data update for years {start_year} to {end_year}")
        
        
        processed_file_path = '../../data/processed/nba_player_data_final_inflated.csv'
        salary_cap_file_path = '../../data/processed/salary_cap_history_inflated.csv'

        # Load existing data
        try:
            existing_data = pd.read_csv(processed_file_path)
        except FileNotFoundError:
            existing_data = pd.DataFrame()

        try:
            if debug:
                print(f"Updating data for years {start_year} to {end_year}")
            updated_data = update_data(existing_data, start_year, end_year, player_filter, min_avg_minutes, debug=debug)

            if not updated_data.empty:
                if debug:
                    print("New data retrieved. Processing and saving...")

                salary_cap_data = scrape_salary_cap_history(debug=debug)

                if salary_cap_data is not None:
                    salary_cap_data.to_csv(salary_cap_file_path, index=False)
                    updated_data = merge_salary_cap_data(updated_data, salary_cap_data)

                # Final cleaning of the data
                updated_data = clean_dataframe(updated_data)

                # Save the updated data
                updated_data.to_csv(processed_file_path, index=False, float_format='%.2f')
                if debug:
                    print(f"Updated data saved to {processed_file_path}")

                # Print summary of the data
                summary_columns = ['Season', 'Player', 'Salary', 'GP', 'PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP', 'Injured', 'FG%', '3P%', 'FT%', 'Team_Salary', 'Salary Cap', 'Salary_Cap_Inflated']
                available_columns = [col for col in summary_columns if col in updated_data.columns]
                print("\nData summary:")
                print(updated_data[available_columns].head().to_string(index=False))
            else:
                print("No new data to save. The dataset is empty.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            
        logging.info("Data update completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Traceback:", exc_info=True)
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Total execution time: {execution_time:.2f} seconds")

        # Print summary to console as well
        print(f"Process completed. Log file saved to: {log_file}")
        print(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Update NBA player data")
    parser.add_argument("--start_year", type=int, default=current_year-1, help="Start year for data update")
    parser.add_argument("--end_year", type=int, default=current_year, help="End year for data update")
    parser.add_argument("--player_filter", type=str, default="all", help="Filter for specific player or 'all'")
    parser.add_argument("--min_avg_minutes", type=float, default=25, help="Minimum average minutes per game")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    main(args.start_year, args.end_year, args.player_filter, args.min_avg_minutes, args.debug)

    

# if __name__ == "__main__":
#     start_year = 2019
#     end_year = 2023
#     player_filter = input("Enter player name or 'all' for all players: ").strip()
#     min_avg_minutes = None
#     if player_filter.lower() == 'all':
#         min_avg_minutes = float(input("Enter the minimum average minutes per game (default 25 mins): ") or 25)

#     debug = True  # Set to False to disable debug output

#     main(start_year, end_year, player_filter, min_avg_minutes, debug)






    
