
#basketball reference ONLY HAS THIS YEARS SALARY's, making this useless



import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep, time

# Define constants for rate limiting
MAX_REQUESTS_PER_MINUTE = 30
DELAY_BETWEEN_REQUESTS = 2  # in seconds

# Function to scrape team salary data from Basketball Reference for multiple seasons
def scrape_team_salary_data_br(team_abbr, seasons):
    all_team_data = pd.DataFrame()
    
    for season in seasons:
        url = f"https://www.basketball-reference.com/contracts/{team_abbr}.html"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for {team_abbr} in season {season}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'id': 'contracts'})
        if not table:
            print(f"No table found for {team_abbr} in season {season}")
            continue

        # Extract the table as a DataFrame
        df = pd.read_html(str(table))[0]

        # Fix multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        # Ensure the 'Player' column exists and contains the right data
        if 'Player' not in df.columns:
            print(f"'Player' column not found in data for {team_abbr} in season {season}")
            continue

        # Select relevant columns and rename them
        df = df[['Player', season]].rename(columns={season: 'Salary'})
        df['Salary'] = df['Salary'].str.replace('$', '').str.replace(',', '').astype(float, errors='ignore')

        # Filter out players without a salary
        df = df.dropna(subset=['Salary'])
        df = df[df['Salary'] != 0]

        # Calculate team total from the 'Team Totals' row
        team_total_row = df[df['Player'] == 'Team Totals']
        if not team_total_row.empty:
            team_total = team_total_row['Salary'].values[0]
        else:
            team_total = df['Salary'].sum()

        # Remove the 'Team Totals' row from the dataframe
        df = df[df['Player'] != 'Team Totals']

        # Add team total and team abbreviation as new columns for each player
        df['Team Total'] = team_total
        df['Team Abbreviation'] = team_abbr

        # Append the data for this season to the overall dataframe
        all_team_data = pd.concat([all_team_data, df], ignore_index=True)

    return all_team_data

def main():
    teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
    seasons = ['1990-91', '1991-92', '1992-93', '1993-94', '1994-95', '1995-96', '1996-97', '1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

    all_team_data = pd.DataFrame()
    request_count = 0
    start_time = time()

    for team_abbr in teams:
        if request_count >= MAX_REQUESTS_PER_MINUTE:
            elapsed_time = time() - start_time
            if elapsed_time < 60:
                sleep(60 - elapsed_time)
            start_time = time()
            request_count = 0

        print(f"Scraping salary data for {team_abbr}...")
        team_data = scrape_team_salary_data_br(team_abbr, seasons)
        if not team_data.empty:
            all_team_data = pd.concat([all_team_data, team_data], ignore_index=True)
        
        request_count += 1
        sleep(DELAY_BETWEEN_REQUESTS)

    # Ensure the directory exists
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save the combined data to CSV files
    all_team_data.to_csv('../data/processed/team_salary_data_br.csv', index=False)

    print("Team data saved to ../data/processed/team_salary_data_br.csv")

if __name__ == "__main__":
    main()
