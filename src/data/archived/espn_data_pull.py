

#Pulled in Historic Player Salary from ESPN:
#https://www.espn.com/nba/salaries/_/year/2025/page/10/seasontype/4

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# List of User-Agent strings to rotate
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
    # Add more user agents if needed
]

def fetch_page_content(url, retries=3):
    for attempt in range(retries):
        headers = {
            "User-Agent": random.choice(user_agents)
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 503:
                print(f"Failed to fetch page content: {response.status_code}, retrying with longer wait...")
                time.sleep((2 ** attempt + random.uniform(0, 1)) * 5)  # Longer wait time
            else:
                print(f"Failed to fetch page content: {response.status_code}, retrying...")
                time.sleep(2 ** attempt + random.uniform(0, 1))
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}, retrying...")
            time.sleep(2 ** attempt + random.uniform(0, 1))
    print(f"Failed to fetch page content after {retries} attempts.")
    return None

def extract_salary_data(soup):
    table = soup.find('table', class_='tablehead')
    if table:
        rows = table.find_all('tr', class_=['oddrow', 'evenrow'])
        player_salaries = []
        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 0:
                player_name = cells[1].text.strip()
                team_name = cells[2].text.strip()
                player_salary = cells[3].text.strip().replace('$', '').replace(',', '')
                player_salaries.append((player_name, team_name, int(player_salary)))
        return player_salaries
    else:
        print("No salary table found.")
        return []

def fetch_season_salaries(season):
    all_player_salaries = []
    page = 1
    while True:
        url = f"https://www.espn.com/nba/salaries/_/year/{season}/page/{page}/seasontype/4"
        page_content = fetch_page_content(url)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            salaries_on_page = extract_salary_data(soup)
            if not salaries_on_page:
                break
            all_player_salaries.extend(salaries_on_page)
            page += 1
            time.sleep(random.uniform(2, 5))  # Random delay between 2 to 5 seconds
        else:
            break
    return all_player_salaries

def get_season_format(year):
    return f"{year}-{str(year + 1)[-2:]}"

def fetch_all_seasons(start_year, end_year, filename='nba_salaries.csv'):
    existing_data = load_existing_data(filename)
    existing_seasons = set(existing_data['Season'].unique()) if not existing_data.empty else set()
    all_seasons_data = []
    for year in range(start_year, end_year + 1):
        season = get_season_format(year)
        if season not in existing_seasons:
            print(f"Fetching data for {season}")
            season_salaries = fetch_season_salaries(year)
            for player_name, team_name, player_salary in season_salaries:
                all_seasons_data.append([player_name, player_salary, team_name, season])
        else:
            print(f"Data for {season} already exists. Skipping...")
    return all_seasons_data

def load_existing_data(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=['Player', 'Player_Salary', 'Team', 'Season'])

def save_data(data, filename):
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        combined_data = pd.concat([existing_data, pd.DataFrame(data, columns=['Player', 'Player_Salary', 'Team', 'Season'])])
        combined_data.to_csv(filename, index=False)
    else:
        pd.DataFrame(data, columns=['Player', 'Player_Salary', 'Team', 'Season']).to_csv(filename, index=False)

def main(start_year, end_year, filename='../data/raw/espn_nba_salaries.csv'):
    all_seasons_data = fetch_all_seasons(start_year, end_year, filename)
    if all_seasons_data:
        save_data(all_seasons_data, filename)
    df = load_existing_data(filename)

    # Add Team Abbreviation and Position columns (if needed)
    # Placeholder function to derive team abbreviation and position
    def derive_team_abbreviation(team_name):
        return team_name.split()[-1][:3].upper()  # Placeholder example

    def derive_position(player_name):
        return 'Position'  # Placeholder example

    df['Team_Abbreviation'] = df['Team'].apply(derive_team_abbreviation)
    df['Position'] = df['Player'].apply(derive_position)

    # Display the DataFrame
    print(df)

# Example usage:
main(1999, 2024, '../data/raw/espn_nba_salaries.csv')
