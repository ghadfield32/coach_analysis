
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_salary_data(season):
    url = f"https://hoopshype.com/salaries/players/{season}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    players = []
    salaries = []
    seasons = []

    table = soup.find('table', class_='hh-salaries-ranking-table')
    rows = table.find_all('tr')[1:]  # Skip the header row

    for row in rows:
        cols = row.find_all('td')
        player_name = cols[1].get_text(strip=True)
        salary = cols[2].get_text(strip=True).replace('$', '').replace(',', '')

        players.append(player_name)
        salaries.append(int(salary))
        seasons.append(season)

    salary_data = pd.DataFrame({
        'Player': players,
        'Salary': salaries,
        'Season': seasons
    })

    return salary_data

def main():
    start_year = 2023
    end_year = 1990
    all_data = pd.DataFrame()

    player_filter = input("Enter player name or 'all' for all players: ").strip()

    for year in range(start_year, end_year-1, -1):
        season = f"{year}-{year+1}"
        print(f"Scraping data for {season}...")
        season_data = scrape_salary_data(season)
        all_data = pd.concat([all_data, season_data], ignore_index=True)

    if player_filter.lower() != 'all':
        all_data = all_data[all_data['Player'].str.contains(player_filter, case=False)]

    # Ensure the directory exists
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save the combined data to a CSV file
    all_data.to_csv('../data/processed/salary_data.csv', index=True)
    print("Data saved to ../data/processed/salary_data.csv")

if __name__ == "__main__":
    main()
