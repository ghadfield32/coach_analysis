
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# Function to get the player URL from the search results
def get_player_url(player_name):
    search_url = f"https://www.basketball-reference.com/search/search.fcgi?search={player_name.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    search_results = soup.find('div', {'class': 'search-results'})
    if search_results:
        for item in search_results.find_all('div', {'class': 'search-item'}):
            link = item.find('a')
            if link and 'players' in link['href']:
                return f"https://www.basketball-reference.com{link['href']}"
    
    raise ValueError(f"Player URL not found for {player_name}")

# General function to scrape data from a specific section of the player's page
def scrape_player_data(player_name, section_id, season_end_year):
    player_url = get_player_url(player_name)
    response = requests.get(player_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data = {}
    table = soup.find('table', {'id': section_id})
    if table:
        df = pd.read_html(StringIO(str(table)))[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel()  # Drop the multi-index level
        df = df[df['Season'].str.contains(season_end_year)]
        
        if not df.empty:
            row = df.iloc[0]
            for col in df.columns:
                if col not in ['Season', 'Age', 'Tm', 'Lg', 'Pos', 'G', 'GS']:
                    try:
                        data[col] = float(row[col])
                    except ValueError:
                        data[col] = row[col]
        else:
            raise ValueError(f"No data found for season {season_end_year}")
    else:
        raise ValueError(f"Section {section_id} not found for {player_name}")
    
    return data

# Main function to execute the script
def main():
    player_name = "Stephen Curry"
    season_end_year = "2023-24"  # Adjust the season as needed
    section_id = "advanced"  # Adjust the section as needed
    #Section IDs available for Stephen Curry:
    #['projection', 'per_game', 'playoffs_per_game', 'stathead_insights', 'totals', 'playoffs_totals', 'advanced', 'playoffs_advanced']

    try:
        player_data = scrape_player_data(player_name, section_id, season_end_year)
        print(f"Data for {player_name} in {season_end_year} from section {section_id}:")
        print(player_data)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()

