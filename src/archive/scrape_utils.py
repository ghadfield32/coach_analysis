import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time

def scrape_salary_cap_history(debug=False):
    url = "https://basketball.realgm.com/nba/info/salary_cap"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='basketball compact')
        
        if not table:
            if debug:
                print("Could not find the salary cap table on the page.")
            return None

        data = []
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        for row in table.find('tbody').find_all('tr'):
            cols = row.find_all('td')
            if cols:
                row_data = [col.text.strip() for col in cols]
                data.append(row_data)

        df = pd.DataFrame(data, columns=headers)
        
        # Clean up the data
        df['Season'] = df['Season'].str.extract(r'(\d{4}-\d{4})')
        df['Salary Cap'] = df['Salary Cap'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Convert other columns to float, handling non-numeric values
        for col in df.columns:
            if col not in ['Season', 'Salary Cap']:
                df[col] = pd.to_numeric(df[col].str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        if debug:
            print("Salary cap data scraped successfully")
            print(df.head())
        return df
    except Exception as e:
        if debug:
            print(f"Error scraping salary cap history: {str(e)}")
        return None

DELAY_BETWEEN_REQUESTS = 3  # seconds

def scrape_player_salary_data(start_season, end_season, player_filter=None, debug=False):
    all_data = []
    
    for season in range(start_season, end_season + 1):
        season_str = f"{season}-{str(season+1)[-2:]}"
        url = f"https://hoopshype.com/salaries/players/{season}-{season+1}/"
        if debug:
            print(f"Scraping data for {season_str} from URL: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='hh-salaries-ranking-table')
        
        if table:
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    player = cols[1].get_text(strip=True)
                    if player_filter is None or player_filter.lower() == 'all' or player.lower() == player_filter.lower():
                        salary_text = cols[2].get_text(strip=True)
                        salary = int(salary_text.replace('$', '').replace(',', ''))
                        all_data.append({'Player': player, 'Salary': salary, 'Season': season_str})
        else:
            if debug:
                print(f"No salary data found for season {season_str}")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)  # Delay between requests to avoid hitting rate limits
    
    df = pd.DataFrame(all_data)
    if debug:
        print(f"Scraped salary data for {'all players' if player_filter is None or player_filter.lower() == 'all' else player_filter} from seasons {start_season}-{end_season}:")
        print(df.head())
    return df
    
    df = pd.DataFrame(all_data)
    if debug:
        print(f"Scraped salary data for {'all players' if player_filter is None or player_filter.lower() == 'all' else player_filter} from seasons {start_season}-{end_season}:")
        print(df.head())
    return df

def scrape_team_salary_data(season, debug=False):
    url = f"https://hoopshype.com/salaries/{season}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='hh-salaries-ranking-table')
    rows = table.find_all('tr')[1:]
    data = []
    for row in rows:
        cols = row.find_all('td')
        team = cols[1].get_text(strip=True)
        salary = int(cols[2].get_text(strip=True).replace('$', '').replace(',', ''))
        data.append({'Team': team, 'Team_Salary': salary, 'Season': season})
    df = pd.DataFrame(data)
    if debug:
        print(f"Scraped team salary data for season {season}:")
        print(df.head())
    return df

def scrape_advanced_metrics(player_name, season, debug=False, max_retries=3, retry_delay=60):
    def make_request(url):
        response = requests.get(url)
        if response.status_code == 429:
            if debug:
                print(f"Rate limit hit. Waiting for {retry_delay} seconds before retrying.")
            time.sleep(retry_delay)
            return None
        return response

    for attempt in range(max_retries):
        try:
            search_url = f"https://www.basketball-reference.com/search/search.fcgi?search={player_name.replace(' ', '+')}"
            response = make_request(search_url)
            if response is None:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            search_results = soup.find('div', {'class': 'search-results'})

            if search_results:
                for item in search_results.find_all('div', {'class': 'search-item'}):
                    link = item.find('a')
                    if link and 'players' in link['href']:
                        player_url = f"https://www.basketball-reference.com{link['href']}"
                        break
                else:
                    if debug:
                        print(f"No player URL found for {player_name}")
                    return {}
            else:
                if debug:
                    print(f"No search results found for {player_name}")
                return {}

            time.sleep(2)  # Wait 2 seconds between requests

            response = make_request(player_url)
            if response is None:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'advanced'})
            if table:
                df = pd.read_html(StringIO(str(table)))[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel()
                df['Season'] = df['Season'].astype(str)
                df = df[df['Season'].str.contains(season.split('-')[0], na=False)]
                if not df.empty:
                    row = df.iloc[0]
                    metrics = ['PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
                    result = {col: row[col] for col in metrics if col in row.index}
                    if debug:
                        print(f"Scraped advanced metrics for {player_name} in season {season}: {result}")
                    return result
                else:
                    if debug:
                        print(f"No advanced metrics found for {player_name} in season {season}")
            else:
                if debug:
                    print(f"No advanced stats table found for {player_name}")

        except Exception as e:
            if debug:
                print(f"Error scraping advanced metrics for {player_name}: {e}")
        
        if attempt < max_retries - 1:
            if debug:
                print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    if debug:
        print(f"Failed to scrape advanced metrics for {player_name} after {max_retries} attempts")
    return {}

def load_injury_data(file_path='../data/processed/NBA Player Injury Stats(1951 - 2023).csv'):
    try:
        injury_data = pd.read_csv(file_path)
        injury_data['Date'] = pd.to_datetime(injury_data['Date'])
        injury_data['Season'] = injury_data['Date'].apply(lambda x: f"{x.year}-{str(x.year+1)[-2:]}" if x.month >= 10 else f"{x.year-1}-{str(x.year)[-2:]}")
        print("Injury data loaded successfully")
        return injury_data
    except FileNotFoundError:
        print("Injury data file not found. Proceeding without injury data.")
        return None

def merge_injury_data(player_data, injury_data):
    if injury_data is None:
        return player_data

    all_players_df = player_data.copy()
    all_players_df['Injured'] = False
    all_players_df['Injury_Periods'] = ''
    all_players_df['Total_Days_Injured'] = 0
    all_players_df['Injury_Risk'] = 'Low Risk'

    for index, row in all_players_df.iterrows():
        player_injuries = injury_data[
            (injury_data['Season'] == row['Season']) & 
            (injury_data['Relinquished'].str.contains(row['Player'], case=False, na=False))
        ]
        if not player_injuries.empty:
            periods = []
            total_days = 0
            for _, injury in player_injuries.iterrows():
                start_date = injury['Date']
                acquired_matches = injury_data[
                    (injury_data['Date'] > start_date) & 
                    (injury_data['Acquired'].str.contains(row['Player'], case=False, na=False))
                ]
                if not acquired_matches.empty:
                    end_date = acquired_matches.iloc[0]['Date']
                else:
                    # Assuming injuries last until the end of the season if no acquired date is found
                    end_year = int(row['Season'].split('-')[1])
                    end_date = pd.Timestamp(f"{end_year}-06-30")
                
                period_days = (end_date - start_date).days
                total_days += period_days
                periods.append(f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")

            all_players_df.at[index, 'Injured'] = True
            all_players_df.at[index, 'Injury_Periods'] = '; '.join(periods)
            all_players_df.at[index, 'Total_Days_Injured'] = total_days
            
            # Categorize injury risk based on total days
            if total_days < 10:
                risk = 'Low Risk'
            elif 10 <= total_days <= 20:
                risk = 'Moderate Risk'
            else:
                risk = 'High Risk'
            all_players_df.at[index, 'Injury_Risk'] = risk

    return all_players_df

if __name__ == "__main__":
    # Example usage and testing of all functions
    debug = True
    start_season = 2022
    end_season = 2023
    sample_player = "Ja Morant"  # Example player
    
    print("1. Testing scrape_salary_cap_history:")
    salary_cap_history = scrape_salary_cap_history(debug=debug)
    
    print("\n2. Testing scrape_player_salary_data:")
    player_salary_data = scrape_player_salary_data(start_season, end_season, player_filter=sample_player, debug=debug)
    
    print("\n3. Testing scrape_team_salary_data:")
    team_salary_data = scrape_team_salary_data(f"{start_season}-{str(start_season+1)[-2:]}", debug=debug)
    
    print("\n4. Testing scrape_advanced_metrics:")
    advanced_metrics = scrape_advanced_metrics(sample_player, f"{start_season}-{str(start_season+1)[-2:]}", debug=debug)
    print(f"Advanced Metrics for {sample_player}:")
    print(advanced_metrics)
    
    print("\n5. Testing load_injury_data and merge_injury_data:")
    injury_data = load_injury_data()
    if not player_salary_data.empty and injury_data is not None:
        merged_data = merge_injury_data(player_salary_data, injury_data)
        print("Merged data with injury info:")
        columns_to_display = ['Player', 'Season', 'Salary']
        if 'Injured' in merged_data.columns:
            columns_to_display.append('Injured')
        if 'Injury_Periods' in merged_data.columns:
            columns_to_display.append('Injury_Periods')
        if 'Total_Days_Injured' in merged_data.columns:
            columns_to_display.append('Total_Days_Injured')
        if 'Injury_Risk' in merged_data.columns:
            columns_to_display.append('Injury_Risk')
        print(merged_data[columns_to_display].head())

    if not player_salary_data.empty:
        avg_salary = player_salary_data['Salary'].mean()
        print(f"Average salary for {sample_player} from {start_season} to {end_season}: ${avg_salary:,.2f}")
    
    if not team_salary_data.empty:
        highest_team_salary = team_salary_data.loc[team_salary_data['Team_Salary'].idxmax()]
        print(f"Team with highest salary in {start_season}-{end_season}: {highest_team_salary['Team']} (${highest_team_salary['Team_Salary']:,.2f})")
    
    if not injury_data.empty:
        injury_count = injury_data['Relinquished'].str.contains(sample_player, case=False).sum()
        print(f"Number of injuries/illnesses for {sample_player} from {start_season} to {end_season}: {injury_count}")

    print("\nAll tests completed.")
