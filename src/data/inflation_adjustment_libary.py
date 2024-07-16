
import pandas as pd
import cpi
from datetime import datetime

# Update CPI data
cpi.update()

# Load the salary cap data
salary_cap_data = pd.read_csv('../data/raw/salary_cap_history.csv')

# Function to adjust for inflation
def inflate_value(value, year_str):
    try:
        year = int(year_str[:4])
        current_year = datetime.now().year
        
        if year >= current_year:
            return value  # Return the original value for future years

        # Adjust to 2022 dollars to match the original data
        return cpi.inflate(value, year, to=2022)
    except ValueError:
        print(f"Invalid year format: {year_str}")
        return value
    except cpi.errors.CPIObjectDoesNotExist:
        # If data for the specific year is not available, use the earliest available year
        earliest_year = min(cpi.SURVEYS['CPI-U'].indexes['annual'].keys()).year
        return cpi.inflate(value, earliest_year, to=2022)

# Add the inflation-adjusted column
salary_cap_data['Salary_Cap_Inflated'] = salary_cap_data.apply(
    lambda row: inflate_value(row['Salary Cap'], row['Year']),
    axis=1
)

# Rename the 'Year' column to 'Season'
salary_cap_data.rename(columns={'Year': 'Season'}, inplace=True)

# Load the player data
player_data = pd.read_csv('../data/processed/final_salary_data_with_yos_and_cap.csv')

# Merge the player data with the salary cap data
merged_data = pd.merge(player_data, salary_cap_data[['Season', 'Salary_Cap_Inflated']], on='Season', how='left')

# Set display options to avoid scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)

# Print the merged data
print("Merged Data:")
print(merged_data)

# Optional: Save the merged data to a new CSV file
merged_data.to_csv('../data/processed/final_salary_data_with_yos_and_inflated_cap.csv', index=False)

# Validate merge
print("Original Salary Cap Data:")
print(salary_cap_data[['Season', 'Salary Cap', '2022 Dollars', 'Salary_Cap_Inflated']])

print("\nMerged Data (Selected Columns):")
print(merged_data[['Season', 'Salary Cap', '2022 Dollars', 'Salary_Cap_Inflated']])

