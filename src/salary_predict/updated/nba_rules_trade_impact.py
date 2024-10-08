# https://www.hoopsrumors.com/2023/09/salary-matching-rules-for-trades-during-2023-24-season.html

import pandas as pd

# Constants for the 2023/24 season
FIRST_TAX_APRON_2023 = 172_346_000
SALARY_CAP_2023 = 136_021_000

# Percentages based on rules
UP_TO_7500K_MULTIPLIER = 2.0
UP_TO_7500K_BONUS = 250_000 / SALARY_CAP_2023

BETWEEN_7501K_AND_29M_BONUS = 7_500_000 / SALARY_CAP_2023

ABOVE_29M_MULTIPLIER = 1.25
ABOVE_29M_BONUS = 250_000 / SALARY_CAP_2023

ABOVE_FIRST_APRON_MULTIPLIER = 1.10

def check_salary_matching_rules(outgoing_salary, incoming_salary, team_salary_before_trade, salary_cap, first_tax_apron, debug=False):
    if debug:
        print(f"Debug: Checking salary matching rules:")
        print(f"  Outgoing Salary: ${outgoing_salary:,.2f}")
        print(f"  Incoming Salary: ${incoming_salary:,.2f}")
        print(f"  Team Salary Before Trade: ${team_salary_before_trade:,.2f}")
        print(f"  Salary Cap: ${salary_cap:,.2f}")
        print(f"  First Tax Apron: ${first_tax_apron:,.2f}")

    if team_salary_before_trade < first_tax_apron:
        if outgoing_salary <= 7_500_000:
            max_incoming_salary = (UP_TO_7500K_MULTIPLIER * outgoing_salary + UP_TO_7500K_BONUS * salary_cap)
            rule = "200% of outgoing + 250,000 (up to 7,500,000)"
            percentage_limit = (UP_TO_7500K_MULTIPLIER * outgoing_salary + UP_TO_7500K_BONUS * salary_cap) / outgoing_salary
        elif outgoing_salary <= 29_000_000:
            max_incoming_salary = outgoing_salary + BETWEEN_7501K_AND_29M_BONUS * salary_cap
            rule = "outgoing + 7,500,000 (7,500,001 to 29,000,000)"
            percentage_limit = (outgoing_salary + BETWEEN_7501K_AND_29M_BONUS * salary_cap) / outgoing_salary
        else:
            max_incoming_salary = (ABOVE_29M_MULTIPLIER * outgoing_salary + ABOVE_29M_BONUS * salary_cap)
            rule = "125% of outgoing + 250,000 (above 29,000,000)"
            percentage_limit = (ABOVE_29M_MULTIPLIER * outgoing_salary + ABOVE_29M_BONUS * salary_cap) / outgoing_salary
    else:
        max_incoming_salary = ABOVE_FIRST_APRON_MULTIPLIER * outgoing_salary
        rule = "110% of outgoing (above first tax apron)"
        percentage_limit = ABOVE_FIRST_APRON_MULTIPLIER

    if debug:
        print(f"  Max Incoming Salary Allowed: ${max_incoming_salary:,.2f}")
        print(f"  Rule Applied: {rule}")
        print(f"  Percentage Limit: {percentage_limit:.2f}")

    return incoming_salary <= max_incoming_salary, max_incoming_salary, rule, percentage_limit

def analyze_trade_scenario(players1, players2, predictions_df, season, debug=False):
    # Filter the dataframe for the specified season
    season_data = predictions_df[predictions_df['Season'] == season]

    # Ensure all players in each list are from the same team
    teams1 = season_data[season_data['Player'].isin(players1)]['Team'].unique()
    teams2 = season_data[season_data['Player'].isin(players2)]['Team'].unique()

    if len(teams1) != 1 or len(teams2) != 1:
        print(f"Error: All players in each list must be from the same team.")
        return None

    team1 = teams1[0]
    team2 = teams2[0]

    if team1 == team2:
        print(f"Error: The two teams involved in the trade must be different.")
        return None

    # Calculate total salaries for each group of players
    outgoing_salary_team1 = season_data[season_data['Player'].isin(players1)]['Salary'].sum()
    incoming_salary_team2 = season_data[season_data['Player'].isin(players2)]['Salary'].sum()
    outgoing_salary_team2 = incoming_salary_team2  # outgoing from team2's perspective
    incoming_salary_team1 = outgoing_salary_team1  # incoming from team1's perspective

    # Check salary matching rules for both teams
    team1_salary_before_trade = season_data[season_data['Team'] == team1]['Salary'].sum()
    team2_salary_before_trade = season_data[season_data['Team'] == team2]['Salary'].sum()

    # Determine tax apron status
    team1_tax_apron_status = "Below" if team1_salary_before_trade < FIRST_TAX_APRON_2023 else "Above"
    team2_tax_apron_status = "Below" if team2_salary_before_trade < FIRST_TAX_APRON_2023 else "Above"

    trade_works_for_team1, team1_max_incoming_salary, team1_rule, team1_percentage_limit = check_salary_matching_rules(
        outgoing_salary_team1, incoming_salary_team1, team1_salary_before_trade, SALARY_CAP_2023, FIRST_TAX_APRON_2023, debug
    )
    trade_works_for_team2, team2_max_incoming_salary, team2_rule, team2_percentage_limit = check_salary_matching_rules(
        outgoing_salary_team2, incoming_salary_team2, team2_salary_before_trade, SALARY_CAP_2023, FIRST_TAX_APRON_2023, debug
    )

    if debug:
        print("\nDebug: Trade Analysis Results:")
        print(f"Team 1 ({team1}):")
        print(f"  Total Outgoing Salary: ${outgoing_salary_team1:,.2f}")
        print(f"  Max Incoming Salary Allowed: ${team1_max_incoming_salary:,.2f} (Rule: {team1_rule})")
        print(f"  Percentage Limit: {team1_percentage_limit:.2f}")
        print(f"Team 2 ({team2}):")
        print(f"  Total Outgoing Salary: ${outgoing_salary_team2:,.2f}")
        print(f"  Max Incoming Salary Allowed: ${team2_max_incoming_salary:,.2f} (Rule: {team2_rule})")
        print(f"  Percentage Limit: {team2_percentage_limit:.2f}")

    print(f"Trade Works for Team 1: {'Yes' if trade_works_for_team1 else 'No'}")
    if not trade_works_for_team1:
        print(f"  Trade fails for Team 1 because incoming salary exceeds max allowed under rule: {team1_rule}")
        print(f"  Team 1 is {team1_tax_apron_status} the First Tax Apron.")

    print(f"Trade Works for Team 2: {'Yes' if trade_works_for_team2 else 'No'}")
    if not trade_works_for_team2:
        print(f"  Trade fails for Team 2 because incoming salary exceeds max allowed under rule: {team2_rule}")
        print(f"  Team 2 is {team2_tax_apron_status} the First Tax Apron.")

    if trade_works_for_team1 and trade_works_for_team2:
        print("The trade is valid according to salary matching rules.")
    else:
        print("The trade does not satisfy salary matching rules.")

    return trade_works_for_team1 and trade_works_for_team2

if __name__ == "__main__":
    # Load the real predictions dataframe
    predictions_df = pd.read_csv('../data/processed/predictions_df.csv')

    # Specify two lists of players for the trade scenario
    players1 = ["Anthony Davis", "LeBron James"]
    players2 = ["Jayson Tatum", "Jaylen Brown"]

    # Analyze the trade scenario for the specified season with debugging enabled
    season = 2023
    print(f"Analyzing trade for the {season} season:")
    analyze_trade_scenario(players1, players2, predictions_df, season, debug=True)
