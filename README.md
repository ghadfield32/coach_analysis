# NBA Salary Analysis and Prediction Project

**[Streamlit_App](https://nba-salary-predictions.streamlit.app/)

## Table of Contents
1. [Introduction](#introduction)
2. [Data Sources](#data-sources)
3. [Project Structure](#project-structure)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Features](#features)
8. [Methodology](#methodology)
9. [Results and Insights](#results-and-insights)
10. [Future Improvements](#future-improvements)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

This project presents a comprehensive analysis of NBA player salaries, leveraging historical data since 2000 to provide insights into salary trends, player performance metrics, and future salary predictions. By combining advanced statistical analysis with machine learning techniques, we aim to identify the most overpaid and underpaid players, analyze career trajectories, and predict future salaries based on player performance and other relevant factors.

## Data Sources

Our analysis is based on a rich dataset compiled from various reliable sources:

1. **Salary Data**: 
   - Source: [Basketball Reference Salary Cap History](https://www.basketball-reference.com/contracts/salary-cap-history.html)
   - Description: Historical NBA salary cap data and maximum salary details based on years of service.

2. **Advanced Metrics**:
   - Source: [Basketball Reference](https://www.basketball-reference.com)
   - Description: Advanced player metrics including Player Efficiency Rating (PER), True Shooting Percentage (TS%), and Value Over Replacement Player (VORP).

3. **Player Salaries and Team Data**:
   - Source: [Hoopshype](https://hoopshype.com)
   - Description: Detailed player salary information and team salary data across multiple seasons.

4. **Injury Data** (for future improvements):
   - Source: [Kaggle NBA Injury Stats 1951-2023](https://www.kaggle.com/datasets/loganlauton/nba-injury-stats-1951-2023/data)
   - Description: Comprehensive NBA injury statistics from 1951 to 2023.

## Project Structure

```
nba-salary-analysis/
│
├── src/
│   ├── salary_predict/
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── model_trainer.py
│   │   ├── predictor.py
│   │   └── app.py
│   │
│   └── scripts/
│       ├── data_collection.py
│       └── data_cleaning.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── predictions/
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_development.ipynb
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

## Pipeline Architecture

Our data pipeline has been modernized with a **split DAG architecture** for better reliability, maintainability, and monitoring:

### 🏗️ DAG Structure

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ nba_salary_     │    │ nba_advanced_    │    │ salary_cap_     │
│ ingest          │    │ ingest           │    │ snapshot        │
│ (Daily)         │    │ (Daily)          │    │ (Yearly)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    nba_data_loader        │
                    │    (Daily)                │
                    │    • ExternalTaskSensors  │
                    │    • Validation           │
                    │    • DuckDB Loading       │
                    └───────────────────────────┘
```

### 📊 DAG Details

| DAG | Schedule | Purpose | Retries | SLA |
|-----|----------|---------|---------|-----|
| `nba_salary_ingest` | Daily | Player & team salary scraping | 3 | 2h |
| `nba_advanced_ingest` | Daily | Advanced metrics scraping | 2 | 1h |
| `salary_cap_snapshot` | Yearly (July 1) | Salary cap history | 2 | 1h |
| `injury_etl` | Monthly | Injury data processing | 1 | 30m |
| `nba_data_loader` | Daily | Unified validation & loading | 2 | 3h |

### 🎯 Benefits

- **Isolated Failures**: One source failing doesn't block others
- **Different Cadences**: Salary scraping daily, cap history yearly  
- **Easier Maintenance**: Teams can iterate on one source independently
- **Better Monitoring**: Granular SLAs and retry policies per source
- **Faster DagBag Parsing**: Smaller DAG files load faster

### 🔄 ESPN URL Update

The ESPN fallback scraper has been updated to use the new URL pattern:
- **Old**: `.../year/{year}/page/{page}`
- **New**: `.../year/{year}/page/{page}/seasontype/4` (page optional for page 1)

### 🚀 Usage

```bash
# Query the Parquet data lake with DuckDB
python -c "
import duckdb
con = duckdb.connect('/workspace/data/nba_stats.duckdb')
result = con.execute('SELECT COUNT(*) FROM nba_2024_25').fetchone()
print(f'Loaded {result[0]} salary records')
"

# Run individual DAGs
airflow dags trigger nba_salary_ingest --conf '{"season": "2024-25"}'
airflow dags trigger nba_advanced_ingest --conf '{"season": "2024-25"}'
```

### 📈 Monitoring

- **ExternalTaskSensors**: Wait for upstream DAGs with configurable timeouts
- **Quality Gates**: Validation fails the DAG if data quality issues detected
- **Email Alerts**: Notifications on failures with retry policies
- **SLA Monitoring**: Track performance against service level agreements

## Installation

To set up the project environment:

```bash
git clone https://github.com/your-username/nba-salary-analysis.git
cd nba-salary-analysis
pip install -r requirements.txt
```

## Usage

To run the Streamlit app:

```bash
streamlit run src/salary_predict/app.py
```

## Features

1. **Data Overview**: Display raw and processed NBA salary and performance data.
2. **Exploratory Data Analysis**: 
   - Visualize salary distributions
   - Analyze age vs. salary trends
   - Explore positional salary differences
3. **Advanced Analytics**:
   - VORP to Salary Ratio analysis
   - Career trajectory clustering
   - Performance metrics comparison
4. **Salary Predictions**: 
   - Predict future player salaries
   - Compare actual vs. predicted salaries
5. **Player Comparisons**: Compare selected players based on predicted salaries and performance metrics.
6. **Overpaid vs. Underpaid Analysis**: Identify and analyze the most overpaid and underpaid players.
7. **Model Selection and Evaluation**: Evaluate different machine learning models for salary prediction.
8. **Model Retraining**: Option to retrain models with updated data.

## Methodology

1. **Data Collection and Preprocessing**:
   - Web scraping using BeautifulSoup for advanced metrics and salary data
   - Data cleaning and merging from multiple sources
   - Feature engineering (e.g., calculating PPG, APG, RPG)
   - Handling missing values and outliers

2. **Exploratory Data Analysis**:
   - Statistical analysis of salary trends since 2000
   - Visualization of key metrics and their relationships with salaries

3. **Advanced Analytics**:
   - Calculation of VORP to Salary Ratio for value analysis
   - K-means clustering for career trajectory analysis

4. **Machine Learning Models**:
   - Implementation of various models: Random Forest, Gradient Boosting, Ridge Regression, ElasticNet, SVR, Decision Tree
   - Hyperparameter tuning using GridSearchCV
   - Model evaluation using MSE and R² score

5. **Salary Predictions**:
   - Feature selection using Recursive Feature Elimination (RFE)
   - Prediction of future salaries based on historical data and performance metrics

6. **Overpaid/Underpaid Analysis**:
   - Comparison of actual salaries to model predictions
   - Identification of players with the largest discrepancies between actual and predicted salaries

## Results and Insights

1. **Salary Trends**: Analysis of how NBA salaries have evolved since 2000, accounting for factors such as inflation and changes in the salary cap.

2. **Performance-Salary Correlation**: Insights into how various performance metrics correlate with player salaries across different positions and career stages.

3. **Value Analysis**: Identification of players providing the best value based on their VORP to Salary Ratio.

4. **Career Trajectories**: Classification of players into distinct career trajectory clusters, providing insights into typical career progressions and their impact on salaries.

5. **Predictive Accuracy**: Evaluation of our salary prediction models, with a breakdown of performance across different player categories.

6. **Overpaid/Underpaid Players**: A list of the most overpaid and underpaid players based on our model predictions, with potential explanations for these discrepancies.

## Future Improvements

1. **Injury Data Integration**: Incorporate injury data to analyze its impact on player performance and salaries.

2. **Market Value Analysis**: Develop a model to estimate a player's market value based on performance, age, and league trends.

3. **Team Salary Cap Management**: Create tools to assist in team salary cap management and optimization.

4. **Player Development Projection**: Implement models to project player development trajectories and their impact on future salaries.

5. **External Factors**: Analyze the impact of external factors such as team market size, player popularity, and endorsement potential on salaries.

## Contributing

We welcome contributions to this project! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.