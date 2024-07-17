# NBA Salary Analysis and Prediction Project

**[Streamlit_App](https://nba-salary-predictions.streamlit.app/)

## Table of Contents
1. [Introduction](#introduction)
2. [Data Sources](#data-sources)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Features](#features)
7. [Methodology](#methodology)
8. [Results and Insights](#results-and-insights)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)

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