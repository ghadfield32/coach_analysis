
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import joblib
from sklearn.inspection import permutation_importance

# Load the data
data = pd.read_csv('../data/processed/final_salary_data_with_yos_and_cap.csv')

# Initial data inspection
# print("Initial Data Overview:")
# print(data.head())
# print("Initial Data Describe:")
# print(data.describe())
# print("\nData Info:")
# print(data.info())
# print("\nMissing Values:")
# print(data.isnull().sum())

# Drop the '2022 Dollars' column
data.drop(columns=['2022 Dollars'], inplace=True)

# Convert 'Season' to an integer
data['Season'] = data['Season'].str[:4].astype(int)

# Handle missing values for numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Feature engineering
data['PPG'] = data['PTS'] / data['GP']
data['APG'] = data['AST'] / data['GP']
data['RPG'] = data['TRB'] / data['GP']
data['SPG'] = data['STL'] / data['GP']
data['BPG'] = data['BLK'] / data['GP']
data['TOPG'] = data['TOV'] / data['GP']
data['WinPct'] = data['Wins'] / (data['Wins'] + data['Losses'])
data['SalaryGrowth'] = data['Salary'].pct_change().fillna(0)
data['Availability'] = data['GP'] / 82
data['SalaryPct'] = data['Salary'] / data['Salary Cap']


#for training the Salary_cap_inflated data 
# # Drop the '2022 Dollars' column
# data.drop(columns=['2022 Dollars', 'Salary Cap'], inplace=True)

# # Convert 'Season' to an integer
# data['Season'] = data['Season'].str[:4].astype(int)

# # Handle missing values for numerical columns
# numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# # Feature engineering
# data['PPG'] = data['PTS'] / data['GP']
# data['APG'] = data['AST'] / data['GP']
# data['RPG'] = data['TRB'] / data['GP']
# data['SPG'] = data['STL'] / data['GP']
# data['BPG'] = data['BLK'] / data['GP']
# data['TOPG'] = data['TOV'] / data['GP']
# data['WinPct'] = data['Wins'] / (data['Wins'] + data['Losses'])
# data['SalaryGrowth'] = data['Salary'].pct_change().fillna(0)
# data['Availability'] = data['GP'] / 82
# data['SalaryPct'] = data['Salary'] / data['Salary_Cap_Inflated']




# Identify categorical and numerical columns
categorical_cols = ['Player', 'Season', 'Position', 'Team']
numerical_cols = data.columns.difference(categorical_cols + ['Salary', 'SalaryPct'])

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cats = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

# Combine the numerical and encoded categorical data
data = pd.concat([data[numerical_cols], encoded_cats, data[['Player', 'Season', 'Salary', 'SalaryPct']]], axis=1)

# Select initial features
initial_features = ['Age', 'Years of Service', 'GP', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TOPG', 'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP', 'Availability'] + list(encoded_cats.columns)

# Create a new DataFrame with only the features we're interested in and the target variable
data_subset = data[initial_features + ['SalaryPct']].copy()

# Drop rows with any missing values
data_cleaned = data_subset.dropna()

# Separate features and target variable
X = data_cleaned[initial_features]
y = data_cleaned['SalaryPct']

# Perform feature selection
rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10)
rfe = rfe.fit(X, y)
selected_features = [feature for feature, selected in zip(initial_features, rfe.support_) if selected]

print("Selected features by RFE:", selected_features)

# Use only the selected features
X = data_cleaned[selected_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_filename = "../data/models/scaler.joblib"
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to '{scaler_filename}'")

# Define models with updated parameters
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge Regression': Ridge(),
    'ElasticNet': ElasticNet(max_iter=10000),
    'SVR': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Define parameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [8, 10, 12],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    },
    'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
    'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]},
    'Decision Tree': {'max_depth': [6, 8, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
}

# Save the selected features
selected_features_filename = "../data/models/selected_features.joblib"
joblib.dump(selected_features, selected_features_filename)
print(f"Selected features saved to '{selected_features_filename}'")

# Train and evaluate models
best_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_models[name] = grid_search.best_estimator_
    
    # Cross-validation
    cv_scores = cross_val_score(best_models[name], X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} - Best params: {grid_search.best_params_}")
    print(f"{name} - Cross-validation MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test set performance
    y_pred = best_models[name].predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Test MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Feature importance
    if name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        importances = best_models[name].feature_importances_
        feature_importance = pd.DataFrame({'feature': selected_features, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print(f"\n{name} - Top 5 important features:")
        print(feature_importance.head())
    else:
        perm_importance = permutation_importance(best_models[name], X_test_scaled, y_test, n_repeats=10, random_state=42)
        feature_importance = pd.DataFrame({'feature': selected_features, 'importance': perm_importance.importances_mean})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print(f"\n{name} - Top 5 important features (Permutation Importance):")
        print(feature_importance.head())
    
    # Save the model
    model_filename = f"../data/models/{name}_salary_prediction_model_inflated.joblib"
    joblib.dump(best_models[name], model_filename)
    print(f"{name} model saved to '{model_filename}'")


# Identify the best overall model
best_model_name = min(best_models, key=lambda x: -cross_val_score(best_models[x], X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error').mean())
best_model = best_models[best_model_name]

print(f"Best overall model: {best_model_name}")

# Create predictions dataset
all_players = data['Player'].unique()
predictions = []

for player in all_players:
    player_data = data[data['Player'] == player]
    latest_season = player_data['Season'].max()
    next_season_data = player_data[player_data['Season'] == latest_season].copy()
    next_season_data['Age'] += 1
    next_season_data['Season'] += 1
    next_season_data_scaled = scaler.transform(next_season_data[selected_features])
    predicted_salary_pct = best_model.predict(next_season_data_scaled)[0]
    predicted_salary = predicted_salary_pct * next_season_data['Salary Cap'].values[0]
    
    predictions.append({
        'Player': player,
        'Predicted_Season': int(next_season_data['Season'].values[0]),
        'Age': int(next_season_data['Age'].values[0]),
        'Predicted_Salary_Pct': predicted_salary_pct,
        'Predicted_Salary': predicted_salary,
        'Previous_Season_Salary': player_data[player_data['Season'] == latest_season]['Salary'].values[0],
        'Salary_Change': predicted_salary - player_data[player_data['Season'] == latest_season]['Salary'].values[0]
    })

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('../data/predictions/salary_predictions.csv', index=False)
print("Predictions saved to '../data/predictions/salary_predictions.csv'")

# Display some statistics about the predictions
print("\nPrediction Statistics:")
print(predictions_df[['Predicted_Salary_Pct', 'Predicted_Salary', 'Salary_Change']].describe())

# Plot top 10 highest predicted salaries
top_10_salaries = predictions_df.nlargest(10, 'Predicted_Salary')
plt.figure(figsize=(12, 6))
sns.barplot(x='Player', y='Predicted_Salary', data=top_10_salaries)
plt.title("Top 10 Highest Predicted Salaries")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Predicted Salary ($)")
plt.tight_layout()
plt.savefig('../data/predictions/top_10_predicted_salaries.png')
print("Top 10 predicted salaries plot saved to '../data/predictions/top_10_predicted_salaries.png'")

# Plot distribution of salary changes
plt.figure(figsize=(12, 6))
sns.histplot(predictions_df['Salary_Change'], kde=True)
plt.title("Distribution of Predicted Salary Changes")
plt.xlabel("Salary Change ($)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('../data/predictions/salary_change_distribution.png')
print("Salary change distribution plot saved to '../data/predictions/salary_change_distribution.png'")
