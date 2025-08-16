import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def format_season(data):
    try:
        data['Season'] = data['Season'].apply(lambda x: int(x.split('-')[0]))
        logger.info(f"Seasons in data: {data['Season'].unique()}")
        return data
    except Exception as e:
        logger.error(f"Failed to format season data: {e}")
        raise

def clean_data(data, nan_filter_percentage: float = 0.01):
    """
    Clean the raw DataFrame with a *threshold-aware* NaN filter.

    Behavior:
      • Drop a few unneeded columns (errors='ignore' kept).
      • Fill selected percentage columns with their mean (existing behavior).
      • NEW: Compute per-column missing rate and drop rows *only for columns
        whose missing rate <= `nan_filter_percentage`*. This preserves data
        when a column has substantial missingness, and removes "cheap" NaNs.
      • Remove rows that are entirely NaN (safety).

    Args
    ----
    data : pd.DataFrame
        Input dataset.
    nan_filter_percentage : float, default 0.01 (1%)
        Columns with NaN rate in (0, nan_filter_percentage] are considered
        "low-missing", and rows with NaNs in those columns are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with low-missing NaNs removed and light fills applied.
    """
    try:
        data_clean = data.copy()

        # 1) Drop clearly unneeded columns (keep your existing list)
        columns_to_drop = ['Injury_Periods', '2nd Apron', 'Wins', 'Losses']
        data_clean.drop(columns_to_drop, axis=1, errors='ignore', inplace=True)

        # 2) Fill a few % columns with mean (existing behavior)
        percentage_cols = ['3P%', '2P%', 'FT%', 'TS%']
        for col in percentage_cols:
            if col in data_clean.columns:
                # Cast to float defensively; fill only that column's NaNs
                data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
                data_clean[col] = data_clean[col].fillna(data_clean[col].mean())

        # 3) NEW: threshold-aware NaN filtering
        if not (0.0 <= float(nan_filter_percentage) <= 1.0):
            raise ValueError("nan_filter_percentage must be in [0.0, 1.0]")

        null_pct = data_clean.isna().mean()  # per-column NaN fraction
        cols_to_strict_drop = null_pct[(null_pct > 0) & (null_pct <= nan_filter_percentage)].index.tolist()

        before_rows = len(data_clean)
        if cols_to_strict_drop:
            data_clean = data_clean.dropna(subset=cols_to_strict_drop)
            dropped = before_rows - len(data_clean)
            logger.info(
                f"Threshold NaN filter: dropped {dropped} rows based on "
                f"{len(cols_to_strict_drop)} low-missing columns "
                f"(<= {nan_filter_percentage*100:.2f}%)."
            )
            logger.debug(f"Low-missing columns used for drop: {cols_to_strict_drop}")
        else:
            logger.info(
                "Threshold NaN filter: no columns at or below the threshold; no rows dropped."
            )

        # 4) Light cleanup: drop *entirely* empty rows, if any
        data_clean.dropna(axis=0, how='all', inplace=True)

        logger.info(f"Data cleaned. Remaining shape: {data_clean.shape}")
        return data_clean

    except Exception as e:
        logger.error(f"Failed to clean data: {e}")
        raise

def engineer_features(data):
    # Calculate per-game statistics to normalize performance data
    per_game_cols = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV']
    for col in per_game_cols:
        data[f'{col[0]}PG'] = data[col] / data['GP']
    
    # Derive additional features to capture important aspects of a player's performance
    data['Availability'] = data['GP'] / 82
    
    # Handle salary cap calculation (column name may vary)
    if 'Salary_Cap_Inflated' in data.columns:
        data['SalaryPct'] = data['Salary'] / data['Salary_Cap_Inflated']
    elif 'Salary_Cap' in data.columns:
        data['SalaryPct'] = data['Salary'] / data['Salary_Cap']
    else:
        # Fallback: use a reasonable default cap value
        data['SalaryPct'] = data['Salary'] / 136000000  # Approximate 2023-24 cap
    
    data['Efficiency'] = (data['PTS'] + data['TRB'] + data['AST'] + data['STL'] + data['BLK']) / (data['FGA'] + data['FTA'] + data['TOV'] + 1)
    
    # Handle VORP-based features (may not be available)
    if 'VORP' in data.columns:
        data['ValueOverReplacement'] = data['VORP'] / (data['Salary'] + 1)
    else:
        data['ValueOverReplacement'] = 0  # Default if VORP not available
    
    # Handle Years of Service (column name may vary)
    if 'Years of Service' in data.columns:
        data['ExperienceSquared'] = data['Years of Service'] ** 2
    elif 'Years_of_Service' in data.columns:
        data['ExperienceSquared'] = data['Years_of_Service'] ** 2
    else:
        data['ExperienceSquared'] = 0  # Default if not available
    
    # Handle injury-related features
    if 'Total_Days_Injured' in data.columns:
        data['Days_Injured_Percentage'] = data['Total_Days_Injured'] / data['GP']
    else:
        data['Days_Injured_Percentage'] = 0  # Default if not available
    
    # Handle Win Shares features
    if 'WS' in data.columns:
        data['WSPG'] = data['WS'] / data['GP']
    else:
        data['WSPG'] = 0
    
    if 'DWS' in data.columns:
        data['DWSPG'] = data['DWS'] / data['GP']
    else:
        data['DWSPG'] = 0
    
    if 'OWS' in data.columns:
        data['OWSPG'] = data['OWS'] / data['GP']
    else:
        data['OWSPG'] = 0
    # Handle per-game statistics with defensive checks
    if 'PF' in data.columns:
        data['PFPG'] = data['PF'] / data['GP']
    else:
        data['PFPG'] = 0
    
    if 'ORB' in data.columns:
        data['ORPG'] = data['ORB'] / data['GP']
    else:
        data['ORPG'] = 0
    
    if 'DRB' in data.columns:
        data['DRPG'] = data['DRB'] / data['GP']
    else:
        data['DRPG'] = 0
    
    # Drop columns used in feature creation or deemed less relevant
    columns_to_drop = ['GP', '2PA', 'OBPM', 'BPM', 'DBPM', '2P', 'GS', 'PTS', 'AST', 'TRB', 'STL', 'BLK',
                       'TOV', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB',
                       'TS%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'Luxury Tax', '1st Apron',
                       'Standard /Non-Taxpayer', 'Taxpayer', 'Team Room /Under Cap', 'WS', 'DWS', 'WS/48', 'PF', 'OWS', 'Injured']
    # Note: BAE removed from drop list to preserve cap-related data
    data.drop(columns_to_drop, axis=1, errors='ignore', inplace=True)
    print("New features added.")
    return data

def encode_injury_risk(data):
    # Encode injury risk levels for model training
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    data['Injury_Risk'] = data['Injury_Risk'].map(risk_mapping).fillna(0)  # Default to Low if unknown
    return data, risk_mapping

def encode_categorical(data, columns):
    # Encode categorical columns using one-hot encoding
    encoders = {}
    for col in columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(data[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=data.index)
        data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)
        encoders[col] = encoder
    return data, encoders


def encode_data(data, encoders=None, player_encoder=None):
    """
    Encode Injury_Risk, Player, and categorical features, and scale numerics.
    NEW:
      • Fill categorical ('Position','Team') missing with 'Unknown' before one-hot.
      • Impute numeric columns with median via SimpleImputer before StandardScaler,
        so the pipeline remains robust when higher-missing columns (> threshold)
        were intentionally kept during cleaning.
    """
    from sklearn.impute import SimpleImputer  # local import to avoid changing top-level imports
    print("Columns before encoding:", data.columns)

    # Encode Injury_Risk
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    data['Injury_Risk'] = data['Injury_Risk'].map(risk_mapping).fillna(0)  # Default to Low if unknown

    # Encode Player column if it's present
    if 'Player' in data.columns:
        if player_encoder is None:
            player_encoder = LabelEncoder()
            data['Player_Encoded'] = player_encoder.fit_transform(data['Player'])
        else:
            data['Player_Encoded'] = player_encoder.transform(data['Player'])
        data.drop('Player', axis=1, inplace=True)  # Drop original Player column after encoding

    # Identify initial numeric columns (before one-hot)
    initial_numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Categorical variables — fill missing *before* one-hot
    categorical_cols = ['Position', 'Team']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')

    # One-hot encode categoricals
    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            if col in data.columns:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(data[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoder.get_feature_names_out([col]),
                    index=data.index
                )
                data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)
                encoders[col] = encoder
    else:
        for col in categorical_cols:
            if col in data.columns and col in encoders:
                encoded = encoders[col].transform(data[[col]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoders[col].get_feature_names_out([col]),
                    index=data.index
                )
                data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)

    # Determine final numeric cols (excluding encoded/categorical artifacts)
    numeric_cols = [c for c in data.select_dtypes(include=['int64', 'float64']).columns
                    if c not in ['Season', 'Injury_Risk', 'Player_Encoded']]

    # NEW: Impute numerics before scaling (robust to remaining NaNs)
    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

    # Scale numeric features (excluding 'Player_Encoded')
    scaler = StandardScaler()
    if numeric_cols:
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    print("Encoded data shape:", data.shape)
    print("Columns after encoding:", data.columns)

    # NOTE: We keep the return signature unchanged for compatibility.
    return data, risk_mapping, encoders, scaler, numeric_cols, player_encoder



def scale_features(data, numeric_cols):
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data, scaler

def decode_data(encoded_data, injury_risk_mapping, encoders, scaler, numeric_cols, player_encoder):
    decoded_data = encoded_data.copy()
    
    # Decode Injury_Risk
    inv_injury_risk_mapping = {v: k for k, v in injury_risk_mapping.items()}
    decoded_data['Injury_Risk'] = decoded_data['Injury_Risk'].map(inv_injury_risk_mapping)
    
    # Decode Player column
    if 'Player_Encoded' in decoded_data.columns:
        decoded_data['Player'] = player_encoder.inverse_transform(decoded_data['Player_Encoded'])
        decoded_data.drop('Player_Encoded', axis=1, inplace=True)
    
    # Decode categorical variables
    for col, encoder in encoders.items():
        encoded_cols = [c for c in decoded_data.columns if c.startswith(f"{col}_")]
        decoded_col = encoder.inverse_transform(decoded_data[encoded_cols])
        decoded_data[col] = decoded_col.ravel()  # Flatten the 2D array to 1D
        decoded_data.drop(encoded_cols, axis=1, inplace=True)
    
    # Inverse transform scaled features
    decoded_data[numeric_cols] = scaler.inverse_transform(decoded_data[numeric_cols])
    
    return decoded_data

def select_top_features(X, y, k=10):
    # Select top features based on statistical significance
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    top_features = X.columns[selector.get_support()].tolist()
    print(f"Top {k} features:", top_features)
    return top_features

def calculate_tree_feature_importance(X, y):
    # Calculate feature importance using a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
    plt.title('Top 20 Feature Importances from Random Forest')
    plt.show()
    
    return feature_importances

if __name__ == "__main__":
    try:
        file_path = '../data/processed/nba_player_data_final_inflated.csv'
        data = load_data(file_path)
        data = format_season(data)
        data = clean_data(data)
        data = engineer_features(data)

        # Separate features and target
        X = data.drop(['SalaryPct', 'Salary'], axis=1)
        y = data['SalaryPct']

        # Encode data
        encoded_data, injury_risk_mapping, encoders, scaler, numeric_cols, player_encoder = encode_data(X)
        
        logger.info("Data preprocessing completed. Ready for model training.")
        

        print("\nInjury Risk Mapping:", injury_risk_mapping)
        print("Encoded Injury Risk range:", encoded_data['Injury_Risk'].min(), "-", encoded_data['Injury_Risk'].max())
        print("\nNumeric columns for scaling:", numeric_cols)

        # Calculate feature importance
        feature_importances = calculate_tree_feature_importance(encoded_data, y)
        print("\nTree-based feature importances:")
        print(feature_importances.head(20))

        # Select top features
        top_features = select_top_features(encoded_data, y)
        print("\nTop features selected using statistical methods:", top_features)

        # Decoding example
        print("\nDecoding Example:")
        decoded_data = decode_data(encoded_data, injury_risk_mapping, encoders, scaler, numeric_cols, player_encoder)
        
        print("\nFirst few rows of decoded data:")
        print(decoded_data[['Player', 'Injury_Risk', 'Position', 'Team', 'Season'] + top_features].head())

        print("\nData types after decoding:")
        print(decoded_data.dtypes)

        print("\nData preprocessing completed. Ready for model training.")
        
    except Exception as e:
        logger.critical(f"Critical error in data processing pipeline: {e}")
        raise

