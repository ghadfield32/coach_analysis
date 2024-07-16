
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data.data_loader import fetch_shots_data
from features.feature_engineering import engineer_features, prepare_features
from models.model_trainer import load_model
from models.model_predictor import predict_shot
from visualization.plot_utils import plot_court, plot_shots_with_predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run():
    st.header("Shot Prediction")
    
    team_name = st.selectbox("Select Team", ["Boston Celtics", "Los Angeles Lakers", "Golden State Warriors"])
    season = st.selectbox("Select Season", ["2023-24", "2022-23", "2021-22"])
    
    shots = fetch_shots_data(team_name, True, season)
    game_ids = shots['GAME_ID'].unique()
    selected_game = st.selectbox("Select Game for Prediction", game_ids)
    
    if st.button("Predict Shots"):
        with st.spinner("Loading model and predicting shots..."):
            model, scaler, imputer = load_model(f'../models/{team_name.replace(" ", "_")}_{season}_model.joblib')
            
            train_shots = shots[shots['GAME_ID'] != selected_game]
            test_shots = shots[shots['GAME_ID'] == selected_game]
            
            train_encoded = engineer_features(train_shots)
            X_train, y_train = prepare_features(train_encoded)
            
            test_encoded = engineer_features(test_shots)
            X_test, y_test = prepare_features(test_encoded)
            
            predictions = predict_shot(model, scaler, imputer, test_shots, X_train.columns)
        
        st.success("Predictions complete!")
        
        fig, ax = plt.subplots(figsize=(12, 11))
        plot_shots_with_predictions(predictions)
        st.pyplot(fig)
        
        st.subheader("Model Performance")
        accuracy = accuracy_score(y_test, predictions['PREDICTION'])
        precision = precision_score(y_test, predictions['PREDICTION'])
        recall = recall_score(y_test, predictions['PREDICTION'])
        f1 = f1_score(y_test, predictions['PREDICTION'])
        roc_auc = roc_auc_score(y_test, predictions['PREDICTION_PROB'])
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Value': [accuracy, precision, recall, f1, roc_auc]
        })
        st.table(metrics_df)
        
        st.subheader("Comparison with Actual Shots")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        plot_court(ax1)
        ax1.scatter(predictions[predictions['SHOT_MADE_FLAG'] == 1]['LOC_X'], 
                    predictions[predictions['SHOT_MADE_FLAG'] == 1]['LOC_Y'], 
                    color='green', alpha=0.7, label='Made')
        ax1.scatter(predictions[predictions['SHOT_MADE_FLAG'] == 0]['LOC_X'], 
                    predictions[predictions['SHOT_MADE_FLAG'] == 0]['LOC_Y'], 
                    color='red', alpha=0.7, label='Missed')
        ax1.set_title("Actual Shots")
        ax1.legend()
        
        plot_court(ax2)
        ax2.scatter(predictions[predictions['PREDICTION'] == 1]['LOC_X'], 
                    predictions[predictions['PREDICTION'] == 1]['LOC_Y'], 
                    color='green', alpha=0.7, label='Predicted Made')
        ax2.scatter(predictions[predictions['PREDICTION'] == 0]['LOC_X'], 
                    predictions[predictions['PREDICTION'] == 0]['LOC_Y'], 
                    color='red', alpha=0.7, label='Predicted Missed')
        ax2.set_title("Predicted Shots")
        ax2.legend()
        
        st.pyplot(fig)
