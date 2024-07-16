
import streamlit as st
import pandas as pd
from data.data_loader import fetch_shots_data
from features.feature_engineering import engineer_features, prepare_features
from models.model_trainer import train_model, save_model

def run():
    st.header("Model Training")
    
    team_name = st.selectbox("Select Team for Training", ["Boston Celtics", "Los Angeles Lakers", "Golden State Warriors"])
    season = st.selectbox("Select Season for Training", ["2023-24", "2022-23", "2021-22"])
    
    if st.button("Train Model"):
        with st.spinner("Fetching data and training model..."):
            shots = fetch_shots_data(team_name, True, season)
            shots_encoded = engineer_features(shots)
            X, y = prepare_features(shots_encoded)
            model, scaler, imputer = train_model(X, y)
            
            save_model(model, scaler, imputer, f'../models/{team_name.replace(" ", "_")}_{season}_model.joblib')
        
        st.success(f"Model for {team_name} ({season}) trained and saved successfully!")
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.best_estimator_.named_steps['votingclassifier'].estimators_[0].feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        st.bar_chart(feature_importance.set_index('feature'))
