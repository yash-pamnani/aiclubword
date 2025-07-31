#!/usr/bin/env python3
"""
TikTok Popularity Prediction App
A complete machine learning application for predicting song popularity on TikTok
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Configure page
st.set_page_config(
    page_title="TikTok Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

# Global variables
MODEL_PATH = "/workspace/xc/model.joblib"
SCALER_PATH = "/workspace/xc/scaler.joblib"
ENCODER_PATH = "/workspace/xc/encoder.joblib"

def load_data():
    """Load the TikTok songs dataset using KaggleHub"""
    try:
        # Load the dataset from Kaggle
        file_path = "songs_normalize.csv"
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sveta151/tiktok-popular-songs-2022",
            file_path,
        )
        
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Fallback: create sample data for demonstration
        st.warning("Using sample data for demonstration purposes")
        return create_sample_data()

def create_sample_data():
    """Create sample data if Kaggle dataset is not available"""
    np.random.seed(42)
    n_samples = 1000
    
    artists = ['Taylor Swift', 'Drake', 'Ariana Grande', 'The Weeknd', 'Billie Eilish', 
               'Post Malone', 'Ed Sheeran', 'Dua Lipa', 'Harry Styles', 'Olivia Rodrigo']
    
    data = {
        'artist_name': np.random.choice(artists, n_samples),
        'danceability': np.random.uniform(0, 1, n_samples),
        'energy': np.random.uniform(0, 1, n_samples),
        'acousticness': np.random.uniform(0, 1, n_samples),
        'valence': np.random.uniform(0, 1, n_samples),
        'tempo': np.random.uniform(50, 200, n_samples),
        'popularity': np.random.uniform(0, 100, n_samples)
    }
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the dataset"""
    st.subheader("Data Preprocessing")
    
    # Display basic info
    st.write("**Dataset Info:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Handle missing values
    st.write("**Handling Missing Values:**")
    missing_before = df.isnull().sum().sum()
    
    # Drop rows with missing target variable
    if 'popularity' in df.columns:
        df = df.dropna(subset=['popularity'])
    
    # Impute missing values for numerical features
    numerical_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].median())
    
    # Handle missing artist names
    if 'artist_name' in df.columns:
        df['artist_name'] = df['artist_name'].fillna('Unknown Artist')
    
    missing_after = df.isnull().sum().sum()
    st.write(f"Missing values before: {missing_before}, after: {missing_after}")
    
    return df

def encode_features(df, fit_encoders=True):
    """Encode categorical and scale numerical features"""
    
    # Prepare features
    categorical_features = ['artist_name']
    numerical_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
    
    # Filter features that exist in the dataframe
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    df_processed = df.copy()
    
    if fit_encoders:
        # Initialize encoders
        label_encoder = LabelEncoder()
        scaler = StandardScaler()
        
        # One-hot encode categorical features (using LabelEncoder for simplicity)
        if categorical_features:
            for feature in categorical_features:
                df_processed[feature + '_encoded'] = label_encoder.fit_transform(df_processed[feature])
                df_processed = df_processed.drop(feature, axis=1)
        
        # Scale numerical features
        if numerical_features:
            df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
        
        # Save encoders
        joblib.dump(label_encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        return df_processed, label_encoder, scaler
    else:
        # Load existing encoders
        label_encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Apply encoding
        if categorical_features:
            for feature in categorical_features:
                try:
                    df_processed[feature + '_encoded'] = label_encoder.transform(df_processed[feature])
                    df_processed = df_processed.drop(feature, axis=1)
                except ValueError:
                    # Handle unseen categories
                    df_processed[feature + '_encoded'] = 0
                    df_processed = df_processed.drop(feature, axis=1)
        
        # Apply scaling
        if numerical_features:
            df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
        
        return df_processed, label_encoder, scaler

def train_model(df):
    """Train the RandomForest model"""
    st.subheader("Model Training")
    
    # Prepare data
    if 'popularity' not in df.columns:
        st.error("Target variable 'popularity' not found in dataset")
        return None, None, None, None
    
    # Encode features
    df_processed, label_encoder, scaler = encode_features(df, fit_encoders=True)
    
    # Separate features and target
    target_col = 'popularity'
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    st.write(f"Training set size: {X_train.shape[0]}")
    st.write(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Display metrics
    st.write("**Model Performance:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.4f}")
    with col3:
        st.metric("MSE", f"{mse:.4f}")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    st.success("Model saved successfully!")
    
    return model, X_train.columns.tolist(), label_encoder, scaler

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    if model is None:
        return
    
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.grid(axis='x', alpha=0.3)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    return fig

def create_prediction_interface(df, model, feature_names, label_encoder, scaler):
    """Create the Streamlit prediction interface"""
    st.subheader("🎵 Make Predictions")
    
    if model is None:
        st.error("Model not available. Please train the model first.")
        return
    
    # Sidebar inputs
    st.sidebar.header("Input Features")
    
    # Artist dropdown
    unique_artists = df['artist_name'].unique() if 'artist_name' in df.columns else ['Unknown Artist']
    selected_artist = st.sidebar.selectbox("Artist Name", unique_artists)
    
    # Numerical feature sliders
    danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
    energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5, 0.01)
    acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
    valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5, 0.01)
    tempo = st.sidebar.slider("Tempo", 50.0, 200.0, 120.0, 1.0)
    
    # Predict button
    if st.sidebar.button("🎯 Predict Popularity", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'artist_name': [selected_artist],
            'danceability': [danceability],
            'energy': [energy],
            'acousticness': [acousticness],
            'valence': [valence],
            'tempo': [tempo]
        })
        
        # Encode input data
        input_processed, _, _ = encode_features(input_data, fit_encoders=False)
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_processed.columns:
                input_processed[feature] = 0
        
        # Reorder columns to match training data
        input_processed = input_processed[feature_names]
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        
        # Display prediction
        st.success(f"**Predicted Popularity: {prediction:.1f}%**")
        
        # Create a gauge-like visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(['Popularity'], [prediction], color='#1f77b4', height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Popularity (%)')
        ax.set_title(f'Predicted Popularity: {prediction:.1f}%')
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate([prediction]):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        st.pyplot(fig)
        plt.close()

def main():
    """Main application function"""
    st.title("🎵 TikTok Popularity Predictor")
    st.markdown("---")
    
    # Load data
    st.header("📊 Data Loading")
    df = load_data()
    
    if df is not None:
        # Display data preview
        with st.expander("View Data Preview"):
            st.dataframe(df.head())
            st.write(f"Dataset shape: {df.shape}")
        
        # Preprocess data
        df_clean = preprocess_data(df)
        
        # Train model
        model, feature_names, label_encoder, scaler = train_model(df_clean)
        
        # Create two columns for main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Feature importance plot
            if model is not None and feature_names is not None:
                st.subheader("📈 Feature Importance")
                fig = plot_feature_importance(model, feature_names)
                if fig:
                    st.pyplot(fig)
                    plt.close()
        
        with col2:
            # Prediction interface
            create_prediction_interface(df, model, feature_names, label_encoder, scaler)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and scikit-learn")

if __name__ == "__main__":
    # Console output for model metrics (when run directly)
    if not hasattr(st, '_is_running_with_streamlit'):
        print("Loading and training model...")
        df = load_data()
        if df is not None:
            df_clean = preprocess_data(df)
            model, feature_names, label_encoder, scaler = train_model(df_clean)
            print("Model training completed!")
    else:
        main()