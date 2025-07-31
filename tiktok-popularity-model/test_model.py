#!/usr/bin/env python3
"""
Test script for TikTok Popularity Model
Demonstrates model training and evaluation with console output
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

def create_sample_data():
    """Create sample TikTok songs data for testing"""
    print("Creating sample dataset...")
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
    
    df = pd.DataFrame(data)
    print(f"Dataset created with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n=== DATA PREPROCESSING ===")
    print(f"Original dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Handle missing values
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
    print(f"Missing values before preprocessing: {missing_before}")
    print(f"Missing values after preprocessing: {missing_after}")
    
    return df

def encode_features(df):
    """Encode categorical and scale numerical features"""
    print("\n=== FEATURE ENCODING ===")
    
    # Prepare features
    categorical_features = ['artist_name']
    numerical_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
    
    # Filter features that exist in the dataframe
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    df_processed = df.copy()
    
    # Initialize encoders
    label_encoder = LabelEncoder()
    scaler = StandardScaler()
    
    # One-hot encode categorical features (using LabelEncoder for simplicity)
    if categorical_features:
        for feature in categorical_features:
            print(f"Encoding categorical feature: {feature}")
            df_processed[feature + '_encoded'] = label_encoder.fit_transform(df_processed[feature])
            df_processed = df_processed.drop(feature, axis=1)
    
    # Scale numerical features
    if numerical_features:
        print(f"Scaling numerical features: {numerical_features}")
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    # Save encoders
    os.makedirs('/workspace/xc', exist_ok=True)
    joblib.dump(label_encoder, '/workspace/xc/encoder.joblib')
    joblib.dump(scaler, '/workspace/xc/scaler.joblib')
    print("Encoders saved successfully!")
    
    return df_processed, label_encoder, scaler

def train_model(df):
    """Train the RandomForest model"""
    print("\n=== MODEL TRAINING ===")
    
    # Prepare data
    if 'popularity' not in df.columns:
        print("ERROR: Target variable 'popularity' not found in dataset")
        return None, None, None, None
    
    # Encode features
    df_processed, label_encoder, scaler = encode_features(df)
    
    # Separate features and target
    target_col = 'popularity'
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Train model with specified hyperparameters
    print("\nTraining RandomForestRegressor...")
    print("Hyperparameters: n_estimators=100, random_state=42")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Display metrics
    print("\n=== MODEL EVALUATION ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    
    # Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Save model
    model_path = '/workspace/xc/model.joblib'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, X_train.columns.tolist(), label_encoder, scaler

def make_sample_predictions(model, feature_names, label_encoder, scaler):
    """Make sample predictions to demonstrate functionality"""
    print("\n=== SAMPLE PREDICTIONS ===")
    
    if model is None:
        print("ERROR: Model not available")
        return
    
    # Sample input data
    sample_data = [
        {
            'artist_name': 'Taylor Swift',
            'danceability': 0.7,
            'energy': 0.8,
            'acousticness': 0.2,
            'valence': 0.6,
            'tempo': 120.0
        },
        {
            'artist_name': 'Drake',
            'danceability': 0.9,
            'energy': 0.6,
            'acousticness': 0.1,
            'valence': 0.4,
            'tempo': 140.0
        },
        {
            'artist_name': 'Billie Eilish',
            'danceability': 0.5,
            'energy': 0.3,
            'acousticness': 0.8,
            'valence': 0.3,
            'tempo': 90.0
        }
    ]
    
    for i, data in enumerate(sample_data, 1):
        print(f"\nSample {i}:")
        print(f"Artist: {data['artist_name']}")
        print(f"Danceability: {data['danceability']}")
        print(f"Energy: {data['energy']}")
        print(f"Acousticness: {data['acousticness']}")
        print(f"Valence: {data['valence']}")
        print(f"Tempo: {data['tempo']}")
        
        # Prepare input data
        input_df = pd.DataFrame([data])
        
        # Encode categorical features
        try:
            input_df['artist_name_encoded'] = label_encoder.transform(input_df['artist_name'])
        except ValueError:
            # Handle unseen categories
            input_df['artist_name_encoded'] = 0
        
        input_df = input_df.drop('artist_name', axis=1)
        
        # Scale numerical features
        numerical_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Ensure all required features are present and in correct order
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        print(f"Predicted Popularity: {prediction:.1f}%")

def main():
    """Main function"""
    print("🎵 TikTok Popularity Prediction Model")
    print("=" * 50)
    
    # Load or create data
    try:
        # Try to load Kaggle data (this might fail in some environments)
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        print("Attempting to load Kaggle dataset...")
        file_path = "songs_normalize.csv"
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sveta151/tiktok-popular-songs-2022",
            file_path,
        )
        print(f"Kaggle dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Could not load Kaggle dataset: {str(e)}")
        print("Using sample data instead...")
        df = create_sample_data()
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Train model
    model, feature_names, label_encoder, scaler = train_model(df_clean)
    
    # Make sample predictions
    if model is not None:
        make_sample_predictions(model, feature_names, label_encoder, scaler)
    
    print("\n" + "=" * 50)
    print("Model training and evaluation completed!")
    print("You can now run the Streamlit app with:")
    print("streamlit run app.py --server.port 8501 --server.address 0.0.0.0")

if __name__ == "__main__":
    main()