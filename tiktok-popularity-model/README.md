# 🎵 TikTok Popularity Predictor

A comprehensive machine learning application that predicts the popularity of songs on TikTok using audio features and artist information.

## Features

- **Data Loading**: Loads TikTok songs dataset from Kaggle using KaggleHub
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features
- **Model Training**: Uses RandomForestRegressor with specified hyperparameters
- **Model Evaluation**: Provides R², MAE, and MSE metrics
- **Interactive UI**: Streamlit-based web interface for making predictions
- **Feature Importance**: Visualizes which features are most important for predictions

## Dataset Features

- **Artist Name**: Categorical feature (one-hot encoded)
- **Numerical Features**:
  - `danceability`: How suitable a track is for dancing (0.0 to 1.0)
  - `energy`: Perceptual measure of intensity and power (0.0 to 1.0)
  - `acousticness`: Confidence measure of whether the track is acoustic (0.0 to 1.0)
  - `valence`: Musical positiveness conveyed by a track (0.0 to 1.0)
  - `tempo`: Overall estimated tempo of a track in beats per minute (BPM)

## Model Specifications

- **Algorithm**: RandomForestRegressor
- **Hyperparameters**: 
  - `n_estimators`: 100
  - `random_state`: 42
- **Data Split**: 80% training, 20% testing
- **Preprocessing**: StandardScaler for numerical features, LabelEncoder for categorical features

## Installation

1. Install dependencies:
```bash
pip install pandas scikit-learn streamlit joblib matplotlib seaborn kagglehub
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Running the Console Version

To see model training and evaluation metrics in the console:

```bash
python3 test_model.py
```

## Web Interface Features

### Sidebar Controls
- **Artist Dropdown**: Select from available artists in the dataset
- **Feature Sliders**: Adjust audio features:
  - Danceability (0.0 - 1.0)
  - Energy (0.0 - 1.0)
  - Acousticness (0.0 - 1.0)
  - Valence (0.0 - 1.0)
  - Tempo (50 - 200 BPM)

### Main Interface
- **Predict Button**: Generate popularity prediction as a percentage
- **Feature Importance Chart**: Bar chart showing which features contribute most to predictions
- **Data Preview**: Expandable section showing dataset information
- **Model Metrics**: R², MAE, and MSE displayed after training

## File Structure

```
tiktok-popularity-model/
├── app.py              # Main Streamlit application
├── test_model.py       # Console-based model testing
├── requirements.txt    # Python dependencies
└── README.md          # This file

/workspace/xc/
├── model.joblib       # Trained RandomForest model
├── scaler.joblib      # StandardScaler for numerical features
└── encoder.joblib     # LabelEncoder for categorical features
```

## Model Performance

The model provides the following evaluation metrics:
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

## Sample Predictions

The application can make predictions for different artist-feature combinations:

Example:
- **Artist**: Taylor Swift
- **Danceability**: 0.7
- **Energy**: 0.8
- **Acousticness**: 0.2
- **Valence**: 0.6
- **Tempo**: 120 BPM
- **Predicted Popularity**: ~51%

## Data Source

The application attempts to load data from the Kaggle dataset "TikTok Popular Songs 2022" by sveta151. If the dataset is not available, it generates sample data for demonstration purposes.

## Technical Details

- **Framework**: Streamlit for web interface
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

## Error Handling

- Graceful fallback to sample data if Kaggle dataset is unavailable
- Handles missing values through imputation and dropping
- Manages unseen categorical values during prediction
- Comprehensive error messages and user feedback

## Future Enhancements

- Support for additional audio features
- Multiple model comparison
- Real-time data updates
- Export predictions to CSV
- Model retraining capabilities