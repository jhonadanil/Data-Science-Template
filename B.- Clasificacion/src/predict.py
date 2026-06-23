#!/usr/bin/env python3
"""Inference script: load trained model and predict on new data.

Usage:
    python src/predict.py path/to/new_data.csv
    python src/predict.py  # uses a sample from the training data
"""

import sys
import os
import pickle
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import SELECTED_FEATURES

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')


def load_artifacts():
    """Load trained model, scaler, and encoder from disk."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return model, scaler, encoder


def predict(model, scaler, encoder, data):
    """Make predictions on new data.

    Args:
        data: DataFrame or array-like with SELECTED_FEATURES columns.
    Returns:
        predictions: array of predicted class labels (original encoding).
        probabilities: array of probabilities for the positive class.
    """
    # Ensure only selected features, in correct order
    if isinstance(data, pd.DataFrame):
        X = data[SELECTED_FEATURES]
    else:
        X = pd.DataFrame(data, columns=SELECTED_FEATURES)

    X_scaled = scaler.transform(X)
    y_pred_enc = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = encoder.inverse_transform(y_pred_enc)
    return y_pred, y_proba


if __name__ == '__main__':
    # Load model artifacts
    print('Loading model artifacts...')
    model, scaler, encoder = load_artifacts()
    print(f'Model loaded. Classes: {list(encoder.classes_)}')

    if len(sys.argv) > 1:
        # Predict on provided CSV file
        data_path = sys.argv[1]
        print(f'Loading data from {data_path}...')
        df_new = pd.read_csv(data_path)
    else:
        # Use a sample from training data for demonstration
        DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'breast-cancer.csv')
        df_new = pd.read_csv(DATA_PATH).head(10)
        print('No file provided. Using first 10 rows of training data as example.')

    predictions, probabilities = predict(model, scaler, encoder, df_new)

    print('
Predictions:')
    result = pd.DataFrame({
        'prediction': predictions,
        'probability_malignant': probabilities,
    })
    print(result.to_string(index=False))
