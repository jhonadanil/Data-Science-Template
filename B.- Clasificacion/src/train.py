#!/usr/bin/env python3
"""Final training and evaluation script for breast cancer classification.

Loads configuration from src/config.py, runs the full pipeline,
trains the best LogisticRegression model, and evaluates it.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from src.config import TARGET, SELECTED_FEATURES, BEST_PARAMS
from preprocessing.split_data import dividir_datos
from preprocessing.pipeline import estandarizar, codificar

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'breast-cancer.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('Loading data...')
df = pd.read_csv(DATA_PATH)

# Drop the non-predictive 'id' column if present
if 'id' in df.columns:
    df = df.drop(columns=['id'])

print(f'Dataset shape: {df.shape}')
print(f'Target distribution:
{df[TARGET].value_counts()}
')

# ---------------------------------------------------------------------------
# 2. Select features and target
# ---------------------------------------------------------------------------
print('Selecting features...')
X = df[SELECTED_FEATURES]
y = df[TARGET]

print(f'Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}')

# ---------------------------------------------------------------------------
# 3. Train/test split (stratified)
# ---------------------------------------------------------------------------
print('Splitting data (80/20 stratified)...')
X_train, X_test, y_train, y_test = dividir_datos(
    df, target=TARGET, test_size=0.2, random_state=RANDOM_STATE
)

# Keep only selected features after split
X_train = X_train[SELECTED_FEATURES]
X_test = X_test[SELECTED_FEATURES]

print(f'Train size: {len(X_train)} | Test size: {len(X_test)}')

# ---------------------------------------------------------------------------
# 4. Preprocessing: standardize and encode
# ---------------------------------------------------------------------------
print('Preprocessing...')
X_train_scaled, X_test_scaled, scaler = estandarizar(X_train, X_test)
y_train_enc, y_test_enc, encoder = codificar(y_train, y_test)

print(f'Classes: {list(encoder.classes_)}  ->  {list(encoder.transform(encoder.classes_))}')

# ---------------------------------------------------------------------------
# 5. Train model
# ---------------------------------------------------------------------------
print('
Training LogisticRegression with best params...')
print(f'Params: {BEST_PARAMS}')

model = LogisticRegression(**BEST_PARAMS, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train_enc)

print('Training complete.')

# ---------------------------------------------------------------------------
# 6. Evaluate on test set
# ---------------------------------------------------------------------------
print('
=============== EVALUATION ON TEST SET ===============')
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of positive class

accuracy = accuracy_score(y_test_enc, y_pred)
precision = precision_score(y_test_enc, y_pred)
recall = recall_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred)
roc_auc = roc_auc_score(y_test_enc, y_proba)
cm = confusion_matrix(y_test_enc, y_pred)

print(f'Accuracy : {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'F1-score : {f1:.4f}')
print(f'ROC-AUC  : {roc_auc:.4f}')
print()
print('Confusion Matrix:')
print(f'              Predicted')
print(f'               Neg   Pos')
print(f'Actual Neg   {cm[0, 0]:5d} {cm[0, 1]:5d}')
print(f'       Pos   {cm[1, 0]:5d} {cm[1, 1]:5d}')
print()
print('Classification Report:')
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

# Feature importance (coefficients)
print('
Feature Coefficients:')
coef_df = pd.DataFrame({
    'Feature': SELECTED_FEATURES,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Save model, scaler, and encoder
# ---------------------------------------------------------------------------
print(f'
Saving model to {MODEL_PATH}...')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f'Saving scaler to {SCALER_PATH}...')
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f'Saving encoder to {ENCODER_PATH}...')
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoder, f)

print('
All artifacts saved. Done!')
