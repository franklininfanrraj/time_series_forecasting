#!/usr/bin/env python
"""
Fast version of main.py - reduced epochs and trials for quick results
"""
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# ===== DATA GENERATION =====
np.random.seed(42)
days = 3 * 365
t = np.arange(days)
trend = 0.01 * t
weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)
f1 = trend + weekly_seasonality + np.random.normal(0, 2, days)
f2 = 0.8 * np.roll(f1, 1) + np.random.normal(0, 1, days)
f3 = np.random.normal(5, 1.5, days)
f4 = 0.5 * f1 + np.random.normal(0, 1, days)
f5 = np.random.normal(0, 3, days)
target = 0.4*f1 + 0.3*f2 + 0.2*f4 + np.random.normal(0, 2, days)

df = pd.DataFrame({
    "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "target": target
})

results = []
results.append("=" * 70)
results.append("ADVANCED TIME SERIES FORECASTING WITH LSTM + XAI")
results.append("=" * 70)
results.append("")
results.append("[1] DATA GENERATION")
results.append(f"    Dataset Shape: {df.shape}")
results.append("")
results.append("    First 5 rows:")
for line in str(df.head()).split('\n'):
    results.append(f"    {line}")

# ===== PREPROCESSING =====
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :-1])
        y.append(data[i+window, -1])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, 30)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

results.append("")
results.append("[2] PREPROCESSING")
results.append(f"    Train Set: X{X_train.shape}, y{y_train.shape}")
results.append(f"    Test Set:  X{X_test.shape}, y{y_test.shape}")

# ===== LSTM MODEL =====
def build_lstm(input_shape, units, dropout):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ===== HYPERPARAMETER TUNING =====
results.append("")
results.append("[3] HYPERPARAMETER TUNING")
results.append("    Running Optuna optimization (2 trials)...")

def objective(trial):
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    model = build_lstm(X_train.shape[1:], units, dropout)
    history = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
    return min(history.history["loss"])

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2)
best_params = study.best_params

results.append(f"    Best Units: {best_params['units']}")
results.append(f"    Best Dropout: {best_params['dropout']:.4f}")

# ===== TRAIN FINAL MODEL =====
results.append("")
results.append("[4] TRAINING FINAL LSTM MODEL")
final_model = build_lstm(X_train.shape[1:], best_params["units"], best_params["dropout"])
history = final_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
results.append("    Training complete!")

# ===== EVALUATION =====
results.append("")
results.append("[5] EVALUATION - LSTM MODEL")
y_pred = final_model.predict(X_test, verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

results.append(f"    RMSE: {rmse:.4f}")
results.append(f"    MAE:  {mae:.4f}")
results.append(f"    MAPE: {mape:.2f}%")

# ===== BASELINE =====
results.append("")
results.append("[6] BASELINE - SARIMAX MODEL")
try:
    sarimax = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,7))
    sarimax_result = sarimax.fit(disp=False)
    sarimax_forecast = sarimax_result.forecast(len(y_test))
    sarimax_rmse = np.sqrt(mean_squared_error(y_test, sarimax_forecast))
    results.append(f"    SARIMAX RMSE: {sarimax_rmse:.4f}")
except Exception as e:
    results.append(f"    SARIMAX Error: {str(e)}")
    sarimax_rmse = None

# ===== COMPARISON =====
results.append("")
results.append("[7] MODEL COMPARISON")
results.append(f"    LSTM RMSE:    {rmse:.4f}")
if sarimax_rmse:
    improvement = ((sarimax_rmse - rmse) / sarimax_rmse) * 100
    results.append(f"    SARIMAX RMSE: {sarimax_rmse:.4f}")
    results.append(f"    LSTM Improvement: {improvement:.2f}%")

# ===== FEATURE IMPORTANCE =====
results.append("")
results.append("[8] FEATURE IMPORTANCE (Gradient-based)")
feature_names = ["f1", "f2", "f3", "f4", "f5"]
sample = X_test[0:1]
with tf.GradientTape() as tape:
    sample_float = tf.cast(sample, tf.float32)
    tape.watch(sample_float)
    pred = final_model(sample_float)
grads = tape.gradient(pred, sample_float)
feature_importance = np.abs(grads.numpy()).mean(axis=(0, 1))
importance_df = pd.DataFrame({
    "Feature": feature_names, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

results.append("")
results.append("    Feature Importance Ranking:")
for idx, row in importance_df.iterrows():
    results.append(f"    {row['Feature']}: {row['Importance']:.6f}")

results.append("")
results.append("=" * 70)
results.append("✓✓✓ PROGRAM COMPLETED SUCCESSFULLY ✓✓✓")
results.append("=" * 70)

# Save results
output = "\n".join(results)
print(output)

with open("program_results.txt", "w") as f:
    f.write(output)

print("\nResults saved to: program_results.txt")
