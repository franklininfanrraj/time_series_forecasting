# =========================================================
# Advanced Time Series Forecasting with LSTM + XAI (SHAP)
# Single-file Jupyter Notebook Implementation
# =========================================================

# ---------------------------
# 1. Imports
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import optuna
import shap

from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# 2. Data Generation
# ---------------------------
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
    "f1": f1,
    "f2": f2,
    "f3": f3,
    "f4": f4,
    "f5": f5,
    "target": target
})

print("Dataset Shape:", df.shape)
df.head()

# ---------------------------
# 3. Preprocessing & Sliding Window
# ---------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :-1])
        y.append(data[i+window, -1])
    return np.array(X), np.array(y)

WINDOW = 30
X, y = create_sequences(scaled_data, WINDOW)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train Shape:", X_train.shape, y_train.shape)
print("Test Shape:", X_test.shape, y_test.shape)

# ---------------------------
# 4. LSTM Model Builder
# ---------------------------
def build_lstm(input_shape, units, dropout):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ---------------------------
# 5. Hyperparameter Tuning (Optuna)
# ---------------------------
def objective(trial):
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = build_lstm(X_train.shape[1:], units, dropout)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )
    return min(history.history["loss"])

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best Parameters:", best_params)

# ---------------------------
# 6. Train Final LSTM Model
# ---------------------------
final_model = build_lstm(
    X_train.shape[1:], 
    best_params["units"], 
    best_params["dropout"]
)

history = final_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
# 7. Evaluation
# ---------------------------
y_pred = final_model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"LSTM RMSE: {rmse:.3f}")
print(f"LSTM MAE : {mae:.3f}")
print(f"LSTM MAPE: {mape:.2f}%")

# ---------------------------
# 8. Baseline SARIMAX Model
# ---------------------------
sarimax = SARIMAX(
    y_train,
    order=(1,1,1),
    seasonal_order=(1,1,1,7)
)
sarimax_result = sarimax.fit(disp=False)

sarimax_forecast = sarimax_result.forecast(len(y_test))

sarimax_rmse = np.sqrt(mean_squared_error(y_test, sarimax_forecast))
print(f"SARIMAX RMSE: {sarimax_rmse:.3f}")

# ---------------------------
# 9. Explainable AI (SHAP)
# ---------------------------
background = X_train[:50]
explainer = shap.DeepExplainer(final_model, background)

shap_values = explainer.shap_values(X_test[:10])

feature_importance = np.mean(np.abs(shap_values), axis=(0,1))
feature_names = ["f1", "f2", "f3", "f4", "f5"]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (SHAP):")
importance_df

# ---------------------------
# 10. Visualization
# ---------------------------
plt.figure(figsize=(12,4))
plt.plot(y_test[:200], label="Actual")
plt.plot(y_pred[:200], label="LSTM Prediction")
plt.legend()
plt.title("LSTM Forecast vs Actual")
plt.show()

plt.figure(figsize=(6,4))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.title("SHAP Feature Importance")
plt.show()
