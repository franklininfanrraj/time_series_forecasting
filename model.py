"""
Deep Learning Models for Time Series Forecasting
Implements LSTM and TCN architectures for multivariate time series prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.regularizers import l2
from typing import Tuple, Optional
import numpy as np


class LSTMModel:
    """
    LSTM-based model for time series forecasting.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        regularization: float = 1e-5,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_shape: (sequence_length, n_features)
            output_shape: (forecast_horizon, n_features)
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            regularization: L2 regularization strength
            learning_rate: Optimizer learning rate
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self) -> Model:
        """Build the LSTM model architecture."""
        model = Sequential([
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=self.input_shape,
                kernel_regularizer=l2(self.regularization)
            ),
            layers.Dropout(self.dropout_rate),
            
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=True,
                kernel_regularizer=l2(self.regularization)
            ),
            layers.Dropout(self.dropout_rate),
            
            layers.LSTM(
                self.lstm_units // 4,
                kernel_regularizer=l2(self.regularization)
            ),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=l2(self.regularization)
            ),
            layers.Dropout(self.dropout_rate),
            
            layers.Dense(int(np.prod(self.output_shape)))
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> dict:
        """
        Train the model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Reshape targets
        y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
        y_val_reshaped = y_val.reshape(y_val.shape[0], -1)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train_reshaped,
            validation_data=(X_val, y_val_reshaped),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions reshaped to output shape
        """
        predictions = self.model.predict(X, verbose=0)
        return predictions.reshape(-1, *self.output_shape)
    
    def get_summary(self):
        """Print model summary."""
        return self.model.summary()


class TCNModel:
    """
    Temporal Convolutional Network for time series forecasting.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        filters: int = 64,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        regularization: float = 1e-5,
        learning_rate: float = 0.001
    ):
        """
        Initialize TCN model.
        
        Args:
            input_shape: (sequence_length, n_features)
            output_shape: (forecast_horizon, n_features)
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernels
            dropout_rate: Dropout rate
            regularization: L2 regularization
            learning_rate: Optimizer learning rate
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self) -> Model:
        """Build the TCN model architecture."""
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        
        # TCN layers with dilated convolutions
        dilations = [1, 2, 4, 8]
        for dilation in dilations:
            x = layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation,
                padding='same',
                activation='relu',
                kernel_regularizer=l2(self.regularization)
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=l2(self.regularization)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(np.prod(self.output_shape))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> dict:
        """Train the TCN model."""
        y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
        y_val_reshaped = y_val.reshape(y_val.shape[0], -1)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train_reshaped,
            validation_data=(X_val, y_val_reshaped),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.model.predict(X, verbose=0)
        return predictions.reshape(-1, *self.output_shape)
    
    def get_summary(self):
        """Print model summary."""
        return self.model.summary()


class EnsembleModel:
    """
    Ensemble of multiple forecasting models for improved predictions.
    """
    
    def __init__(self):
        """Initialize ensemble."""
        self.models = []
        self.weights = None
    
    def add_model(self, model):
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def predict(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ensemble predictions using weighted average.
        
        Args:
            X: Input data
            weights: Model weights (default: equal)
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = np.array([model.predict(X) for model in self.models])
        
        if weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        
        return np.average(predictions, axis=0, weights=weights)


if __name__ == "__main__":
    # Example usage
    input_shape = (30, 5)  # 30 timesteps, 5 features
    output_shape = (1, 5)  # Forecast 1 step ahead
    
    # Create LSTM model
    lstm_model = LSTMModel(input_shape, output_shape, lstm_units=64)
    lstm_model.get_summary()
    
    # Create TCN model
    tcn_model = TCNModel(input_shape, output_shape, filters=64)
    tcn_model.get_summary()
