"""
Preprocessing Module
Handles data scaling, differencing, feature engineering, and windowing.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional
import pandas as pd


class TimeSeriesPreprocessor:
    """
    Preprocessing pipeline for time series data.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        self.scaler_type = scaler_type
        self.is_fitted = False
        self.original_mean = None
        self.original_std = None
    
    def fit_scaler(self, data: np.ndarray) -> 'TimeSeriesPreprocessor':
        """
        Fit the scaler on training data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        self.scaler.fit(data)
        self.is_fitted = True
        self.original_mean = np.mean(data, axis=0)
        self.original_std = np.std(data, axis=0)
        return self
    
    def scale(self, data: np.ndarray) -> np.ndarray:
        """Scale data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first using fit_scaler()")
        return self.scaler.transform(data)
    
    def unscale(self, data: np.ndarray) -> np.ndarray:
        """Unscale data back to original scale."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first using fit_scaler()")
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def difference(data: np.ndarray, lag: int = 1) -> np.ndarray:
        """
        Apply differencing to remove trend.
        
        Args:
            data: Time series data
            lag: Differencing lag
            
        Returns:
            Differenced data
        """
        return np.diff(data, n=1, axis=0)
    
    @staticmethod
    def add_lagged_features(
        data: np.ndarray,
        lags: list = [1, 7, 14]
    ) -> np.ndarray:
        """
        Add lagged variables as features.
        
        Args:
            data: Time series data of shape (n_samples, n_features)
            lags: List of lag values to create
            
        Returns:
            Data with lagged features appended
        """
        n_samples, n_features = data.shape
        max_lag = max(lags)
        
        # Initialize output array (we lose max_lag samples due to lagging)
        lagged_data = data[max_lag:].copy()
        
        for lag in lags:
            lagged = data[max_lag-lag:-lag or None]
            lagged_data = np.hstack([lagged_data, lagged])
        
        return lagged_data
    
    @staticmethod
    def add_moving_average(
        data: np.ndarray,
        windows: list = [7, 14, 30]
    ) -> np.ndarray:
        """
        Add moving average features.
        
        Args:
            data: Time series data
            windows: List of window sizes for moving averages
            
        Returns:
            Data with moving average features
        """
        n_samples, n_features = data.shape
        ma_features = np.zeros((n_samples, n_features * len(windows)))
        
        for i, window in enumerate(windows):
            for j in range(n_features):
                ma = np.convolve(data[:, j], np.ones(window)/window, mode='same')
                ma_features[:, i*n_features + j] = ma
        
        return np.hstack([data, ma_features])
    
    @staticmethod
    def create_sequences(
        data: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for supervised learning.
        
        Args:
            data: Time series data of shape (n_samples, n_features)
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Target values (n_sequences, forecast_horizon, n_features)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def preprocess_pipeline(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        sequence_length: int = 30,
        forecast_horizon: int = 1,
        add_lags: bool = True,
        add_ma: bool = True
    ) -> Dict:
        """
        Full preprocessing pipeline.
        
        Args:
            X_train, X_val, X_test: Train/val/test data
            sequence_length: Length of sequences for model
            forecast_horizon: Forecast horizon
            add_lags: Whether to add lagged features
            add_ma: Whether to add moving average features
            
        Returns:
            Dictionary with processed data
        """
        # Feature engineering
        if add_lags:
            X_train = self.add_lagged_features(X_train)
            X_val = self.add_lagged_features(X_val)
            X_test = self.add_lagged_features(X_test)
        
        if add_ma:
            X_train = self.add_moving_average(X_train)
            X_val = self.add_moving_average(X_val)
            X_test = self.add_moving_average(X_test)
        
        # Scaling
        self.fit_scaler(X_train)
        X_train_scaled = self.scale(X_train)
        X_val_scaled = self.scale(X_val)
        X_test_scaled = self.scale(X_test)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, sequence_length, forecast_horizon
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val_scaled, sequence_length, forecast_horizon
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test_scaled, sequence_length, forecast_horizon
        )
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_val': X_val_seq,
            'y_val': y_val_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'scaler': self.scaler,
            'n_features': X_train.shape[1],
            'sequence_length': sequence_length
        }


if __name__ == "__main__":
    from data_generation import TimeSeriesDataGenerator
    
    # Generate sample data
    generator = TimeSeriesDataGenerator()
    X_train, X_val, X_test, _ = generator.generate_synthetic_dataset(n_features=5)
    
    # Preprocess
    preprocessor = TimeSeriesPreprocessor()
    processed = preprocessor.preprocess_pipeline(
        X_train, X_val, X_test,
        sequence_length=30,
        forecast_horizon=1
    )
    
    print(f"X_train shape: {processed['X_train'].shape}")
    print(f"y_train shape: {processed['y_train'].shape}")
    print(f"Total features after engineering: {processed['n_features']}")
