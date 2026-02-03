"""
Data Generation Module
Generates synthetic multivariate time series data with seasonality, trend, and noise.
Simulates complex financial or sensor data scenarios.
"""

import numpy as np
from typing import Tuple, Dict
from scipy import signal
import pandas as pd


class TimeSeriesDataGenerator:
    """
    Generates synthetic multivariate time series data with realistic patterns.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_data(
        self,
        n_samples: int = 1095,
        n_features: int = 5,
        seasonality_period: int = 7,
        trend_strength: float = 0.1,
        noise_level: float = 0.05
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate synthetic multivariate time series data.
        
        Args:
            n_samples: Number of time steps (3 years of daily data = 1095)
            n_features: Number of features/variables
            seasonality_period: Period of seasonality (7 for weekly)
            trend_strength: Strength of trend component
            noise_level: Level of Gaussian noise
            
        Returns:
            data: Array of shape (n_samples, n_features)
            metadata: Dictionary containing generation parameters
        """
        data = np.zeros((n_samples, n_features))
        
        # Time index
        t = np.arange(n_samples)
        
        for i in range(n_features):
            # Trend component
            trend = trend_strength * t / n_samples * 100
            
            # Seasonality (multiple frequencies for realism)
            seasonality = (
                10 * np.sin(2 * np.pi * t / seasonality_period) +
                5 * np.sin(2 * np.pi * t / (seasonality_period * 4)) +
                3 * np.cos(2 * np.pi * t / (seasonality_period * 2))
            )
            
            # AR(1) component for autocorrelation
            ar_component = np.zeros(n_samples)
            ar_component[0] = np.random.normal(0, 5)
            for j in range(1, n_samples):
                ar_component[j] = 0.7 * ar_component[j-1] + np.random.normal(0, 2)
            
            # Noise
            noise = np.random.normal(0, noise_level * 10, n_samples)
            
            # Combine components with feature-specific scaling
            scale = 50 + i * 10
            data[:, i] = scale + trend + seasonality + ar_component + noise
        
        metadata = {
            'n_samples': n_samples,
            'n_features': n_features,
            'seasonality_period': seasonality_period,
            'trend_strength': trend_strength,
            'noise_level': noise_level,
            'generation_seed': self.seed
        }
        
        return data, metadata
    
    def generate_synthetic_dataset(
        self,
        n_samples: int = 1095,
        n_features: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate and split data into train, validation, and test sets.
        
        Args:
            n_samples: Total number of samples
            n_features: Number of features
            test_size: Fraction for test set (temporal split)
            val_size: Fraction for validation set
            
        Returns:
            X_train, X_val, X_test: Feature arrays
            metadata: Generation metadata
        """
        data, metadata = self.generate_data(n_samples, n_features)
        
        # Temporal split (important for time series!)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        X_train = data[:val_idx]
        X_val = data[val_idx:test_idx]
        X_test = data[test_idx:]
        
        metadata.update({
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        })
        
        return X_train, X_val, X_test, metadata


if __name__ == "__main__":
    # Example usage
    generator = TimeSeriesDataGenerator(seed=42)
    X_train, X_val, X_test, metadata = generator.generate_synthetic_dataset(
        n_samples=1095,
        n_features=5
    )
    
    print(f"Generated dataset with shape: {X_train.shape}")
    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
    print(f"Metadata: {metadata}")
