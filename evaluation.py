"""
Evaluation Module
Comprehensive metrics and benchmarking for time series forecasting.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesMetrics:
    """
    Collection of time series forecasting evaluation metrics.
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        return float(mean_absolute_percentage_error(y_true, y_pred))
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return float(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional accuracy - percentage of time direction is predicted correctly.
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return float(np.mean(true_direction == pred_direction))
    
    @staticmethod
    def dtw(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Dynamic Time Warping distance.
        Measures similarity between two temporal sequences.
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        n, m = len(y_true), len(y_pred)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(y_true[i - 1] - y_pred[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
        
        return float(dtw_matrix[n, m])
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (coefficient of determination)"""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    @staticmethod
    def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Theil's U statistic - for comparing forecast accuracy.
        Values < 1 indicate better than naive forecast.
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        if len(y_true) < 2:
            return 0.0
        
        numerator = np.sum((y_true[1:] - y_pred[1:]) ** 2)
        denominator = np.sum((y_true[1:] - y_true[:-1]) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return float(np.sqrt(numerator / denominator))


class ModelEvaluator:
    """
    Comprehensive model evaluation on multiple metrics.
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_calculator = TimeSeriesMetrics()
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate model on all metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of model for tracking
            
        Returns:
            Dictionary with all metrics
        """
        # Flatten if needed
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        metrics = {
            'rmse': self.metrics_calculator.rmse(y_true_flat, y_pred_flat),
            'mae': self.metrics_calculator.mae(y_true_flat, y_pred_flat),
            'mape': self.metrics_calculator.mape(y_true_flat, y_pred_flat),
            'mse': self.metrics_calculator.mse(y_true_flat, y_pred_flat),
            'r_squared': self.metrics_calculator.r_squared(y_true_flat, y_pred_flat),
            'directional_accuracy': self.metrics_calculator.directional_accuracy(y_true, y_pred),
            'theil_u': self.metrics_calculator.theil_u(y_true_flat, y_pred_flat),
            'dtw': self.metrics_calculator.dtw(y_true_flat, y_pred_flat)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self) -> Dict:
        """
        Compare all evaluated models.
        
        Returns:
            Summary of model comparison
        """
        if not self.results:
            return {}
        
        comparison = {}
        for metric in list(self.results.values())[0].keys():
            comparison[metric] = {}
            for model_name, metrics in self.results.items():
                comparison[metric][model_name] = metrics[metric]
        
        return comparison
    
    def print_results(self, model_name: str = None):
        """Print evaluation results"""
        if model_name and model_name in self.results:
            metrics = self.results[model_name]
            print(f"\n{model_name} Results:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"{metric:20s}: {value:.6f}")
        else:
            for model_name, metrics in self.results.items():
                print(f"\n{model_name} Results:")
                print("-" * 40)
                for metric, value in metrics.items():
                    print(f"{metric:20s}: {value:.6f}")


class BaselineModels:
    """
    Baseline models for comparison (Naive, Seasonal Naive, Exponential Smoothing).
    """
    
    @staticmethod
    def naive_forecast(y_train: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Naive forecast - use last value repeated.
        
        Args:
            y_train: Training data
            steps: Number of steps to forecast
            
        Returns:
            Forecast array
        """
        last_value = y_train[-1]
        return np.full(steps, last_value)
    
    @staticmethod
    def seasonal_naive(y_train: np.ndarray, season_length: int = 7, steps: int = 1) -> np.ndarray:
        """
        Seasonal naive forecast - use value from same season.
        
        Args:
            y_train: Training data
            season_length: Length of seasonal period
            steps: Number of steps to forecast
            
        Returns:
            Forecast array
        """
        forecasts = []
        for i in range(steps):
            idx = (len(y_train) - season_length + i) % len(y_train)
            forecasts.append(y_train[idx])
        return np.array(forecasts)
    
    @staticmethod
    def exponential_smoothing(
        y_train: np.ndarray,
        alpha: float = 0.3,
        steps: int = 1
    ) -> np.ndarray:
        """
        Exponential smoothing forecast.
        
        Args:
            y_train: Training data
            alpha: Smoothing parameter
            steps: Number of steps to forecast
            
        Returns:
            Forecast array
        """
        # Fit exponential smoothing
        s = np.zeros(len(y_train) + steps)
        s[0] = y_train[0]
        
        for t in range(1, len(y_train)):
            s[t] = alpha * y_train[t - 1] + (1 - alpha) * s[t - 1]
        
        # Forecast
        for t in range(len(y_train), len(y_train) + steps):
            s[t] = s[t - 1]
        
        return s[-steps:]


if __name__ == "__main__":
    print("Evaluation module loaded")
import numpy as np

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape
