"""
Hyperparameter Tuning Module
Uses Optuna for automated hyperparameter optimization.
"""

import optuna
from optuna.trial import Trial
import numpy as np
from typing import Callable, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Automated hyperparameter tuning using Optuna.
    """
    
    def __init__(self, n_trials: int = 50, n_jobs: int = 1, seed: int = 42):
        """
        Initialize the tuner.
        
        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            seed: Random seed
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.study = None
        self.best_params = None
    
    def create_objective(
        self,
        train_func: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'lstm'
    ) -> Callable:
        """
        Create an objective function for Optuna.
        
        Args:
            train_func: Function to train and evaluate model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type: Type of model ('lstm' or 'tcn')
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial: Trial) -> float:
            # Hyperparameter suggestions
            if model_type == 'lstm':
                params = {
                    'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=32),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_int('batch_size', 16, 64, step=16),
                    'epochs': 100  # Early stopping will limit this
                }
            else:  # TCN
                params = {
                    'filters': trial.suggest_int('filters', 32, 128, step=32),
                    'kernel_size': trial.suggest_int('kernel_size', 2, 5),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_int('batch_size', 16, 64, step=16),
                    'epochs': 100
                }
            
            try:
                # Train model and return validation loss
                val_loss = train_func(params, X_train, y_train, X_val, y_val)
                return val_loss
            except Exception as e:
                print(f"Trial failed with error: {e}")
                return float('inf')
        
        return objective
    
    def optimize(
        self,
        objective: Callable,
        direction: str = 'minimize',
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective: Objective function for optimization
            direction: 'minimize' or 'maximize'
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with best parameters and value
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress
        )
        
        self.best_params = self.study.best_params
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def get_trials_dataframe(self):
        """
        Get all trials as a dataframe for analysis.
        
        Returns:
            Pandas DataFrame with trial information
        """
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        
        return self.study.trials_dataframe()
    
    def plot_optimization_history(self):
        """
        Plot optimization history.
        Requires matplotlib or plotly.
        """
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        
        try:
            import matplotlib.pyplot as plt
            trials_df = self.get_trials_dataframe()
            
            plt.figure(figsize=(10, 6))
            plt.plot(trials_df.index, trials_df['value'], 'b-o')
            plt.xlabel('Trial Number')
            plt.ylabel('Objective Value')
            plt.title('Optimization History')
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def get_param_importance(self):
        """
        Get parameter importance analysis.
        
        Returns:
            Dictionary with parameter importances
        """
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        
        try:
            importances = optuna.importance.get_param_importances(self.study)
            return importances
        except Exception as e:
            print(f"Could not calculate importance: {e}")
            return None


class GridSearchTuner:
    """
    Simple grid search for hyperparameter tuning.
    """
    
    @staticmethod
    def grid_search(
        param_grid: Dict[str, list],
        train_func: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid: Dictionary of parameters and values to search
            train_func: Training function
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary with best parameters and results
        """
        import itertools
        
        best_loss = float('inf')
        best_params = None
        results = []
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            try:
                val_loss = train_func(params, X_train, y_train, X_val, y_val)
                results.append({'params': params, 'loss': val_loss})
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    
            except Exception as e:
                print(f"Failed with params {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_value': best_loss,
            'all_results': results
        }


if __name__ == "__main__":
    print("Hyperparameter tuning module loaded")

def objective(trial, X_train, y_train):
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = build_model(X_train.shape[1:], units, dropout)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    return min(history.history["loss"])

def tune(X_train, y_train):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=20)
    return study.best_params
