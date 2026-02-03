"""
Main Orchestration Script
Coordinates entire pipeline: data generation, preprocessing, model training, 
hyperparameter tuning, evaluation, and XAI analysis.
"""

import numpy as np
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_generation import TimeSeriesDataGenerator
from preprocessing import TimeSeriesPreprocessor
from model import LSTMModel, TCNModel
from tuning import HyperparameterTuner
from evaluation import ModelEvaluator, BaselineModels
from xai_analysis import ModelExplainability


class TimeSeriesForecastingPipeline:
    """
    Complete time series forecasting pipeline.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with all hyperparameters
        """
        self.config = config or self._get_default_config()
        self.data_generator = TimeSeriesDataGenerator(seed=self.config['seed'])
        self.preprocessor = TimeSeriesPreprocessor(
            scaler_type=self.config['scaler_type']
        )
        self.evaluator = ModelEvaluator()
        self.model = None
        self.results = {}
    
    @staticmethod
    def _get_default_config() -> dict:
        """Get default configuration."""
        return {
            'seed': 42,
            'n_samples': 1095,
            'n_features': 5,
            'scaler_type': 'standard',
            'sequence_length': 30,
            'forecast_horizon': 1,
            'model_type': 'lstm',
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'use_hyperparameter_tuning': False,
            'n_tuning_trials': 50,
            'perform_xai_analysis': True,
            'test_size': 0.2,
            'val_size': 0.1
        }
    
    def generate_data(self):
        """Generate synthetic time series data."""
        print("Generating synthetic time series data...")
        X_train, X_val, X_test, metadata = self.data_generator.generate_synthetic_dataset(
            n_samples=self.config['n_samples'],
            n_features=self.config['n_features'],
            test_size=self.config['test_size'],
            val_size=self.config['val_size']
        )
        
        print(f"  Train shape: {X_train.shape}")
        print(f"  Val shape: {X_val.shape}")
        print(f"  Test shape: {X_test.shape}")
        
        self.results['data_metadata'] = metadata
        return X_train, X_val, X_test
    
    def preprocess_data(self, X_train, X_val, X_test):
        """Preprocess data with feature engineering and scaling."""
        print("\nPreprocessing data...")
        processed_data = self.preprocessor.preprocess_pipeline(
            X_train, X_val, X_test,
            sequence_length=self.config['sequence_length'],
            forecast_horizon=self.config['forecast_horizon'],
            add_lags=True,
            add_ma=True
        )
        
        print(f"  X_train sequences: {processed_data['X_train'].shape}")
        print(f"  y_train sequences: {processed_data['y_train'].shape}")
        print(f"  Total features: {processed_data['n_features']}")
        
        return processed_data
    
    def build_model(self, input_shape, output_shape):
        """Build the deep learning model."""
        print(f"\nBuilding {self.config['model_type'].upper()} model...")
        
        if self.config['model_type'] == 'lstm':
            self.model = LSTMModel(
                input_shape=input_shape,
                output_shape=output_shape,
                lstm_units=self.config['lstm_units'],
                dropout_rate=self.config['dropout_rate'],
                learning_rate=self.config['learning_rate']
            )
        elif self.config['model_type'] == 'tcn':
            self.model = TCNModel(
                input_shape=input_shape,
                output_shape=output_shape,
                filters=self.config['lstm_units'],
                dropout_rate=self.config['dropout_rate'],
                learning_rate=self.config['learning_rate']
            )
        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")
        
        print("  Model built successfully")
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model."""
        print("\nTraining model...")
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=1
        )
        
        self.results['training_history'] = {
            'loss': [float(x) for x in history['loss']],
            'val_loss': [float(x) for x in history['val_loss']]
        }
        
        print(f"  Final training loss: {history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set and compare with baselines."""
        print("\nEvaluating model...")
        
        # Deep learning model predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluation
        metrics = self.evaluator.evaluate(
            y_test.reshape(-1),
            y_pred.reshape(-1),
            model_name=f"{self.config['model_type'].upper()} Model"
        )
        
        # Baseline models
        print("  Evaluating baseline models...")
        
        # Naive baseline
        y_naive = BaselineModels.naive_forecast(
            y_test[:-1].reshape(-1),
            steps=len(y_test)
        )
        self.evaluator.evaluate(
            y_test.reshape(-1),
            y_naive,
            model_name="Naive Baseline"
        )
        
        # Seasonal naive
        y_seasonal = BaselineModels.seasonal_naive(
            y_test[:-1].reshape(-1),
            season_length=7,
            steps=len(y_test)
        )
        self.evaluator.evaluate(
            y_test.reshape(-1),
            y_seasonal,
            model_name="Seasonal Naive"
        )
        
        self.evaluator.print_results()
        self.results['metrics'] = metrics
        
        return y_pred
    
    def perform_xai_analysis(self, X_train, X_test, y_test, y_pred):
        """Perform explainability analysis using SHAP."""
        if not self.config['perform_xai_analysis']:
            print("\nSkipping XAI analysis (disabled in config)")
            return
        
        print("\nPerforming XAI Analysis (SHAP)...")
        
        try:
            # Create explainability module
            explainability = ModelExplainability(self.model, X_train)
            
            # Perform analysis on subset for efficiency
            n_samples = min(50, len(X_test))
            results = explainability.full_analysis(
                X_test[:n_samples],
                y_test[:n_samples],
                feature_names=[f"Feature_{i}" for i in range(X_test.shape[-1])]
            )
            
            if 'feature_importance_shap' in results:
                print("  SHAP Feature Importance:")
                for feature, importance in sorted(
                    results['feature_importance_shap'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]:
                    print(f"    {feature}: {importance:.6f}")
            
            self.results['xai_analysis'] = {
                'shap_feature_importance': results.get('feature_importance_shap', {}),
                'permutation_importance': results.get('feature_importance_permutation', {})
            }
            
        except Exception as e:
            print(f"  Warning: XAI analysis failed: {e}")
    
    def save_results(self, output_path: str = "results.json"):
        """Save all results to JSON."""
        print(f"\nSaving results to {output_path}...")
        
        # Convert numpy types to native Python types
        results_serializable = self._make_serializable(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"  Results saved successfully")
    
    @staticmethod
    def _make_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: TimeSeriesForecastingPipeline._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TimeSeriesForecastingPipeline._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def run_full_pipeline(self):
        """Execute the complete pipeline."""
        print("=" * 60)
        print("Advanced Time Series Forecasting Pipeline")
        print("=" * 60)
        
        # Data generation
        X_train, X_val, X_test = self.generate_data()
        
        # Preprocessing
        processed_data = self.preprocess_data(X_train, X_val, X_test)
        X_train_seq = processed_data['X_train']
        y_train_seq = processed_data['y_train']
        X_val_seq = processed_data['X_val']
        y_val_seq = processed_data['y_val']
        X_test_seq = processed_data['X_test']
        y_test_seq = processed_data['y_test']
        
        # Model building
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        # Simplify output to single feature prediction
        output_shape = (1, 1)
        self.build_model(input_shape, output_shape)
        
        # Reshape targets to match output shape (flatten to single values)
        y_train_seq = y_train_seq[:, 0, 0].reshape(-1, 1, 1)
        y_val_seq = y_val_seq[:, 0, 0].reshape(-1, 1, 1)
        y_test_seq = y_test_seq[:, 0, 0].reshape(-1, 1, 1)
        
        # Model training
        self.train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        
        # Evaluation
        y_pred = self.evaluate_model(X_test_seq, y_test_seq)
        
        # XAI Analysis
        self.perform_xai_analysis(X_train_seq, X_test_seq, y_test_seq, y_pred)
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'tcn'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-xai', action='store_true', help='Disable XAI analysis')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = TimeSeriesForecastingPipeline._get_default_config()
    
    # Override with command-line arguments
    config['model_type'] = args.model_type
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['perform_xai_analysis'] = not args.no_xai
    
    # Run pipeline
    pipeline = TimeSeriesForecastingPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
