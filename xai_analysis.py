"""
XAI Analysis Module
Implements SHAP (SHapley Additive exPlanations) for time series interpretability.
"""

import numpy as np
import shap
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesSHAPAnalyzer:
    """
    SHAP-based explainability for time series models.
    """
    
    def __init__(self, model, background_data: np.ndarray, n_samples: int = 100):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained model with predict method
            background_data: Background data for SHAP (subset of training data)
            n_samples: Number of samples for SHAP calculation
        """
        self.model = model
        self.background_data = background_data[:n_samples]
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def create_explainer(self, explainer_type: str = 'kernel') -> None:
        """
        Create SHAP explainer.
        
        Args:
            explainer_type: Type of explainer ('kernel', 'tree', 'deep', etc.)
        """
        if explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                self.background_data
            )
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(
                self.model.model,
                self.background_data
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    def explain_predictions(
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None
    ) -> Dict:
        """
        Explain model predictions using SHAP.
        
        Args:
            X: Input data to explain
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[-1])]
        
        return {
            'shap_values': self.shap_values,
            'base_values': self.explainer.expected_value,
            'feature_names': self.feature_names
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from SHAP values.
        
        Returns:
            Dictionary with feature importances (mean absolute SHAP values)
        """
        if self.shap_values is None:
            raise ValueError("Must explain predictions first")
        
        # Calculate mean absolute SHAP values across all samples
        mean_abs_shap = np.abs(self.shap_values).mean(axis=(0, 1))
        
        importance = {
            name: float(value)
            for name, value in zip(self.feature_names, mean_abs_shap)
        }
        
        return importance
    
    def plot_summary(self, max_display: int = 10) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("Must explain predictions first")
        
        try:
            # Reshape SHAP values for visualization
            shap_values_reshaped = self.shap_values.reshape(
                self.shap_values.shape[0], -1
            )
            
            # Create feature names for time steps
            n_samples, n_features = self.shap_values.shape[0], self.shap_values.shape[-1]
            feature_names_extended = [
                f"{self.feature_names[j]}_t-{i}"
                for i in range(self.shap_values.shape[1])
                for j in range(n_features)
            ]
            
            shap.summary_plot(
                shap_values_reshaped,
                self.shap_values.reshape(n_samples, -1),
                feature_names=feature_names_extended[:max_display],
                show=False,
                plot_type='bar'
            )
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not create summary plot: {e}")
    
    def plot_dependence(self, feature_idx: int = 0) -> None:
        """
        Plot SHAP dependence plot for a feature.
        
        Args:
            feature_idx: Index of feature to plot
        """
        if self.shap_values is None:
            raise ValueError("Must explain predictions first")
        
        try:
            shap.dependence_plot(
                feature_idx,
                self.shap_values[:, :, feature_idx],
                self.background_data,
                feature_names=self.feature_names,
                show=False
            )
            plt.show()
        except Exception as e:
            print(f"Could not create dependence plot: {e}")
    
    def get_local_explanations(self, sample_idx: int = 0) -> Dict:
        """
        Get local explanation for a specific sample.
        
        Args:
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with local SHAP values and feature contributions
        """
        if self.shap_values is None:
            raise ValueError("Must explain predictions first")
        
        local_shap = self.shap_values[sample_idx]
        
        # Calculate contribution of each feature
        contributions = {}
        for t in range(local_shap.shape[0]):
            for f, fname in enumerate(self.feature_names):
                key = f"{fname}_t-{t}"
                contributions[key] = float(local_shap[t, f])
        
        # Sort by absolute contribution
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'sample_idx': sample_idx,
            'contributions': contributions,
            'top_contributors': sorted_contrib[:10],
            'base_value': float(self.explainer.expected_value)
        }


class FeatureImportanceAnalyzer:
    """
    Traditional feature importance analysis for comparison.
    """
    
    @staticmethod
    def permutation_importance(
        model,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """
        Calculate permutation importance.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            n_repeats: Number of repeats for each feature
            
        Returns:
            Dictionary with feature importances
        """
        # Get baseline score
        baseline_pred = model.predict(X)
        baseline_loss = np.mean((baseline_pred - y) ** 2)
        
        importances = {}
        n_features = X.shape[-1]
        
        for feature_idx in range(n_features):
            losses = []
            for _ in range(n_repeats):
                # Permute feature
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[..., feature_idx])
                
                # Calculate loss
                pred = model.predict(X_permuted)
                loss = np.mean((pred - y) ** 2)
                losses.append(loss - baseline_loss)
            
            importances[f"Feature_{feature_idx}"] = np.mean(losses)
        
        return importances
    
    @staticmethod
    def plot_feature_importance(importances: Dict[str, float]) -> None:
        """
        Plot feature importance.
        
        Args:
            importances: Dictionary of feature importances
        """
        features = list(importances.keys())
        values = list(importances.values())
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, values)
        plt.xlabel('Importance (Loss Increase)')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()


class ModelExplainability:
    """
    Comprehensive explainability analysis combining multiple techniques.
    """
    
    def __init__(self, model, background_data: np.ndarray):
        """
        Initialize explainability module.
        
        Args:
            model: Trained model
            background_data: Background data for SHAP
        """
        self.model = model
        self.shap_analyzer = TimeSeriesSHAPAnalyzer(model, background_data)
        self.feature_analyzer = FeatureImportanceAnalyzer()
    
    def full_analysis(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None
    ) -> Dict:
        """
        Perform comprehensive explainability analysis.
        
        Args:
            X_test: Test data
            y_test: Test targets
            feature_names: Feature names
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # SHAP analysis
        try:
            shap_results = self.shap_analyzer.explain_predictions(X_test, feature_names)
            results['shap_values'] = shap_results['shap_values']
            results['feature_importance_shap'] = self.shap_analyzer.get_feature_importance()
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
        
        # Permutation importance
        try:
            results['feature_importance_permutation'] = (
                self.feature_analyzer.permutation_importance(self.model, X_test, y_test)
            )
        except Exception as e:
            print(f"Permutation importance failed: {e}")
        
        return results


if __name__ == "__main__":
    print("XAI Analysis module loaded")
import numpy as np

def explain(model, X_sample):
    explainer = shap.DeepExplainer(model, X_sample[:50])
    shap_values = explainer.shap_values(X_sample[:10])
    return np.mean(np.abs(shap_values), axis=(0,1))
