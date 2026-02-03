# Advanced Time Series Forecasting with Deep Learning and Explainable AI (XAI)

## Overview

This project implements a comprehensive pipeline for **multivariate time series forecasting** using deep learning models with integrated **explainability analysis**. The system combines state-of-the-art forecasting techniques (LSTM, TCN) with interpretability methods (SHAP) to create production-ready, interpretable forecasting solutions.

### Key Features

✨ **Advanced Data Generation**
- Synthetic multivariate time series with realistic patterns
- Configurable seasonality, trends, and noise
- Support for 3+ years of daily data

✨ **Sophisticated Preprocessing**
- Robust data scaling (StandardScaler, MinMaxScaler)
- Feature engineering (lagged variables, moving averages)
- Temporal sequence creation for supervised learning

✨ **Deep Learning Models**
- **LSTM (Long Short-Term Memory)** - Excellent for capturing long-term dependencies
- **TCN (Temporal Convolutional Network)** - Parallelizable with large receptive fields
- Multi-layer architectures with dropout regularization

✨ **Hyperparameter Optimization**
- Automated tuning using Optuna framework
- TPE (Tree-structured Parzen Estimator) sampling
- Grid search alternative for smaller search spaces

✨ **Explainable AI (XAI)**
- SHAP (SHapley Additive exPlanations) for model interpretability
- Feature importance analysis
- Local and global explanations
- Permutation-based importance metrics

✨ **Comprehensive Evaluation**
- 8+ time series metrics (RMSE, MAE, MAPE, R², DTW, Theil's U, etc.)
- Baseline comparison (Naive, Seasonal Naive, Exponential Smoothing)
- Directional accuracy and error analysis

✨ **Production-Ready Code**
- Modular architecture with clear separation of concerns
- Extensive documentation and type hints
- Configuration-based execution
- JSON result serialization

---

## Project Structure

```
.
├── data_generation.py      # Synthetic data generation
├── preprocessing.py        # Data preprocessing and feature engineering
├── model.py               # LSTM and TCN implementations
├── tuning.py              # Hyperparameter optimization with Optuna
├── evaluation.py          # Metrics and baseline models
├── xai_analysis.py        # SHAP-based explainability
├── main.py               # Main orchestration pipeline
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── results.json         # Output results (generated)
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
source venv/bin/activate      # On Linux/Mac
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import tensorflow; import shap; print('All packages installed!')"
```

---

## Usage

### Quick Start

Run the complete pipeline with default configuration:

```bash
python main.py
```

### Customized Execution

#### Command-line Arguments

```bash
# Use TCN model instead of LSTM
python main.py --model-type tcn

# Increase training epochs
python main.py --epochs 150

# Change batch size
python main.py --batch-size 64

# Skip XAI analysis
python main.py --no-xai

# Load custom configuration
python main.py --config config.json
```

#### Configuration File

Create `config.json`:

```json
{
  "seed": 42,
  "n_samples": 1095,
  "n_features": 5,
  "scaler_type": "standard",
  "sequence_length": 30,
  "forecast_horizon": 1,
  "model_type": "lstm",
  "lstm_units": 128,
  "dropout_rate": 0.3,
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 100,
  "use_hyperparameter_tuning": false,
  "n_tuning_trials": 50,
  "perform_xai_analysis": true,
  "test_size": 0.2,
  "val_size": 0.1
}
```

Then run:
```bash
python main.py --config config.json
```

---

## Module Details

### 1. Data Generation (`data_generation.py`)

Generates synthetic time series data with realistic patterns:

```python
from data_generation import TimeSeriesDataGenerator

generator = TimeSeriesDataGenerator(seed=42)
X_train, X_val, X_test, metadata = generator.generate_synthetic_dataset(
    n_samples=1095,
    n_features=5,
    test_size=0.2,
    val_size=0.1
)
```

**Features:**
- Trend component
- Multiple seasonality frequencies
- Autoregressive (AR) component
- Gaussian noise

### 2. Preprocessing (`preprocessing.py`)

Comprehensive data preprocessing pipeline:

```python
from preprocessing import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
processed_data = preprocessor.preprocess_pipeline(
    X_train, X_val, X_test,
    sequence_length=30,
    forecast_horizon=1,
    add_lags=True,      # Add lagged features
    add_ma=True         # Add moving averages
)
```

**Operations:**
- Data scaling (fit on train, transform all sets)
- Lagged features (t-1, t-7, t-14)
- Moving averages (7, 14, 30 day)
- Sequence creation for supervised learning

### 3. Models (`model.py`)

Two deep learning architectures:

#### LSTM Model
```python
from model import LSTMModel

model = LSTMModel(
    input_shape=(30, 5),      # 30 timesteps, 5 features
    output_shape=(1, 5),      # Forecast 1 step ahead
    lstm_units=64,
    dropout_rate=0.2,
    learning_rate=0.001
)

history = model.train(X_train, y_train, X_val, y_val, epochs=100)
predictions = model.predict(X_test)
```

#### TCN Model
```python
from model import TCNModel

model = TCNModel(
    input_shape=(30, 5),
    output_shape=(1, 5),
    filters=64,
    kernel_size=3,
    dropout_rate=0.2
)
```

### 4. Hyperparameter Tuning (`tuning.py`)

Automated optimization with Optuna:

```python
from tuning import HyperparameterTuner

tuner = HyperparameterTuner(n_trials=50)

def train_wrapper(params, X_train, y_train, X_val, y_val):
    # Training logic returning validation loss
    return val_loss

objective = tuner.create_objective(
    train_wrapper, X_train, y_train, X_val, y_val,
    model_type='lstm'
)

results = tuner.optimize(objective)
print(f"Best params: {results['best_params']}")
```

### 5. Evaluation (`evaluation.py`)

Comprehensive metrics and baselines:

```python
from evaluation import ModelEvaluator, BaselineModels

evaluator = ModelEvaluator()

# Evaluate deep learning model
metrics = evaluator.evaluate(y_test, y_pred, model_name="LSTM")

# Compare with baselines
y_naive = BaselineModels.naive_forecast(y_train, steps=len(y_test))
evaluator.evaluate(y_test, y_naive, model_name="Naive")

evaluator.print_results()
```

**Available Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- MSE (Mean Squared Error)
- R² (Coefficient of Determination)
- DTW (Dynamic Time Warping)
- Theil's U Statistic
- Directional Accuracy

### 6. XAI Analysis (`xai_analysis.py`)

SHAP-based explainability:

```python
from xai_analysis import TimeSeriesSHAPAnalyzer, ModelExplainability

analyzer = TimeSeriesSHAPAnalyzer(model, background_data=X_train)
analyzer.create_explainer(explainer_type='kernel')

results = analyzer.explain_predictions(X_test)
importance = analyzer.get_feature_importance()

for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")

# Plot interpretability
analyzer.plot_summary(max_display=10)
analyzer.plot_dependence(feature_idx=0)
```

### 7. Main Pipeline (`main.py`)

Orchestrates the complete workflow:

```python
from main import TimeSeriesForecastingPipeline

config = TimeSeriesForecastingPipeline._get_default_config()
config['model_type'] = 'lstm'
config['epochs'] = 150

pipeline = TimeSeriesForecastingPipeline(config)
pipeline.run_full_pipeline()
```

---

## Performance Benchmarking

The system includes built-in benchmarking against statistical baselines:

| Model | RMSE | MAE | MAPE | R² | Theil's U |
|-------|------|-----|------|-------|----------|
| LSTM  | 4.23 | 3.12| 0.045| 0.876 | 0.923    |
| TCN   | 4.18 | 3.08| 0.044| 0.881 | 0.918    |
| Naive | 8.45 | 6.23| 0.089| 0.421 | 1.425    |
| ARIMA | 5.67 | 4.15| 0.062| 0.754 | 1.087    |

*(Example values - actual results depend on data)*

---

## Explainability & Interpretability

### SHAP Feature Importance

The XAI module provides insights into which features drive predictions:

```python
# Global feature importance
shap_importance = analyzer.get_feature_importance()
# Returns: {'Feature_0': 0.245, 'Feature_1': 0.189, ...}

# Local explanations (per-sample)
local_explanation = analyzer.get_local_explanations(sample_idx=0)
# Shows top contributing features for specific prediction
```

### When to Use Each Technique

| Technique | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **SHAP** | Global & local explanations | Theoretically sound, handles interactions | Computationally expensive |
| **Permutation Importance** | Feature importance ranking | Model-agnostic, fast | Doesn't show direction of impact |
| **LIME** | Local explanations | Good for specific predictions | Approximate interpretations |

---

## Advanced Features

### 1. Ensemble Models

Combine multiple models for improved robustness:

```python
from model import EnsembleModel

ensemble = EnsembleModel()
ensemble.add_model(lstm_model)
ensemble.add_model(tcn_model)

predictions = ensemble.predict(X_test, weights=[0.6, 0.4])
```

### 2. Custom Configuration

Create domain-specific configurations:

```python
financial_config = {
    'n_features': 10,  # More features for financial data
    'sequence_length': 60,  # Longer context
    'forecast_horizon': 5,  # Multi-step ahead
    'lstm_units': 256,  # Larger model
}

pipeline = TimeSeriesForecastingPipeline(financial_config)
pipeline.run_full_pipeline()
```

### 3. Batch Predictions

Efficient prediction on large datasets:

```python
# Single prediction
y_pred = model.predict(X_test[:1])

# Batch prediction with memory efficiency
batch_size = 100
predictions = []
for i in range(0, len(X_test), batch_size):
    batch_pred = model.predict(X_test[i:i+batch_size])
    predictions.append(batch_pred)

y_pred_full = np.vstack(predictions)
```

---

## Troubleshooting

### GPU/CUDA Issues
```bash
# Check TensorFlow GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU
export CUDA_VISIBLE_DEVICES=-1
python main.py
```

### Memory Issues
```python
# Reduce batch size in config
config['batch_size'] = 16

# Reduce number of SHAP samples
config['n_shap_samples'] = 25
```

### Slow SHAP Analysis
```python
# Use gradient explainer (faster but requires TensorFlow model access)
analyzer.create_explainer(explainer_type='gradient')

# Reduce background data size
analyzer = TimeSeriesSHAPAnalyzer(model, background_data, n_samples=50)
```

---

## Output Files

### results.json

Complete results summary:

```json
{
  "data_metadata": {
    "n_samples": 1095,
    "n_features": 5,
    "train_size": 876,
    "val_size": 109,
    "test_size": 110
  },
  "metrics": {
    "rmse": 4.23,
    "mae": 3.12,
    "mape": 0.045,
    "r_squared": 0.876
  },
  "xai_analysis": {
    "shap_feature_importance": {...},
    "permutation_importance": {...}
  }
}
```

---

## Research & References

### Key Papers

1. **LSTM Networks**: [Hochreiter & Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
2. **TCN for Time Series**: [Bai et al. (2018)](https://arxiv.org/abs/1803.01271)
3. **SHAP Values**: [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)
4. **Time Series Forecasting**: [Benidis et al. (2020)](https://arxiv.org/abs/1906.04397)

### Related Libraries

- **statsmodels**: Traditional statistical models (ARIMA, Prophet alternatives)
- **PyTorch Forecasting**: Advanced deep learning for time series
- **Prophet**: Facebook's forecasting tool
- **AutoML**: Auto-sklearn, TPOT for automated model selection

---

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Multi-step ahead forecasting
- [ ] Attention mechanisms
- [ ] Transformer models
- [ ] Uncertainty quantification
- [ ] Cross-validation strategies
- [ ] Real-world data connectors

---

## License

This project is provided as-is for educational and research purposes.

---

## Contact & Support

For issues, questions, or suggestions:
1. Check existing documentation
2. Review example configurations
3. Examine module docstrings

---

## Changelog

### Version 1.0.0 (Initial Release)
- ✅ LSTM and TCN models
- ✅ Optuna hyperparameter tuning
- ✅ SHAP explainability
- ✅ Comprehensive evaluation metrics
- ✅ Production-ready pipeline

---

**Last Updated**: February 2026
**Maintainer**: Advanced ML Team
