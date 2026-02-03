import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    days = 3 * 365
    t = np.arange(days)

    trend = 0.01 * t
    weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)

    feature_1 = trend + weekly_seasonality + np.random.normal(0, 2, days)
    feature_2 = np.roll(feature_1, 1) * 0.8 + np.random.normal(0, 1, days)
    feature_3 = np.random.normal(5, 1.5, days)
    feature_4 = 0.5 * feature_1 + np.random.normal(0, 1, days)
    feature_5 = np.random.normal(0, 3, days)

    target = (
        0.4 * feature_1 +
        0.3 * feature_2 +
        0.2 * feature_4 +
        np.random.normal(0, 2, days)
    )

    df = pd.DataFrame({
        "f1": feature_1,
        "f2": feature_2,
        "f3": feature_3,
        "f4": feature_4,
        "f5": feature_5,
        "target": target
    })

    df.to_csv("data/synthetic_data.csv", index=False)
    return df
