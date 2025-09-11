# env_model/env_inference.py
"""
Provides:
  - load_env_model(path)
  - predict_env_effect(temp, RH, aw, pH, substrate, initial_cfu): returns predicted growth_rate_per_hour
  - predict_growth_multiplier(...): optional helper to compute multiplier for CFU over a period
"""

import pickle
import numpy as np
import pandas as pd

def load_env_model(path="models/env_model_v1.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["pipeline"]

def predict_env_effect(env_pipeline, temp, humidity, aw, pH, substrate, initial_cfu):
    """
    Safe wrapper around env_pipeline.predict.
    Ensures numeric conversion and returns float(rate per hour).
    """
    # validate inputs and coerce
    temp = float(temp) if temp is not None and not np.isnan(temp) else 25.0
    humidity = float(humidity) if humidity is not None and not np.isnan(humidity) else 80.0
    aw = float(aw) if aw is not None and not np.isnan(aw) else 0.95
    pH = float(pH) if pH is not None and not np.isnan(pH) else 6.2
    initial_cfu = float(initial_cfu) if initial_cfu is not None else 1e3

    # prepare df according to pipeline preprocessor expectation (simple)
    X = {"temp_C": [temp], "RH_pct": [humidity], "aw":[aw], "pH":[pH], "initial_cfu_g":[initial_cfu], "substrate":[substrate]}
    import pandas as pd
    Xdf = pd.DataFrame(X)
    try:
        pred = env_pipeline.predict(Xdf)
        rate = float(pred[0])
    except Exception:
        rate = 0.0
    return rate

def compute_multiplier(growth_rate_per_hour, hours, initial_cfu, carrying_capacity=1e9):
    """Logistic model 기반 multiplier"""
    if growth_rate_per_hour <= 0:
        return 1.0
    K = carrying_capacity
    N0 = initial_cfu
    t = hours
    Nt = K / (1 + ((K - N0) / N0) * np.exp(-growth_rate_per_hour * t))
    return Nt / N0

def growth_multiplier_from_rate(rate_per_hour, hours, initial_cfu, carrying_capacity=1e9):
    """
    Compute growth multiplier using logistic growth to avoid extreme exponential explosion.
    N(t) = K / (1 + ((K - N0)/N0) * exp(-r * t))
    return multiplier = N(t) / N0
    """
    import math
    if rate_per_hour <= 0 or initial_cfu <= 0:
        return 1.0
    K = float(carrying_capacity)
    N0 = float(initial_cfu)
    t = float(hours)
    # convert hours if user passes days, ensure r*t within stable numeric
    try:
        exponent = -rate_per_hour * t
        denom = 1.0 + ((K - N0) / N0) * math.exp(exponent)
        Nt = K / denom
        multiplier = Nt / N0
    except OverflowError:
        # fallback to cap
        multiplier = min(K / N0, float('1e12'))
    return float(multiplier)
