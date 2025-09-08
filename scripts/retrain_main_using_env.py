"""
retrain_main_using_env.py

- 목적: 새 EnvModel(예: models/env_model_from_history_v1.pkl)을 사용해 main dataset(bacillus1.csv)에 env-derived features를 만들고 MainModel을 재학습
- 출력: model/cfu_model_retrained_from_history.pkl

Usage:
  python scripts/retrain_main_using_env.py --main data/bacillus1.csv --env models/env_model_from_history_v1.pkl --out model/cfu_model_retrained_from_history.pkl
"""
import argparse, os, pickle, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.preprocess import fit_preprocess
from env_model.env_inference import load_env_model, predict_env_effect, growth_multiplier_from_rate

SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--main", default="data/bacillus1.csv")
parser.add_argument("--env", default="models/env_model_from_history_v1.pkl")
parser.add_argument("--out", default="model/cfu_model_retrained_from_history.pkl")
args = parser.parse_args()

# Load main data
main = pd.read_csv(args.main)
main.columns = main.columns.str.strip()

# Ensure env columns exist
for col in ["temp","humidity","aw"]:
    if col not in main.columns:
        main[col] = np.nan

# Load env model pipeline
env_pipe = load_env_model(args.env)

# Predict env_rate for each main row (use measured if present else predict using median fallbacks)
env_rates = []
env_mul_30d = []
for _, row in main.iterrows():
    temp = row.get("temp", np.nan)
    RH = row.get("humidity", np.nan)
    aw = row.get("aw", np.nan)
    pH = row.get("pH", 6.2) if "pH" in row and pd.notna(row.get("pH")) else 6.2
    substrate = row.get("substrate", "soybean_paste") if "substrate" in row else "soybean_paste"
    initial = row.get("initial_cfu", 1e3)
    if pd.notna(temp) and pd.notna(RH) and pd.notna(aw):
        rate = predict_env_effect(env_pipe, float(temp), float(RH), float(aw), float(pH), substrate, float(initial))
    else:
        # impute with medians
        rate = predict_env_effect(env_pipe, 25.0, 80.0, 0.95, float(pH), substrate, float(initial))
    env_rates.append(rate)
    env_mul_30d.append(growth_multiplier_from_rate(rate, hours=24*30))

main["env_pred_rate_per_hour"] = env_rates
main["env_multiplier_30d"] = env_mul_30d

# Use existing preprocess to include new numeric columns (ensure NUM_COLS in utils includes them)
X = main.drop(columns=["bacillus_cfu"])
y = main["bacillus_cfu"]

X_ext = X.copy()
X_ext["env_pred_rate_per_hour"] = main["env_pred_rate_per_hour"]
X_ext["env_multiplier_30d"] = main["env_multiplier_30d"]

X_proc, scaler, feature_columns = fit_preprocess(X_ext)

X_train, X_val, y_train, y_val = train_test_split(X_proc, y, test_size=0.3, random_state=SEED)
model = RandomForestRegressor(n_estimators=200, random_state=SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Retrained MainModel Validation MSE:", mean_squared_error(y_val, y_pred))
print("Retrained MainModel R2:", r2_score(y_val, y_pred))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feature_columns": feature_columns, "num_cols": ["moisture","salt","initial_cfu","period","pH","env_pred_rate_per_hour","env_multiplier_30d"]}, f)
print("Saved retrained main model to", args.out)
