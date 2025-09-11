# train_retrain_main.py
"""
Usage:
  python train_retrain_main.py --main_csv data/bacillus1.csv --env_csv data/bacillus_env_growth_dataset.csv --out model/cfu_model_retrained.pkl

What it does:
  - loads main training CSV (doenjang batches)
  - loads env model pipeline (models/env_model_v1.pkl) and predicts growth_rate per hour for each sample
    (if sample has measured temp/RH/aw, use measured; else use env_model to impute)
  - constructs a new feature 'env_pred_growth_rate' and 'env_multiplier_30d'
  - retrains RandomForest main model with these extra features
  - saves model + scaler + feature_columns
"""
import argparse, os, pickle
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
parser.add_argument("--main_csv", default="data/bacillus1.csv")
parser.add_argument("--env_model", default="models/env_model_v1.pkl")
parser.add_argument("--out", default="model/cfu_model_retrained.pkl")
args = parser.parse_args()

# load main data
main = pd.read_csv(args.main_csv)
main.columns = main.columns.str.strip()
# ensure columns exist
for col in ["temp","humidity","aw"]:
    if col not in main.columns:
        main[col] = np.nan

# Load env model
env_pipe = load_env_model(args.env_model)

# For each row, decide whether to use measured env values or impute
env_pred_rates = []
env_mul_30d = []
for _, row in main.iterrows():
    temp = row.get("temp", np.nan)
    RH = row.get("humidity", np.nan)
    aw = row.get("aw", np.nan)
    pH = row.get("pH", np.nan)
    substrate = row.get("substrate", "soybean_paste") if "substrate" in row else "soybean_paste"
    initial = row.get("initial_cfu", row.get("initial_cfu_g", 50))
    # if measured all env present -> predict using those; else use imputed via env model with median fallbacks
    if pd.notna(temp) and pd.notna(RH) and pd.notna(aw):
        rate = predict_env_effect(env_pipe, float(temp), float(RH), float(aw), float(pH if pd.notna(pH) else 5.2), substrate, float(initial))
    else:
        # impute using medians
        med_temp = 25
        med_RH = 80
        med_aw = 0.95
        rate = predict_env_effect(env_pipe, med_temp, med_RH, med_aw, float(pH if pd.notna(pH) else 5.2), substrate, float(initial))
    env_pred_rates.append(rate)
    hours = row.get("period") * 24
    env_mul_30d.append(growth_multiplier_from_rate(rate, hours=hours, initial_cfu=initial))
main["env_pred_rate_per_hour"] = env_pred_rates
main["env_multiplier_30d"] = env_mul_30d

# Now use existing feature pipeline: select features then fit
# we adapt existing preprocess.fit_preprocess to accept new feature 'env_pred_rate_per_hour', 'env_multiplier_30d'
# For simplicity, append these numeric features to X before calling fit_preprocess
X = main.drop(columns=["bacillus_cfu", "day"])
y = main["bacillus_cfu"]

# ensure our preprocess fit accepts new columns by including them in NUM_COLS in utils/preprocess.py
X_extended = X.copy()
X_extended["env_pred_rate_per_hour"] = main["env_pred_rate_per_hour"]
X_extended["env_multiplier_30d"] = main["env_multiplier_30d"]

# call fit_preprocess (make sure preprocess.NUM_COLS includes the new numeric names)
X_proc, scaler, feature_columns, num_cols = fit_preprocess(X_extended)

# train-test
X_train, X_val, y_train, y_val = train_test_split(X_proc, y, test_size=0.3, random_state=SEED)
model = RandomForestRegressor(n_estimators=200, random_state=SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Retrained MainModel Validation MSE:", mean_squared_error(y_val, y_pred))
print("Retrained MainModel R2:", r2_score(y_val, y_pred))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
import time
meta = {
    "created_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_script": __file__ if '__file__' in globals() else None,
    "num_cols": num_cols if 'num_cols' in locals() else None
}
with open(args.out, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feature_columns": feature_columns, "meta": meta}, f)
print("Saved retrained main model to", args.out)
