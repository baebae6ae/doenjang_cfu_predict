# env_model/train_env_model.py
"""
Usage:
  python env_model/train_env_model.py --data data/bacillus_env_growth_dataset.csv --out models/env_model_v1.pkl

What it does:
  - loads bacillus_env_growth_dataset.csv
  - basic cleaning, feature engineering (temp, RH, aw, pH, substrate one-hot, initial_cfu)
  - computes target 'growth_rate_per_hour' if missing from final/initial/time
  - trains XGBoost regressor with 5-fold CV (random_state fixed)
  - saves model pipeline (preprocessor + xgb) to models/env_model_v1.pkl
"""

import os, argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import pickle

SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/bacillus_env_growth_dataset.csv")
parser.add_argument("--out", default="models/env_model_v1.pkl")
args = parser.parse_args()

df = pd.read_csv(args.data, encoding="utf-8-sig")
# basic cleaning
df.columns = df.columns.str.strip()
# compute growth_rate_per_hour if missing
if "growth_rate_per_hour" not in df.columns:
    # growth_rate = ln(final/initial)/time_hours
    def compute_rate(row):
        try:
            if pd.notna(row.get("final_cfu_g")) and pd.notna(row.get("initial_cfu_g")) and pd.notna(row.get("time_hours")):
                if row["initial_cfu_g"]>0 and row["final_cfu_g"]>0 and row["time_hours"]>0:
                    return np.log(row["final_cfu_g"]/row["initial_cfu_g"]) / float(row["time_hours"])
            return np.nan
        except Exception:
            return np.nan
    df["growth_rate_per_hour"] = df.apply(compute_rate, axis=1)

# drop rows without target
df = df[df["growth_rate_per_hour"].notna()].copy()
print(f"Loaded {len(df)} env rows for training")

# features
NUM_COLS = ["temp_C", "RH_pct", "aw", "pH", "initial_cfu_g"]
CAT_COLS = ["substrate"]
for c in NUM_COLS:
    if c not in df.columns:
        df[c] = np.nan

# simple imputation: numeric median, categorical fill 'unknown'
for c in NUM_COLS:
    df[c] = df[c].fillna(df[c].median())

df["initial_cfu_g"] = df["initial_cfu_g"].astype(float)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
], remainder="drop", sparse_threshold=0)

X = df[NUM_COLS + CAT_COLS]
y = df["growth_rate_per_hour"].values

# model
model = XGBRegressor(n_estimators=200, random_state=SEED, n_jobs=4)

pipeline = Pipeline([("pre", preprocessor), ("model", model)])

# CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
scores = -cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=4)
rmse = np.sqrt(scores)
print(f"CV RMSE: mean={rmse.mean():.6f}, std={rmse.std():.6f}")

# fit on full data
pipeline.fit(X, y)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "wb") as f:
    pickle.dump({"pipeline": pipeline}, f)
print(f"Saved env model to {args.out}")
