"""
retrain_env_from_history.py

- 추출: data/predictions.db (verified rows)
- 구성: env training CSV (temp_C, RH_pct, aw, pH, substrate, initial_cfu_g, growth_rate_per_hour)
- 학습: XGBRegressor pipeline (same form as env_model/train_env_model.py)
- 저장: models/env_model_from_history_v1.pkl

Usage:
  python scripts/retrain_env_from_history.py --db data/predictions.db --out models/env_model_from_history_v1.pkl
"""
import argparse, sqlite3, json, os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
import pickle

SEED = 42
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--db", default="data/predictions.db")
parser.add_argument("--out", default="models/env_model_from_history_v1.pkl")
parser.add_argument("--min_samples", type=int, default=20)
args = parser.parse_args()

# 1) Load verified rows
conn = sqlite3.connect(args.db)
df_raw = pd.read_sql_query("SELECT * FROM predictions WHERE verified=1", conn)
conn.close()

if df_raw.empty:
    raise SystemExit("No verified records in DB. Cannot retrain EnvModel.")

# 2) Build env training DataFrame
rows = []
for _, r in df_raw.iterrows():
    try:
        inputs = json.loads(r["inputs_json"] or "{}")
    except Exception:
        inputs = {}
    # possible keys
    temp = inputs.get("temp", inputs.get("temp_C", None))
    rh = inputs.get("humidity", inputs.get("RH_pct", None))
    aw = inputs.get("aw", None)
    pH = inputs.get("pH", None)
    substrate = inputs.get("substrate", inputs.get("matrix", "soybean_paste"))
    initial = inputs.get("initial_cfu", inputs.get("initial_cfu_g", None))
    # prefer computed growth rate if present; else compute from actual fields
    growth = r.get("computed_growth_rate_per_hour")
    if pd.isna(growth) or growth is None:
        # try to compute from actual_final_cfu & actual_time_hours
        if pd.notna(r.get("actual_final_cfu")) and pd.notna(r.get("actual_time_hours")) and initial:
            try:
                growth = np.log(float(r["actual_final_cfu"]) / float(initial)) / float(r["actual_time_hours"])
            except Exception:
                growth = None
    if growth is None or pd.isna(growth):
        continue
    rows.append({
        "temp_C": float(temp) if temp is not None else np.nan,
        "RH_pct": float(rh) if rh is not None else np.nan,
        "aw": float(aw) if aw is not None else np.nan,
        "pH": float(pH) if pH is not None else np.nan,
        "substrate": substrate,
        "initial_cfu_g": float(initial) if initial is not None else np.nan,
        "growth_rate_per_hour": float(growth)
    })

df = pd.DataFrame(rows)
print("Extracted env training rows:", len(df))
if len(df) < args.min_samples:
    raise SystemExit(f"Not enough history training samples ({len(df)}). Need >= {args.min_samples}.")

# 3) Simple imputation / preproc
NUM_COLS = ["temp_C","RH_pct","aw","pH","initial_cfu_g"]
CAT_COLS = ["substrate"]
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

# 4) Train pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
], remainder="drop", sparse_threshold=0)

X = df[NUM_COLS + CAT_COLS]
y = df["growth_rate_per_hour"].values

model = XGBRegressor(n_estimators=200, random_state=SEED, n_jobs=4)
pipeline = Pipeline([("pre", preprocessor), ("model", model)])

kf = KFold(n_splits=5 if len(X)>=5 else max(2, len(X)), shuffle=True, random_state=SEED)
scores = -cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=1)
rmse = np.sqrt(scores)
print(f"CV RMSE mean={rmse.mean():.6f}, std={rmse.std():.6f}")

pipeline.fit(X, y)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "wb") as f:
    pickle.dump({"pipeline": pipeline, "train_rows": len(df)}, f)
print("Saved env model:", args.out)
