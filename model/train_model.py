# model/train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.preprocess import fit_preprocess

# 데이터 불러오기
data = pd.read_csv("data/bacillus1.csv", encoding="utf-8-sig")
data.columns = data.columns.str.strip()
print(data.head())
print(data.columns)
X = data[['period', 'month', 'tank', 'initial_cfu', 'moisture', 'salt', 'pH']]
y = data["bacillus_cfu"]

print(X.columns)

# 전처리 (학습 전용)
X, scaler, feature_columns, num_cols = fit_preprocess(X)

# 학습/검증 분리 (seed 고정)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 모델 학습 (seed 고정)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 검증
y_pred = model.predict(X_val)
print(f"Validation MSE: {mean_squared_error(y_val, y_pred):.2f}")
print(f"Validation R2: {r2_score(y_val, y_pred):.2f}")

# 모델, 스케일러, 컬럼 저장
import time
meta = {
    "created_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_script": __file__ if '__file__' in globals() else None,
    "num_cols": num_cols if 'num_cols' in locals() else None
}
with open("model/cfu_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feature_columns": feature_columns, "meta": meta}, f)
