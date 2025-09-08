# utils/preprocess.py (PATCHED)
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 기본 numeric 후보 (훈련/추론에서 자주 쓰이는 이름)
DEFAULT_NUM_COLS = ["moisture","salt","initial_cfu","period","temp","humidity","pH","aw","env_pred_rate_per_hour","env_multiplier_30d"]

def fit_preprocess(df, num_cols=None):
    """
    학습 시 호출: returns (df_transformed, scaler, feature_columns, num_cols_used)
    num_cols: optional list to enforce numeric columns order for scaler
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    if num_cols is None:
        # choose intersection of DEFAULT with df columns in a stable order
        num_cols = [c for c in DEFAULT_NUM_COLS if c in df.columns]
    # ensure numeric columns exist
    for c in num_cols:
        if c not in df.columns:
            df[c] = pd.NA
    scaler = StandardScaler()
    # Fill numeric na with median before scaling
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().all():
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(df[c].median())
    df[num_cols] = scaler.fit_transform(df[num_cols])
    feature_columns = list(df.columns)
    return df, scaler, feature_columns, num_cols

def transform_preprocess(df, scaler, feature_columns, saved_num_cols=None):
    """
    예측 시 호출: feature_columns 순서/누락 보정 + scaler.transform 방어
    - saved_num_cols: 리스트(숫자 컬럼, in order) that scaler was fit on
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    # one-hot for tank if present and not numeric
    if "tank" in df.columns and not pd.api.types.is_numeric_dtype(df["tank"]):
        df = pd.get_dummies(df, columns=["tank"], drop_first=True)

    # ensure all feature_columns present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # keep only feature_columns (order)
    df = df[feature_columns]

    # decide numeric cols to transform
    if saved_num_cols:
        num_cols = [c for c in saved_num_cols if c in df.columns]
    else:
        # fallback: pick first n columns equal to scaler.mean_.shape
        if hasattr(scaler, "mean_"):
            n = len(scaler.mean_)
            cand = [c for c in DEFAULT_NUM_COLS if c in df.columns]
            num_cols = cand[:n] if len(cand) >= n else list(df.columns[:n])
        else:
            num_cols = [c for c in DEFAULT_NUM_COLS if c in df.columns]

    if len(num_cols) == 0:
        # nothing to transform
        return df

    # ensure numeric dtype and fill na
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df[num_cols] = scaler.transform(df[num_cols])
    return df
