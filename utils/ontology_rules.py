# utils/ontology_rules.py
"""
Ontology rule extractor & applier.

Usage:
    from utils.ontology_rules import OntologyRuleManager
    mgr = OntologyRuleManager(min_samples=5, error_threshold=0.12)
    rules = mgr.extract_rules(df_history)   # df_history = history.list_predictions(...)
    factor = mgr.apply_rules(input_dict)    # multiply predicted by factor

Output rule format (list of dicts):
 - feature, type ('numeric'|'categorical'), lower, upper (for numeric),
   value (for categorical), n_samples, mean_pred, mean_actual, mean_error, rel_error, factor
"""
import json
import math
import pandas as pd
import numpy as np

class OntologyRuleManager:
    def __init__(self, min_samples=5, error_threshold=0.10, bins=4):
        """
        min_samples: 최소 샘플 수 (그룹별)
        error_threshold: 상대오차(절대값) 임계치 (예: 0.10 = 10%)
        bins: numeric bin 개수
        """
        self.rules = []
        self.min_samples = int(min_samples)
        self.error_threshold = float(error_threshold)
        self.bins = int(bins)

    def _expand_inputs(self, df):
        """Inputs 확장: df는 history.list_predictions()로 얻은 DataFrame"""
        if "inputs" in df.columns and df["inputs"].apply(lambda x: isinstance(x, dict)).any():
            inputs_norm = pd.json_normalize(df["inputs"])
            inputs_norm.index = df.index
            df = pd.concat([df, inputs_norm], axis=1)
            return df

        if "inputs_json" in df.columns:
            # parse inputs_json safely (some rows may be null/empty)
            def _parse_js(s):
                if pd.isna(s):
                    return {}
                if isinstance(s, dict):
                    return s
                try:
                    return json.loads(s)
                except Exception:
                    # tolerate single quotes or malformed: try replace
                    try:
                        t = str(s).replace("'", '"')
                        return json.loads(t)
                    except Exception:
                        return {}
            parsed = df["inputs_json"].apply(_parse_js)
            if parsed.isnull().all():
                return df
            inputs_norm = pd.json_normalize(parsed)
            inputs_norm.index = df.index
            df = pd.concat([df, inputs_norm], axis=1)
            return df

        # no inputs column available
        return df

    def extract_rules(self, history_df, features=None):
        """
        history_df: DataFrame from history.list_predictions()
        returns: list of rule dicts
        """
        df = history_df.copy()
        if df.empty:
            self.rules = []
            return []

        # expand inputs JSON to columns if needed
        df = self._expand_inputs(df)

        # require predicted and actual
        if "predicted_cfu" not in df.columns or "actual_final_cfu" not in df.columns:
            self.rules = []
            return []

        # filter rows with both predicted and actual and (preferably) verified
        mask = df["predicted_cfu"].notnull() & df["actual_final_cfu"].notnull()
        if "verified" in df.columns:
            # keep verified==1 rows if any; otherwise use all with pred+actual
            if df["verified"].eq(1).sum() >= self.min_samples:
                mask = mask & df["verified"].eq(1)
        df = df.loc[mask].copy()
        if df.empty:
            self.rules = []
            return []

        # default feature list: include commonly used keys if exist
        default_feats = ["moisture","salt","pH","period","temp","humidity","aw","initial_cfu","month","tank"]
        if features is None:
            features = [f for f in default_feats if f in df.columns]
        else:
            features = [f for f in features if f in df.columns]

        rules = []

        # ensure numeric conversions for preds/actuals
        df["__pred"] = pd.to_numeric(df["predicted_cfu"], errors="coerce")
        df["__act"] = pd.to_numeric(df["actual_final_cfu"], errors="coerce")
        df = df[df["__act"].notnull() & df["__pred"].notnull()].copy()
        if df.empty:
            self.rules = []
            return []

        for feat in features:
            ser = df[feat]
            # try numeric conversion
            ser_num = pd.to_numeric(ser, errors="coerce")
            n_numeric = ser_num.notnull().sum()
            n_unique = ser.dropna().nunique()

            # Numeric-heavy: bin numeric into quantiles (qcut) if enough unique
            if n_numeric >= max(self.min_samples, 10) and n_unique > self.bins:
                try:
                    # qcut to balance samples per bin
                    bins = min(self.bins, n_numeric)
                    df["_feat_bin"] = pd.qcut(ser_num, q=bins, duplicates="drop")
                except Exception:
                    df["_feat_bin"] = pd.cut(ser_num, bins=self.bins, duplicates="drop")

                for interval, g in df.groupby("_feat_bin"):
                    if len(g) < self.min_samples:
                        continue
                    mean_pred = float(g["__pred"].mean())
                    mean_act = float(g["__act"].mean())
                    if mean_act == 0:
                        continue
                    mean_err = float((g["__pred"] - g["__act"]).mean())
                    rel_err = mean_err / mean_act
                    if abs(rel_err) >= self.error_threshold:
                        # get numeric bounds
                        try:
                            lower = float(interval.left)
                            upper = float(interval.right)
                        except Exception:
                            s = str(interval)
                            parts = s.strip("()[]").split(",")
                            try:
                                lower = float(parts[0])
                                upper = float(parts[1])
                            except Exception:
                                continue
                        factor = float(1.0 - rel_err)  # multiply predicted by this
                        rules.append({
                            "feature": feat,
                            "type": "numeric",
                            "lower": lower,
                            "upper": upper,
                            "n_samples": int(len(g)),
                            "mean_pred": mean_pred,
                            "mean_actual": mean_act,
                            "mean_error": mean_err,
                            "rel_error": rel_err,
                            "factor": factor,
                            "repr": f"{feat} in ({lower:.3g}, {upper:.3g}]"
                        })
                df.drop(columns=["_feat_bin"], inplace=True, errors="ignore")
            else:
                # treat as categorical / low-cardinality numeric
                for val, g in df.groupby(feat):
                    if pd.isna(val):
                        continue
                    if len(g) < self.min_samples:
                        continue
                    mean_pred = float(g["__pred"].mean())
                    mean_act = float(g["__act"].mean())
                    if mean_act == 0:
                        continue
                    mean_err = float((g["__pred"] - g["__act"]).mean())
                    rel_err = mean_err / mean_act
                    if abs(rel_err) >= self.error_threshold:
                        factor = float(1.0 - rel_err)
                        rules.append({
                            "feature": feat,
                            "type": "categorical",
                            "value": str(val),
                            "n_samples": int(len(g)),
                            "mean_pred": mean_pred,
                            "mean_actual": mean_act,
                            "mean_error": mean_err,
                            "rel_error": rel_err,
                            "factor": factor,
                            "repr": f"{feat} == {val}"
                        })

        # sort rules by n_samples desc then abs(rel_error)
        rules = sorted(rules, key=lambda r: (-r.get("n_samples",0), -abs(r.get("rel_error",0))))
        self.rules = rules
        return rules

    def apply_rules(self, sample: dict):
        """
        sample: dict of input values (e.g. {"moisture":50.0, ...})
        Returns multiplicative factor (float). Default 1.0 (no change).
        """
        factor = 1.0
        if not self.rules:
            return factor
        for r in self.rules:
            feat = r["feature"]
            if feat not in sample:
                continue
            val = sample.get(feat)
            if val is None:
                continue
            # numeric rule
            if r["type"] == "numeric":
                try:
                    v = float(val)
                except Exception:
                    continue
                # inclusive check: lower < = v <= upper
                if (v >= r["lower"]) and (v <= r["upper"]):
                    factor *= float(r["factor"])
            else:
                # categorical: string compare
                if str(val) == str(r.get("value")):
                    factor *= float(r["factor"])
        return factor
