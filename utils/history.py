# utils/history.py
"""
Prediction history manager using SQLite.

Provides:
 - init_db(db_path)
 - save_prediction(...)
 - list_predictions(limit, offset)
 - get_prediction(id)
 - update_actual(id, actual_final_cfu, actual_time_hours, verified, notes)
 - export_csv(path)
"""

import sqlite3
import json
from datetime import datetime
import hashlib
import os
import pandas as pd

DEFAULT_DB = "data/predictions.db"

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    model_version TEXT,
    inputs_json TEXT,
    predicted_cfu REAL,
    env_pred_rate REAL,
    env_multiplier REAL,
    predicted_hours INTEGER,
    actual_final_cfu REAL,
    actual_time_hours REAL,
    computed_growth_rate_per_hour REAL,
    verified INTEGER DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts ON predictions(timestamp);
"""

class PredictionHistory:
    def __init__(self, db_path=DEFAULT_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._conn()
        try:
            conn.executescript(CREATE_SQL)
            conn.commit()
        finally:
            conn.close()

    def _model_version_from_file(self, model_path):
        try:
            with open(model_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            mtime = os.path.getmtime(model_path)
            return f"{os.path.basename(model_path)}|sha256:{h[:10]}|mtime:{int(mtime)}"
        except Exception:
            return os.path.basename(model_path)

    def save_prediction(self, inputs: dict, predicted_cfu: float,
                        env_pred_rate: float=None, env_multiplier: float=None,
                        predicted_hours:int=24*30, model_path="model/cfu_model_retrained.pkl",
                        notes:str=""):
        ts = datetime.utcnow().isoformat()
        mv = self._model_version_from_file(model_path) if model_path else "unknown"
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO predictions(timestamp, model_version, inputs_json, predicted_cfu,
                                        env_pred_rate, env_multiplier, predicted_hours, notes)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (ts, mv, json.dumps(inputs, ensure_ascii=False), float(predicted_cfu),
                 None if env_pred_rate is None else float(env_pred_rate),
                 None if env_multiplier is None else float(env_multiplier),
                 predicted_hours, notes)
            )
            conn.commit()
            cur = conn.execute("SELECT last_insert_rowid()")
            rowid = cur.fetchone()[0]
            return int(rowid)
        finally:
            conn.close()

    def list_predictions(self, limit=200, offset=0):
        conn = self._conn()
        try:
            df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit} OFFSET {offset}", conn)
            return df
        finally:
            conn.close()

    def get_prediction(self, record_id):
        conn = self._conn()
        try:
            cur = conn.execute("SELECT * FROM predictions WHERE id=?", (record_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("PRAGMA table_info(predictions)")]
            # Using sqlite3.Row would be easier; but return dict
            colnames = [i[1] for i in conn.execute("PRAGMA table_info(predictions)")]
            d = dict(zip(colnames, row))
            # parse inputs_json
            try:
                d["inputs"] = json.loads(d.get("inputs_json") or "{}")
            except Exception:
                d["inputs"] = {}
            return d
        finally:
            conn.close()

    def update_actual(self, record_id, actual_final_cfu=None, actual_time_hours=None, verified=1, notes=None):
        # compute growth_rate_per_hour if possible: ln(final/initial)/time
        rec = self.get_prediction(record_id)
        if not rec:
            raise ValueError("Record not found")
        inputs = rec.get("inputs", {})
        initial = inputs.get("initial_cfu") or inputs.get("initial_cfu_g") or None
        computed = None
        if actual_final_cfu is not None and actual_time_hours is not None and initial is not None:
            try:
                initial = float(initial)
                final = float(actual_final_cfu)
                t = float(actual_time_hours)
                if initial > 0 and final > 0 and t > 0:
                    import math
                    computed = math.log(final/initial)/t
            except Exception:
                computed = None
        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE predictions
                SET actual_final_cfu=?, actual_time_hours=?, computed_growth_rate_per_hour=?, verified=?, notes=COALESCE(notes, '') || ?
                WHERE id=?
                """,
                (actual_final_cfu, actual_time_hours, computed, int(verified), (("\n"+(notes or "")) if notes else ""), record_id)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def export_csv(self, out_path="data/predictions_export.csv"):
        df = self.list_predictions(limit=1000000)
        df.to_csv(out_path, index=False)
        return out_path

    def delete(self, record_id):
        conn = self._conn()
        try:
            conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
            conn.commit()
        finally:
            conn.close()
