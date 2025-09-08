# utils/ontology.py
"""
Simple ontology/graph builder from prediction history.

Outputs:
 - networkx.Graph saved to GraphML: data/ontology.graphml
 - JSON node-link: data/ontology.json
 - triple list: data/ontology_triples.json

Node types:
 - Sample:<id>
 - Condition:<temp/humidity/aw/pH/tank>
 - Measurement:Predicted:<id>, Measurement:Actual:<id>
 - Model:<model_version>

Edges:
 - Sample -> hasCondition -> Condition nodes
 - Sample -> hasMeasurement -> Measurement nodes
 - Measurement -> measuredBy -> Model
"""

import json
import os
import networkx as nx
import sqlite3
from datetime import datetime

DB = "data/predictions.db"

def _get_all_preds(db_path=DB):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, model_version, inputs_json, predicted_cfu, env_pred_rate, env_multiplier, actual_final_cfu, actual_time_hours, computed_growth_rate_per_hour, verified FROM predictions")
        rows = cur.fetchall()
        cols = [d[0] for d in conn.execute("PRAGMA table_info(predictions)")]
        return rows
    finally:
        conn.close()

def build_graph(db_path=DB, out_graphml="data/ontology.graphml", out_json="data/ontology.json", out_triples="data/ontology_triples.json"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, model_version, inputs_json, predicted_cfu, env_pred_rate, env_multiplier, actual_final_cfu, actual_time_hours, computed_growth_rate_per_hour, verified FROM predictions")
    rows = cur.fetchall()
    colnames = [i[1] for i in cur.execute("PRAGMA table_info(predictions)")]
    G = nx.DiGraph()
    triples = []
    for r in rows:
        # unpack with indices
        (rid, timestamp, model_version, inputs_json, predicted_cfu, env_pred_rate, env_multiplier, actual_final_cfu, actual_time_hours, computed_growth_rate_per_hour, verified) = r
        try:
            inputs = json.loads(inputs_json or "{}")
        except Exception:
            inputs = {}
        sample_node = f"Sample:{rid}"
        G.add_node(sample_node, type="Sample", id=rid, timestamp=timestamp)
        # add condition nodes
        for cond in ["temp","temp_C","humidity","rh","RH_pct","aw","pH","tank","period","moisture","salt","initial_cfu"]:
            # check multiple keys
            keys = [cond] if cond in ["aw","pH","period","moisture","salt","initial_cfu"] else [cond, cond.lower()]
            for k in keys:
                if k in inputs and inputs[k] is not None:
                    val = inputs[k]
                    cond_node = f"Condition:{k}={val}"
                    if not G.has_node(cond_node):
                        G.add_node(cond_node, type="Condition", key=k, value=str(val))
                    G.add_edge(sample_node, cond_node, relation="hasCondition")
                    triples.append((sample_node, "hasCondition", cond_node))
                    break
        # predicted measurement
        pred_node = f"Measurement:Predicted:{rid}"
        G.add_node(pred_node, type="Measurement", mode="predicted", value=predicted_cfu)
        G.add_edge(sample_node, pred_node, relation="hasMeasurement")
        triples.append((sample_node, "hasMeasurement", pred_node))
        # measurement -> model
        model_node = f"Model:{model_version}"
        if not G.has_node(model_node):
            G.add_node(model_node, type="Model", version=model_version)
        G.add_edge(pred_node, model_node, relation="measuredBy")
        triples.append((pred_node, "measuredBy", model_node))
        # actual measurement if exists
        if actual_final_cfu is not None:
            act_node = f"Measurement:Actual:{rid}"
            G.add_node(act_node, type="Measurement", mode="actual", value=actual_final_cfu)
            G.add_edge(sample_node, act_node, relation="hasMeasurement")
            triples.append((sample_node, "hasMeasurement", act_node))
            # if actual measured, link to computed growth rate
            if computed_growth_rate_per_hour is not None:
                G.add_node(f"Derived:mu:{rid}", type="Derived", value=computed_growth_rate_per_hour)
                G.add_edge(act_node, f"Derived:mu:{rid}", relation="yields")
                triples.append((act_node, "yields", f"Derived:mu:{rid}"))
    # save graphml and json
    os.makedirs(os.path.dirname(out_graphml) or ".", exist_ok=True)
    nx.write_graphml(G, out_graphml)
    data = nx.node_link_data(G)
    with open(out_json, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(out_triples, "w", encoding="utf8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)

    actuals = [n for n in G.nodes(data=True) if n[1].get("type")=="Measurement" and n[1].get("mode")=="actual"]
    preds   = [n for n in G.nodes(data=True) if n[1].get("type")=="Measurement" and n[1].get("mode")=="predicted"]
    perf = {}
    if actuals and preds:
        try:
            import numpy as np
            y_true = [a[1]["value"] for a in actuals]
            y_pred = [p[1]["value"] for p in preds[:len(y_true)]]
            from sklearn.metrics import mean_squared_error, r2_score
            perf = {
                "n_samples": len(y_true),
                "rmse": float(np.sqrt(mean_squared_error(y_true,y_pred))),
                "r2": float(r2_score(y_true,y_pred))
            }
        except Exception:
            perf = {}
    return {
        "graphml": out_graphml,
        "json": out_json,
        "triples": out_triples,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "graph": G,
        "node_link": data,
        "triples_data": triples,
        "perf": perf,
    }
