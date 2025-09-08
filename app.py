# app.py (history + ontology 통합 버전)
import streamlit as st
import pandas as pd
import pickle
from utils.preprocess import transform_preprocess
from env_model.env_inference import load_env_model, predict_env_effect, growth_multiplier_from_rate
from utils.history import PredictionHistory
from utils import ontology as ontology_mod
import os
import numpy as np
from datetime import datetime

st.set_page_config(page_title="된장 Bacillus cereus CFU 예측 (Env + History)", layout="wide")
st.title("된장 Bacillus cereus CFU 예측 앱 (Env + History + Ontology)")

# ---- model loading ----
MODEL_PATH = "model/cfu_model_retrained.pkl"
ENV_MODEL_PATH = "models/env_model_v1.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"메인 모델 파일이 없습니다: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
model = saved.get("model")
scaler = saved.get("scaler")
feature_columns = saved.get("feature_columns", [])
meta = saved.get("meta", {})
saved_num_cols = meta.get("num_cols") if meta else None

env_pipeline = None
if os.path.exists(ENV_MODEL_PATH):
    env_pipeline = load_env_model(ENV_MODEL_PATH)

# ---- history manager init ----
history = PredictionHistory(db_path="data/predictions.db")

# ---- navigation ----
mode = st.sidebar.radio("모드 선택", ["Predict", "History", "Ontology"])

# ---------------- PREDICT ----------------
if mode == "Predict":
    st.header("CFU 예측")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            moisture = st.number_input("수분 (%)", 30.0, 70.0, 50.0)
            salt = st.number_input("염도 (%)", 5.0, 20.0, 12.0)
            initial_cfu = st.number_input("초기 Bacillus cereus 균수 (CFU/g)", 0, 1000000, 5000)
            period = st.number_input("숙성 기간 (일)", 1, 365, 30)
        with col2:
            temp = st.number_input("숙성 온도 (°C) (선택)", 0.0, 60.0, value=float("nan"), step=0.1, format="%.1f")
            humidity = st.number_input("숙성 습도 (%) (선택)", 0.0, 100.0, value=float("nan"), step=0.1, format="%.1f")
            aw = st.number_input("수분활성도 (a_w) (선택)", 0.0, 1.00, value=float("nan"), step=0.01, format="%.3f")
        with col3:
            pH = st.number_input("pH", 3.0, 8.5, 6.2)
            tank = st.number_input("탱크 순서", 1, 50, 1)
            notes = st.text_input("메모 (선택)")
        use_lit = st.checkbox("문헌기반 보정 사용 (env 값 미입력 시 대체/보정)", value=True)
        submitted = st.form_submit_button("CFU 예측")
    if submitted:
        # env handling
        if not (np.isfinite(temp) and np.isfinite(humidity) and np.isfinite(aw)) and env_pipeline is not None and use_lit:
            med_temp = 25.0 if np.isnan(temp) else temp
            med_RH = 80.0 if np.isnan(humidity) else humidity
            med_aw = 0.95 if np.isnan(aw) else aw
            predicted_rate = predict_env_effect(env_pipeline, med_temp, med_RH, med_aw, pH, "soybean_paste", initial_cfu)
            multiplier_30d = growth_multiplier_from_rate(predicted_rate, hours=24*30, initial_cfu=initial_cfu)
            st.info(f"문헌기반 예측 성장률(시간당 ln비): {predicted_rate:.6f}")
        elif env_pipeline is not None:
            predicted_rate = predict_env_effect(env_pipeline, temp, humidity, aw, pH, "soybean_paste", initial_cfu)
            multiplier_30d = growth_multiplier_from_rate(predicted_rate, hours=24*30, initial_cfu=initial_cfu)
            st.info(f"입력값 기반 예측 성장률(시간당 ln비): {predicted_rate:.6f}")
        else:
            predicted_rate = 0.0
            multiplier_30d = 1.0

        input_dict = {
            "moisture": moisture, "salt": salt, "initial_cfu": initial_cfu,
            "period": period, "pH": pH, "tank": tank, "temp": temp, "humidity": humidity, "aw": aw
        }

        # build dataframe for main model: include env features
        input_df = pd.DataFrame([{
            "moisture": moisture, "salt": salt, "initial_cfu": initial_cfu,
            "period": period, "pH": pH, "tank": tank,
            "env_pred_rate_per_hour": predicted_rate, "env_multiplier_30d": multiplier_30d
        }])

        # transform via preprocess
        try:
            input_processed = transform_preprocess(input_df, scaler=scaler, feature_columns=feature_columns, saved_num_cols=saved_num_cols)
        except KeyError as e:
            st.error(f"Preprocessing error: {e}. Check model/feature mismatch.")
            st.stop()

        pred_cfu = model.predict(input_processed)[0]

        st.success(f"예측 Bacillus cereus CFU: {pred_cfu:.0f} CFU/g")
        st.write(f"환경 기반 multiplier(30d): {multiplier_30d:.3f}")

        # save to history
        rec_id = history.save_prediction(inputs=input_dict, predicted_cfu=float(pred_cfu),
                                         env_pred_rate=float(predicted_rate), env_multiplier=float(multiplier_30d),
                                         predicted_hours=24*30, model_path=MODEL_PATH, notes=notes)
        st.info(f"예측 이력에 저장됨 (id={rec_id}). 'History' 탭에서 실제값을 추가/수정할 수 있습니다.")

# ---------------- HISTORY ----------------
elif mode == "History":
    st.header("예측 이력 (Predictions History)")
    df = history.list_predictions(limit=500)
    if df.empty:
        st.info("저장된 예측 이력이 없습니다.")
    else:
        st.dataframe(df[["id","timestamp","model_version","predicted_cfu","env_pred_rate","env_multiplier","actual_final_cfu","verified"]])
        sel = st.number_input("수정할 기록 ID 입력 (ID)", value=int(df.iloc[0]["id"]), min_value=int(df["id"].min()), max_value=int(df["id"].max()))
        rec = history.get_prediction(int(sel))
        if rec:
            st.subheader(f"레코드 {sel} 상세")
            st.json({
                "id": rec["id"],
                "timestamp": rec["timestamp"],
                "model_version": rec["model_version"],
                "inputs": rec.get("inputs"),
                "predicted_cfu": rec.get("predicted_cfu"),
                "env_pred_rate": rec.get("env_pred_rate"),
                "env_multiplier": rec.get("env_multiplier"),
                "actual_final_cfu": rec.get("actual_final_cfu"),
                "actual_time_hours": rec.get("actual_time_hours"),
                "computed_growth_rate_per_hour": rec.get("computed_growth_rate_per_hour"),
                "verified": rec.get("verified"),
                "notes": rec.get("notes"),
            })
            st.markdown("### 실제 관측값 입력 / 수정")
            with st.form("actual_form"):
                act_final = st.number_input("실제 최종 CFU/g (actual_final_cfu)", value=rec.get("actual_final_cfu") or 0.0, step=1.0)
                act_time = st.number_input("실제 경과 시간 (hours)", value=rec.get("actual_time_hours") or 24.0, step=1.0)
                add_note = st.text_input("추가 메모", value="")
                verify_flag = st.checkbox("확인(verified)으로 표시", value=bool(rec.get("verified")))
                submit_actual = st.form_submit_button("저장 (Update actual)")
            if submit_actual:
                history.update_actual(int(sel), actual_final_cfu=float(act_final), actual_time_hours=float(act_time),
                                      verified=1 if verify_flag else 0, notes=add_note)
                st.success("업데이트 완료. 페이지를 새로고침하면 변경된 값이 반영됩니다.")
            if st.button("Export history to CSV"):
                out = history.export_csv()
                with open(out, "rb") as f:
                    st.download_button("Download CSV", f, file_name=os.path.basename(out))
        else:
            st.error("해당 ID 레코드를 찾을 수 없습니다.")

    import subprocess, shlex, sys

    if st.button("Retrain MainModel with verified history rows"):
        cmd = [
            sys.executable,
            "scripts/retrain_main_with_history.py",
            "--db", "data/predictions.db",
            "--main", "data/bacillus1.csv",
            "--out", "model/cfu_model_retrained.pkl"
        ]

        # 현재 환경 복사
        env = os.environ.copy()
        # 프로젝트 루트를 PYTHONPATH에 추가
        env["PYTHONPATH"] = os.getcwd()

        st.write("Running:", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            st.write(line)
        proc.wait()
        if proc.returncode == 0:
            st.success("MainModel retrained with history and saved.")
        else:
            st.error("Retrain failed. See logs above.")


# ---------------- ONTOLOGY ----------------
elif mode == "Ontology":
    st.header("Ontology 기반 품질 검증")
    st.markdown("Prediction history를 바탕으로 **예측 성능**과 **조건별 오차 패턴**을 분석합니다.")

    if st.button("Build & Analyze Ontology"):
        res = ontology_mod.build_graph(
            db_path=history.db_path,
            out_graphml="data/ontology.graphml",
            out_json="data/ontology.json",
            out_triples="data/ontology_triples.json"
        )

        # ✅ 성능 요약
        if res["perf"]:
            st.metric("샘플 수", res["perf"]["n_samples"])
            st.metric("RMSE", f"{res['perf']['rmse']:.2f}")
            st.metric("R²", f"{res['perf']['r2']:.3f}")
        else:
            st.warning("실측 데이터 부족")

        # 📊 예측 vs 실제 비교 산점도
        st.subheader("예측 vs 실제 비교")
        nodes_df = pd.DataFrame(res["node_link"]["nodes"])
        actuals = nodes_df[(nodes_df["type"]=="Measurement") & (nodes_df["mode"]=="actual")]
        preds   = nodes_df[(nodes_df["type"]=="Measurement") & (nodes_df["mode"]=="predicted")]
        if not actuals.empty and not preds.empty:
            df_comp = pd.DataFrame({
                "pred": preds["value"].astype(float).values[:len(actuals)],
                "actual": actuals["value"].astype(float).values,
                "sample": actuals["id"].str.replace("Measurement:Actual:","")
            })
            df_comp["error"] = df_comp["pred"] - df_comp["actual"]
            st.scatter_chart(df_comp[["pred","actual"]])

        # 📈 조건별 평균 오차 분석
        st.subheader("조건별 평균 오차")
        edges_df = pd.DataFrame(res["node_link"]["links"])
        cond_edges = edges_df[edges_df["relation"]=="hasCondition"]
        cond_nodes = nodes_df[nodes_df["type"]=="Condition"]

        if not cond_nodes.empty and not actuals.empty:
            cond_summary = []
            for _, cond_row in cond_nodes.iterrows():
                cond_id = cond_row["id"]  # Condition:key=value
                key, val = cond_row["key"], cond_row["value"]

                # 연결된 샘플 찾기
                linked_samples = cond_edges[cond_edges["target"]==cond_id]["source"].str.replace("Sample:","")
                linked_samples = linked_samples.tolist()

                if len(linked_samples)==0:
                    continue

                subset = df_comp[df_comp["sample"].isin(linked_samples)]
                if not subset.empty:
                    cond_summary.append({
                        "조건": f"{key}={val}",
                        "샘플수": len(subset),
                        "평균 예측": subset["pred"].mean(),
                        "평균 실제": subset["actual"].mean(),
                        "평균 오차": subset["error"].mean()
                    })

            if cond_summary:
                cond_df = pd.DataFrame(cond_summary)
                st.dataframe(cond_df)

        # 🌐 네트워크 그래프 (실무용)
        st.subheader("샘플-조건-측정 연결망")
        from pyvis.network import Network
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        color_map = {"Sample":"#1f77b4","Condition":"#7f7f7f","Measurement":"#2ca02c","Model":"#9467bd","Derived":"#ff7f0e"}
        for node in res["graph"].nodes(data=True):
            n, attrs = node
            ntype = attrs.get("type","")
            label = n.split(":")[-1] if "Measurement" not in n else f"{attrs.get('mode')}:{attrs.get('value')}"
            net.add_node(n, label=label, color=color_map.get(ntype,"#cccccc"))
        for edge in res["graph"].edges(data=True):
            src, dst, attrs = edge
            net.add_edge(src, dst, label=attrs.get("relation",""))
        html_path = "data/ontology_preview.html"
        net.save_graph(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)
    else:
        st.info("Ontology를 생성하고 분석하려면 버튼을 누르세요.")
