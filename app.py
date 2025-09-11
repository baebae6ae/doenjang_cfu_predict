# app.py (history + ontology 통합 버전)
import streamlit as st
import pandas as pd
import pickle
from utils.preprocess import transform_preprocess
from env_model.env_inference import load_env_model, predict_env_effect, growth_multiplier_from_rate
from utils.history import PredictionHistory
from utils.postprocess import apply_scurve_adjustment
from utils import ontology as ontology_mod
import os
import numpy as np
from datetime import datetime
from utils.ontology_rules import OntologyRuleManager

st.set_page_config(page_title="콩된장 숙성물 Bacillus cereus 증식 예측", layout="wide")
st.title("콩된장 숙성물 Bacillus cereus 증식 예측 앱 (Env + records + History + Ontology)")

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
            month = st.number_input("숙성 시작 월", 1, 12, 6)
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
            "period": period, "month": month, "pH": pH, "tank": tank,
            "env_pred_rate_per_hour": predicted_rate, "env_multiplier_30d": multiplier_30d
        }])

        # transform via preprocess
        try:
            input_processed = transform_preprocess(input_df, scaler=scaler, feature_columns=feature_columns, saved_num_cols=saved_num_cols)
        except KeyError as e:
            st.error(f"Preprocessing error: {e}. Check model/feature mismatch.")
            st.stop()

        pred_cfu = model.predict(input_processed)[0]

        # Phase 3 보정 적용
        pred_cfu_adj = apply_scurve_adjustment(pred_cfu)

        # Phase 4: Ontology 기반 보정 적용
        try:
            # 룰 매니저 초기화
            rule_manager = OntologyRuleManager()

            # History DB 불러오기
            df_hist = history.list_predictions(limit=1000)
            df_hist = df_hist.dropna(subset=["actual_final_cfu"])
            df_hist = df_hist[df_hist["verified"] == 1]

            if not df_hist.empty:
                rules = rule_manager.extract_rules(df_hist)
                factor = rule_manager.apply_rules(input_dict)
                pred_cfu_final = pred_cfu * factor
                st.success(f"Ontology 룰 적용 최종 예측: {pred_cfu_final:.0f} CFU/g")
                if rules:
                    st.info(f"적용된 룰 개수: {len(rules)}")
            else:
                pred_cfu_final = pred_cfu
                st.warning("검증된 히스토리 데이터가 부족하여 Ontology 룰 적용 불가")
        except Exception as e:
            pred_cfu_final = pred_cfu
            st.error(f"Ontology 룰 적용 중 오류 발생: {e}")

        st.success(f"ML 모델 출력값: {pred_cfu:.0f} CFU/g")
        st.success(f"S-curve 보정값: {pred_cfu_adj:.0f} CFU/g")
        st.write(f"환경 기반 multiplier(30d): {multiplier_30d:.3f}")

        # save to history
        rec_id = history.save_prediction(inputs=input_dict, predicted_cfu=float(pred_cfu_final),
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
        st.dataframe(df[["id","timestamp","predicted_cfu","env_pred_rate","env_multiplier","actual_final_cfu","verified","model_version", "notes"]])
        sel = st.number_input("수정할 기록 ID 입력 (ID)", value=int(df.iloc[0]["id"]), min_value=int(df["id"].min()), max_value=int(df["id"].max()))
        if st.button("선택한 기록 삭제"):
            history.delete(int(sel))
            st.success(f"레코드 {sel} 삭제 완료. 페이지를 새로고침하세요.")
        rec = history.get_prediction(int(sel))
        if rec:
            st.subheader(f"레코드 {sel} 상세")
            st.json({
                "id": rec["id"],
                "timestamp": rec["timestamp"],
                "inputs": rec.get("inputs"),
                "predicted_cfu": rec.get("predicted_cfu"),
                "env_pred_rate": rec.get("env_pred_rate"),
                "env_multiplier": rec.get("env_multiplier"),
                "actual_final_cfu": rec.get("actual_final_cfu"),
                "actual_time_hours": rec.get("actual_time_hours"),
                "computed_growth_rate_per_hour": rec.get("computed_growth_rate_per_hour"),
                "verified": rec.get("verified"),
                "model_version": rec["model_version"],
                "notes": rec.get("notes")
            },expanded=False)
            st.markdown("### 실제 관측값 입력 / 수정")
            with st.form("actual_form"):
                act_final = st.number_input("실제 최종 CFU/g (actual_final_cfu)", value=rec.get("actual_final_cfu") or 0.0, step=1.0)
                act_time = st.number_input("실제 숙성 기간 (days)", value=(rec.get("actual_time_hours")/24.0) if rec.get("actual_time_hours") is not None else 1.0, step=1.0)
                add_note = st.text_input("추가 메모", value="")
                verify_flag = st.checkbox("확인(verified)으로 표시", value=bool(rec.get("verified")))
                submit_actual = st.form_submit_button("저장 (Update actual)")
            if submit_actual:
                history.update_actual(int(sel), actual_final_cfu=float(act_final), actual_time_hours=(float(act_time)*24.0),
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

        # 📑 Ontology Rule Report
        st.subheader("Ontology 룰 리포트 (자동 추출)")
        try:
            mgr = OntologyRuleManager(min_samples=5, error_threshold=0.10, bins=4)
            df_hist = history.list_predictions(limit=2000)
            # keep rows with predicted and actual
            df_hist = df_hist.dropna(subset=["predicted_cfu", "actual_final_cfu"])
            # prefer verified rows if available
            if "verified" in df_hist.columns and df_hist["verified"].sum() >= mgr.min_samples:
                df_hist = df_hist[df_hist["verified"]==1]
            rules = mgr.extract_rules(df_hist)
            if rules:
                rules_df = pd.DataFrame(rules)
                st.dataframe(rules_df[["feature","type","repr","n_samples","mean_error","rel_error","factor"]])
                st.markdown("#### 추천 해석")
                for r in rules:
                    sign = "증가" if r["factor"]>1 else "감소"
                    st.markdown(f"- `{r['repr']}`: 샘플수={r['n_samples']}, 평균오차={r['mean_error']:.1f}, 상대오차={r['rel_error']:.2%} → **{sign} {abs(1-r['factor']):.2%}**")
            else:
                st.info("추출된 룰이 없습니다 (데이터 부족 또는 오차 임계치 미달).")

        except Exception as e:
            st.error("Ontology 룰 추출 중 오류 발생: " + str(e))

        # 📊 Feature Importance (직관적 해석용)
        st.subheader("Feature Importance (모델이 중요하게 보는 요인)")

        try:
            if hasattr(model, "feature_importances_") and feature_columns:
                importances = model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)

                # ❌ 불필요한 원본 환경 변수 제거 (main 모델에는 의미 없음)
                drop_feats = ["temp", "humidity", "aw"]
                fi_df = fi_df[~fi_df["Feature"].isin(drop_feats)]

                # ✅ 환경 변수 매핑 사전
                feature_map = {
                    "env_pred_rate_per_hour": "환경 기반 예측 성장률 (temp, humidity, aw 반영)",
                    "env_multiplier_30d": "환경 기반 증식 배수 (30일 기준, temp, humidity, aw 반영)"
                }

                # 해석용 컬럼 추가
                fi_df["Interpretation"] = fi_df["Feature"].map(feature_map).fillna(fi_df["Feature"])

                # Plotly로 보기 좋게 시각화 (라벨 수평 정렬)
                import plotly.express as px
                fig = px.bar(
                    fi_df.sort_values("Importance", ascending=True),
                    x="Importance",
                    y="Interpretation",
                    orientation="h",
                    title="Feature Importance (해석 포함)"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.write("- 수치가 높을수록 해당 변수가 Bacillus cereus 증식 예측에 더 큰 영향을 줍니다.")
            else:
                st.info("모델에서 Feature Importance를 지원하지 않거나 feature_columns 정보가 없습니다.")
        except Exception as e:
            st.error("Feature Importance 표시 중 오류 발생: " + str(e))