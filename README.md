📄 README (English Version)

## Doenjang Bacillus cereus CFU Prediction System

### Overview

This project provides a **machine learning–based prediction system** for estimating the growth of *Bacillus cereus* in fermented soybean paste (**Doenjang**).
It integrates **literature-based environmental growth models** (temperature, humidity, water activity) with **factory process variables** (moisture, salt, pH, fermentation period, tank batch, initial CFU), enabling reliable and data-driven microbial risk management.

The system includes:

* Environmental growth rate model (`env_model`)
* Main CFU prediction model (`train_retrain_main.py`)
* A user-friendly web app (`app.py`) for prediction, history tracking, and ontology-based error analysis
* Ontology rule extraction for condition-specific correction and interpretability

---

### Motivation

Food safety management in fermentation industries requires **quantitative, predictive tools**.
Traditional spreadsheets or static tables cannot fully capture the complex interaction between environmental and process variables.
This project addresses that gap by combining:

* **Experimental/literature-based knowledge** (growth rate estimation using Arrhenius-like and logistic modeling)
* **Factory-specific process data** (Doenjang batches with measured CFU outcomes)
* **Adaptive learning** (automatic retraining using historical prediction records and actuals)

---

### Project Structure

```
doenjang_cfu_predict/
│
├── app.py                          # Streamlit app (prediction, history, ontology)
├── train_retrain_main.py           # Main model training/retraining script
├── env_model/
│   ├── train_env_model.py          # Train environmental growth model (XGBoost)
│   ├── env_inference.py            # Inference utilities for env_model
│
├── utils/
│   ├── preprocess.py               # Data preprocessing pipeline
│   ├── postprocess.py              # S-curve adjustment & ontology integration
│   ├── ontology_rules.py           # Ontology rule extraction & application
│   ├── history.py                  # Prediction history management (SQLite)
│
├── data/
│   ├── bacillus1.csv               # Main training dataset (factory batches)
│   ├── bacillus_env_growth_dataset.csv # Literature-based env growth data
│   ├── predictions.db              # SQLite DB for predictions & retraining
│
└── model/
    └── cfu_model_retrained.pkl     # Saved retrained main model
```

---

### How It Works

1. **Environmental model training (`train_env_model.py`)**

   * Uses literature dataset (`bacillus_env_growth_dataset.csv`)
   * Learns mapping: `(Temp, RH, aw, pH, substrate, initial_cfu) → growth_rate_per_hour`
   * Model: `XGBoostRegressor`

2. **Main model training (`train_retrain_main.py`)**

   * Loads factory dataset (`bacillus1.csv`)
   * Derives features:

     * Process variables: moisture, salt, pH, tank, initial\_cfu, period, month
     * Env-derived features: `env_pred_rate_per_hour`, `env_multiplier_30d`
   * Model: `RandomForestRegressor`

3. **Prediction app (`app.py`)**

   * Input: process conditions
   * Computes predicted CFU (with logistic growth adjustment)
   * Saves predictions into SQLite DB
   * Provides three tabs:

     * **Predict** – run new predictions
     * **History** – view/edit records, update actuals, retrain model
     * **Ontology** – extract rules, visualize errors, show feature importance

4. **Ontology rules (`utils/ontology_rules.py`)**

   * Analyzes prediction history
   * Detects systematic error patterns per condition (e.g., "pH between 5.0–5.5 → model underestimates by 20%")
   * Generates correction factors and interpretable rules

---

### Usage

#### 1. Install

```bash
git clone https://github.com/yourname/doenjang_cfu_predict.git
cd doenjang_cfu_predict
pip install -r requirements.txt
```

#### 2. Train Environmental Model

```bash
python env_model/train_env_model.py --data data/bacillus_env_growth_dataset.csv --out models/env_model_v1.pkl
```

#### 3. Train / Retrain Main Model

```bash
python train_retrain_main.py --main_csv data/bacillus1.csv --env_model models/env_model_v1.pkl --out model/cfu_model_retrained.pkl
```

#### 4. Run App

```bash
streamlit run app.py
```

---

### Example

* Input: Moisture=50%, Salt=12%, pH=6.2, Initial CFU=5000, Period=30 days
* Output: Predicted CFU ≈ 8.3×10³ CFU/g
* Ontology rule: "moisture > 55% → adjust +15%"

---

### Why Reliable?

* **Literature integration**: environmental effects captured from published studies
* **Logistic growth model**: prevents unrealistic exponential predictions
* **Cross-validation & retraining**: reduces overfitting, adapts to factory-specific data
* **Ontology rules**: interpretable, condition-specific corrections

---

### Roadmap

* Phase 4: Dynamic ontology rules (self-updating thresholds)
* Phase 5: Integration with HACCP digital twin systems

---

📄 README (Korean Version)

## 콩된장 Bacillus cereus CFU 예측 시스템

### 개요

이 프로젝트는 \*\*콩된장 숙성 과정에서 Bacillus cereus 균수(CFU)\*\*를 예측하는 머신러닝 기반 시스템입니다.
문헌 기반의 **환경 모델(온도·습도·수분활성도)** 과 공정 변수(수분, 염도, pH, 숙성 기간, 탱크 배치, 초기 균수)를 결합하여, **실제 공장 운영에 활용 가능한 예측 도구**를 제공합니다.

---

### 필요성

* 된장 숙성 과정은 온도·습도·수분 조건에 따라 Bacillus cereus 성장 패턴이 크게 달라집니다.
* 기존의 단순 통계나 엑셀 계산만으로는 복잡한 상호작용을 반영하기 어렵습니다.
* 본 시스템은:

  * **문헌 데이터 기반 환경 효과 반영**
  * **공정 변수 + 실측 균수 데이터 학습**
  * **실측 이력 기반 재학습 및 자동 보정**
    을 통해 신뢰성 있는 CFU 예측을 지원합니다.

---

### 프로젝트 구조

(영문 버전과 동일)

---

### 동작 원리

1. **환경 모델 학습 (`train_env_model.py`)**

   * 문헌 기반 데이터(`bacillus_env_growth_dataset.csv`) 활용
   * `(온도, 습도, 수분활성도, pH, 기질, 초기균수) → 시간당 성장률` 학습
   * 알고리즘: `XGBoost 회귀모델`

2. **메인 모델 학습 (`train_retrain_main.py`)**

   * 공장 배치 데이터(`bacillus1.csv`) 활용
   * 공정 변수 + 환경 파생 변수(`env_pred_rate_per_hour`, `env_multiplier_30d`) 결합
   * 알고리즘: `RandomForest 회귀모델`

3. **앱 실행 (`app.py`)**

   * 예측값 확인 / 이력 관리 / 실제값 업데이트
   * Ontology 탭에서 조건별 예측 오차 규칙 확인 가능
   * Feature Importance 그래프로 모델 해석 제공

4. **Ontology 룰 (`utils/ontology_rules.py`)**

   * 예측 이력 분석
   * 조건별 체계적인 오차 패턴 탐지 (예: "수분 > 55% → 모델 20% 과소예측")
   * 자동 보정 룰 생성 및 적용

---

### 사용 방법

(영문 버전과 동일 – 설치, 학습, 앱 실행)

---

### 예시

* 입력: 수분=50%, 염도=12%, pH=6.2, 초기 CFU=5000, 숙성기간=30일
* 출력: 예측 CFU ≈ 8.3×10³ CFU/g
* Ontology 룰: "수분 > 55% → +15% 보정"

---

### 신뢰성 확보 근거

* 문헌 기반 환경 효과 반영
* 로지스틱 성장식으로 비현실적 폭발 예측 방지
* 교차검증과 이력 기반 재학습 적용
* Ontology 룰을 통한 직관적 해석 및 현장 보정

---

### 향후 계획

* Phase 4: 동적 Ontology 룰 학습
* Phase 5: HACCP 디지털 트윈 시스템과 연계

---
