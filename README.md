![header](https://capsule-render.vercel.app/api?type=blur&height=200&color=gradient&text=SKN20-2nd-3TEAM&reversal=false&fontColor=35007f&fontSize=50)

# <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Telegram-Animated-Emojis/main/Animals%20and%20Nature/Lion.webp" alt="Lion" width="25" height="25" /> Team Member
<table>
  <tr>
    <th>강민지</th>
    <th>김지은</th>
    <th>김효빈</th>
    <th>안채연</th>
    <th>홍혜원</th>
  </tr>
    <td><a href="https://github.com/mminguu"><img src="https://img.shields.io/badge/GitHub-mminguu-green?logo=github"></a></td>
    <td><a href="https://github.com/carookim"><img src="https://img.shields.io/badge/GitHub-carookim-yellow?logo=github"></a></td>
    <td><a href="https://github.com/kimobi"><img src="https://img.shields.io/badge/GitHub-kimobi-blue?logo=github"></a></td>
    <td><a href="https://github.com/hochaeyeon"><img src="https://img.shields.io/badge/GitHub-hochaeyeon-lightblue?logo=github"></a></td>
    <td><a href="https://github.com/newhione"><img src="https://img.shields.io/badge/GitHub-newhione-pink?logo=github"></a></td>
  </tr>
</table>
  
---

# 인터넷 고객 이탈(Churn) 분석 및 예측

## 1. Project Overview
### 1.1. 배경
전 세계적으로 인터넷 서비스 제공업체(Internet Service Providers, ISP) 간의 경쟁이 치열해지고 있다.
새로운 고객을 확보하는 것도 중요하지만, 기존 고객을 유지(Retention) 하는 것이
매출 성장과 장기적인 수익성 확보에 더 큰 영향을 미친다.

서비스 품질 저하, 요금 불만, 약정 만료 등으로 인해 고객이 서비스를 해지하는 현상을 “Churn(이탈)” 이라고 하며,
통신사는 이러한 이탈 징후를 조기에 포착하고 대응하는 것이 매우 중요하다.

본 프로젝트는 **“어떤 고객이 서비스를 해지할 가능성이 높은가?”** 라는 문제를 다루며,
고객의 계약 상태, 이용 패턴, 서비스 품질, 요금 수준 등의 데이터를 분석하여
이탈을 유발하는 주요 요인을 파악하고, ISP의 고객 유지 전략(Retention Strategy) 수립에 필요한 인사이트를 제공하는 것을 목표로 한다.
### 1.2. 목표
- 이탈 고객(`churn=1`) 과 유지 고객(`churn=0`) 간의 특성 차이 분석
- 계약 상태(`contract_type`), 서비스 이용 연수(`subscription_age`), 요금 수준(`bill_avg`), 데이터 사용량(`Download/Upload_avg`), 서비스 품질(`service_failure_count`,`download_over_limit`) 등의 요인이
이탈에 미치는 영향 파악
- 통계적 검정 및 시각화를 통해 이탈에 유의미한 변수 도출
- EDA를 통해 도출된 결과를 기반으로 머신러닝 및 딥러닝 수행 및 성능 비교

---
  
## 2. Data Description & Preprocessing

| 구분          | 설명                       |
| ----------- | ------------------------ |
| **총 데이터 수** | 71,892개 고객 데이터           |
| **총 변수 수**  | 11개 변수                 |
| **분석 단위**   | 고객(user_id) 단위           |
| **타깃 변수**   | `churn` (0 = 유지, 1 = 이탈) |

### 2.1. 데이터셋 설명
| 컬럼명                           | 설명                                 | 데이터 타입 | 전처리 내용                                   |
| -------------------------------- | ------------------------------------ | ------------ | --------------------------------------------- |
| `id`                             | 고객 ID                              | int          | 삭제                                          |
| `bill_avg`                       | 최근 3개월 평균 청구 금액(단위: $)   | float        | 로그 변환(`bill_avg_log`) 수행               |
| `service_failure_count`          | 최근 3개월 동안 고객센터 신고 건수    | int          | 이상치 탐지 및 분포 확인                     |
| `download_avg`                   | 최근 3개월 평균 다운로드 사용량(GB)  | float        | 결측치 381행 제거 후, 로그 변환(`download_avg_log`) 수행          |
| `upload_avg`                     | 최근 3개월 평균 업로드 사용량(GB)    | float        | `download_avg`의 결측 행과 동일 구간에서 결측치 발생 -> 자동 제거 후, 로그 변환(`upload_avg_log`) 수행 |
| `download_over_limit`            | 지난 9개월 동안 다운로드 한도 초과 횟수 | int        | 이상치 탐지 및 분포 확인                     |
| `remaining_contract`             | 남은 약정 기간(년 단위, null이면 계약 없음) | float   | 파생 변수 `contract_type` 생성 후 삭제        |
| `subscription_age`               | 서비스 이용 기간(년 단위)            | float        | -0.02 값 제거 후, 파생 변수 `subscription_age_group` 생성 후 삭제 |
| `is_tv_subscriber`               | TV 구독 여부 (1=구독, 0=미구독)      | int          | `subscription_label` 통합 후 삭제             |
| `is_movie_package_subscriber`    | 영화 패키지 구독 여부 (1=구독, 0=미구독) | int       | `subscription_label` 통합 후 삭제             |
| `churn`                          | 이탈 여부 (1=이탈, 0=유지)           | int          | Target 변수                                  |

### 2.2. 파생 변수 설명
| 파생 컬럼명                       | 생성 기준                                                | 값의 의미                                                                                   |
| ---------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **`contract_type`**          | `remaining_contract` 값 기반으로 분류                       | `0: no_contract` → 계약 없음<br>`1: expired` → 계약 만료<br>`2: active` → 현재 약정 유지 중            |
| **`subscription_age_group`** | `subscription_age`를 구간화하여 그룹 생성                      | `0~1년`: 초기 고객<br>`1~3년`: 중기(약정)<br>`3~5년`: 장기(재계약)<br>`5년 이상`: 충성 고객                    |
| **`subscription_label`**     | `is_tv_subscriber`와 `is_movie_package_subscriber` 결합 | `0: none` (구독 없음)<br>`1: tv` (TV만 구독)<br>`2: movie` (영화만 구독)<br>`3: both` (TV+영화 결합 구독) |

본 프로젝트에서는 분석 목적과 모델 구조에 따라 데이터를 세 가지 형태로 구분하여 활용하였다.

_EDA 및 트리 기반 모델_(Decision Tree, Random Forest, XGBoost 등) 에서는
범주형 변수를 구간화·라벨링한 데이터를 사용하였다.
트리 계열 모델은 범주형 피처를 직접 처리할 수 있으므로
`contract_type`, `subscription_age_group`, `subscription_label`을
라벨 인코딩(Label Encoding) 방식으로 변환하였다.

_회귀 및 비트리 기반 모델_(Logistic Regression, SVM 등) 에서는
모델이 연속형 입력에 민감하기 때문에,
`subscription_age`는 연속형 변수로 유지하고,
`contract_type`과 `subscription_label`은 원-핫 인코딩(one-hot encoding) 하였다.

또한, 모델의 안정성과 예측 성능 향상을 위해
`bill_avg`, `download_avg`, `upload_avg`에 대해 로그 변환을 적용한 버전과
변환하지 않은 원본 버전 두 가지를 비교 실험하였다.
이를 통해 로그 변환이 데이터의 분포 왜곡(skewness) 완화와 모델 성능 개선에 미치는 영향을 검증하였다.

---

## 3. EDA
EDA 결과 **고객의 이탈(`churn`)**은 **요금, 서비스 품질, 계약 상태, 구독 조합, 구독 연수**에 의해 명확히 구분되는 패턴을 보였다.

- 요금(`bill_avg`): 낮을수록 이탈률 상승 → 저요금제 고객의 충성도 낮음
- 데이터 사용량(`download_avg`, `upload_avg`): 낮을수록 이탈률 높음 → 사용률 낮은 고객은 서비스 몰입도 낮음
- 서비스 품질(`service_failure_count`, `download_over_limit`): 불만(장애·초과)이 많을수록 `churn` 증가 → 품질 경험이 이탈의 직접 요인
- 계약 상태(`contract_type`): `active` 상태 고객은 유지율 높고, `expired` 고객은 대부분 이탈
- 구독 조합(`subscription_label`): 복합 구독(`both`) 고객의 이탈률이 가장 낮고, 단일 또는 미구독(`none`) 고객은 이탈률 높음
- 구독 연수(`subscription_age_group`): 연수가 길수록 `churn` 감소 → 장기 고객의 충성도 상승
  
특히 **(Active 계약 × 복합 구독 × 장기 이용) 고객이 이탈률이 가장 낮은 핵심 유지 고객군(retention segment) 으로 확인**되었다.

---

## 4. Feature Engineering
- 파생 변수 추가 및 중요 변수 선정
- 로그 변환 변수 활용 여부 결정
- 중요 변수 시각화 (Feature Importance, Correlation Heatmap)

---

## 5. Modeling & Evaluation

### 5.2. 머신러닝 모델 선정 이유 및 적합성

| 모델 | 선정 이유 | 데이터셋과의 적합성 | 주요 장점 | 한계점 | 교차검증 관련 |
|:------:|:------------|:----------------|:-------------|:-------------|:----------------|
| **LightGBM (LGBMClassifier)** | 대용량 데이터(7만 건)에서도 빠른 학습과 높은 정확도를 제공하는 Gradient Boosting 기반 모델 | 범주형 피처 처리 및 로그 변환된 연속형 변수 모두 잘 다룸 | 빠른 학습 속도, 메모리 효율적, Feature Importance 해석 용이 | 매개변수에 따라 과적합 가능성 존재 | 5-Fold 교차검증으로 일반화 성능 검증 및 최적 하이퍼파라미터 탐색 |
| **XGBoost (XGBClassifier)** | LightGBM과 유사하지만 더 정교한 정규화 기능 제공 | 로그 변환으로 분포 완화된 연속형 변수의 복잡한 비선형 관계 포착 | 과적합 방지용 정규화(`lambda`, `alpha`) 내장 | 학습 속도가 다소 느릴 수 있음 | 교차검증을 통해 학습/검증 데이터 간 불균형 완화 |
| **CatBoost (CatBoostClassifier)** | 범주형 변수 자동 처리에 강점, 별도 인코딩 불필요 | contract_type, subscription_label 등의 범주형 변수를 자동 인식 | Label Encoding 불필요, 적은 전처리로 높은 성능 | 파라미터 튜닝이 다소 복잡 | 교차검증으로 파라미터 조합의 일반화 성능 평가 |
| **Random Forest (RandomForestClassifier)** | 직관적이고 안정적인 앙상블 모델, baseline으로 적합 | 이상치가 존재하거나 스케일링이 불필요한 데이터에도 견고 | 과적합에 강하고 변수 중요도 확인 가능 | 매우 큰 데이터에서는 느릴 수 있음 | 각 트리에 대한 평균화로 기본적으로 과적합 완화, 5-Fold로 추가 검증 |
| **Logistic Regression** | 이탈 여부(0/1) 예측에 적합한 선형 분류 모델 | 로그 변환으로 정규분포에 가까워진 bill_avg_log, download_avg_log 등과 궁합이 좋음 | 해석 용이(회귀계수로 영향도 파악 가능), 빠른 학습 | 비선형 관계 파악에는 한계 | 교차검증을 통해 최적의 정규화 강도(`C`) 탐색 및 과적합 방지 |
#### 요약
- 본 프로젝트는 **이탈 예측(churn prediction)** 문제로, 범주형 + 연속형 변수가 혼합된 데이터셋
- 따라서 **트리 기반 모델(LightGBM, XGBoost, CatBoost, Random Forest)** 과 **비트리 기반(Logistic Regression)** 을 병행
- 모든 모델은 **5-Fold 교차검증**을 통해 과적합을 방지하고, 데이터 분할에 따른 모델의 **일반화 성능(Generalization Performance)** 을 확보
  
---

## 6. 모델 개발 및 검증 과정

### 6.1. 기본 모델 학습 (Baseline)
총 6개의 분류 모델(Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost)을 default parameter로 학습하였다.  

| 모델 | Accuracy | F1-score | 비고 |
|:-----------------------:|:-----------:|:-----------:|:-----------------------------------|
| Logistic Regression | 0.9000 | 0.9087 | 스케일링(`StandardScaler`) 적용 |
| Decision Tree | 0.9727 | 0.9755 | 단일 트리 구조, 해석 용이 |
| Random Forest | 0.9702 | 0.9732 | 앙상블 효과로 안정적 성능 |
| **XGBoost** | **0.9842** | **0.9858** | 가장 높은 성능 기록 |
| LightGBM | 0.9716 | 0.9745 | 빠른 학습 속도와 높은 효율성 |
| CatBoost | 0.9480 | 0.9532 | 범주형 데이터 자동 인코딩 강점 |

#### 6.1.1. 상위 3개 모델 교차검증 (Cross Validation)
Baseline 단계에서 성능이 우수했던 **XGBoost, LightGBM, RandomForest** 3개 모델을 대상으로  
일반화 성능을 평가하기 위해 **5-Fold 교차검증**을 수행하였다.

#### 6.1.2. 교차검증 (5-Fold, F1-score 기준)
| 모델 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 평균 F1-score |
|------|---------|---------|---------|---------|-----------|
| LightGBM | 0.7161 | 0.6932 | 0.6618 | 0.5557 | 0.0025 | 0.5259 |
| XGBoost | 0.7161 | 0.6935 | 0.6527 | 0.5748 | 0.0208 | 0.5316 |
| RandomForest | 0.7158 | 0.6912 | 0.6698 | 0.5664 | 0.0002 | 0.5287 |
이는 학습 시 높은 성능(약 0.97~0.98 F1-score)에 비해 크게 낮은 수치로, 과적합된 상태임을 알 수 있음

### 6.2. 동일한 GridSearchCV 파라미터 탐색 범위 적용
트리 기반 모델 4종(**CatBoost, LightGBM, Random Forest, XGBoost**)에 동일한 GridSearch 조건을 적용하여 성능을 공정하게 비교

| 하이퍼파라미터 | 후보값 |
|:----------------|:-------------------|
| `max_depth` | [4, 6, 8] |
| `learning_rate` | [0.01, 0.05, 0.1] |
| `n_estimators` | [300, 500, 800] |

모든 트리 계열 모델에 동일한 탐색 공간을 적용하여, 모델 구조 자체의 차이에 따른 성능 변화를 명확히 비교

#### 6.2.1. 후보값 선정 이유
- **`max_depth` (트리 깊이):**  
  너무 깊으면 과적합, 너무 얕으면 과소적합이 발생하기 때문에  
  **중간 수준(4~8)** 의 깊이를 선택하여 **복잡도와 일반화 성능 간 균형**을 맞추고자 함.

- **`learning_rate` (학습률):**  
  트리 기반 부스팅 모델의 학습 안정성을 위해  
  일반적으로 0.01~0.1 사이의 값이 자주 사용됨.  
  → **과도한 진동 없이 점진적으로 수렴**하도록 세 구간(저·중·고)을 설정.

- **`n_estimators` (트리 개수):**  
  학습률과 상호 보완 관계에 있으므로,  
  **트리 개수를 다양하게 설정(300, 500, 800)** 하여  
  모델의 **학습 충분도**와 **연산 효율성**을 함께 고려함.
  
### 6.3.  GridSearchCV 결과 요약

| 모델 | F1-score | Accuracy | ROC | 하이퍼파라미터 |
|:----------------:|:-----------:|:-----------:|:-----------:|:----------------------------------|
| **LightGBM** | **0.9437** | **0.9384** | **0.9724** | {'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 300} |
| **CatBoost** | 0.9434 | 0.9381 | 0.9720 | {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500} |
| **XGBoost** | 0.9434 | 0.9380 | 0.9716 | {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300} |
| RandomForest | 0.9429 | 0.9372 | 0.9673 | {'max_depth': 8, 'max_features': 'log2', 'n_estimators': 800} |
| LogisticRegression | 0.9385 | 0.9325 | 0.9579 | {'C': 5, 'penalty': 'l2', 'solver': 'lbfgs'} |

### 6.4. 결론
- 모든 모델의 **F1-score가 0.94 이상**으로 전반적으로 우수한 성능을 보임.  
- 그 중 **LightGBM**이 Accuracy(0.9384)와 ROC(0.9724)에서 가장 우수한 결과를 기록.  
- GridSearchCV를 통해 하이퍼파라미터를 최적화한 결과,**과적합이 완화되고 일반화 성능이 향상**된 것을 확인할 수 있었음.

<br>

## 7. 딥러닝모델 선정 이유 및 적합성

| 선정 이유 | 설명 |
|---|---|
| 특징 학습 능력 | 다층 구조를 통해 데이터의 비선형 패턴을 학습 가능 |
| ML 모델 검증 | 머신러닝 모델 대비 성능 비교를 통해 데이터 특성 평가 |
| 프레임워크 경험 | sklearn → PyTorch → TensorFlow 순으로 구현하여 프레임워크별 학습 특성 검증 |
| 학습 안정화 연구 | BatchNorm, Dropout, EarlyStopping 적용하여 모델 일반화 성능 향상 검증 |

| 모델             | F1 Score | ROC-AUC | 요약                                     |
| -------------- | -------- | ------- | -------------------------------------- |
| sklearn MLP    | 0.94    | 0.96   | Baseline 확보                            |
| PyTorch MLP    | 0.94    | 0.96   | BatchNorm + Dropout + EarlyStopping 적용 |
| TensorFlow MLP | 0.94    | 0.96   | 구조 변경 시도해도 안정적 성능 유지                   |

## 7.1. Baseline: sklearn MLPClassifier

- 빠르게 기준 성능 확보, 하이퍼파라미터 최적 성능 확인
- GridSearchCV로 최적 하이퍼파라미터 탐색  
  - Hidden layer size: `(128-64-32)`, `(64-32)`, `(64-32-16)`
  - Activation: `ReLU`, `Tanh`
  - Learning Rate: `0.001`, `0.01`, `0.1`

### 7.1.1. GridSearchCV 결과 (최적 조합)
- `hidden_layer_sizes = (128, 64, 32)`
- `activation = 'relu'`
- `learning_rate_init = 0.001`


## 7.2. PyTorch MLP

- sklearn 최적 조건을 PyTorch로 재현하여 일관성 검증
- Learning rate 변경 실험을 통해 성능 비교 (0.001 / 0.01 / 0.1)

### 7.2.1. 모델 설정
- Layer 구성: `Input → 128 → 64 → 32 → Output`
- Activation: `ReLU`
- BatchNorm + Dropout(0.2) + EarlyStopping 적용
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Batch size: `32`
- Epoch: `100`

### 7.2.2 결과
- LR: `0.001 / 0.01 / 0.1` 비교 → **성능 차이 거의 없음**
- 
## 7.3 TensorFlow MLP

- 프레임워크 변경 시 성능 일관성 검증
- Hidden Layer 구조를 다양하게 변경하여 최적 구조 탐색
- 주요 구성 및 설정
  - Activation: `ReLU`(hidden), `Sigmoid`(output)
  - EarlyStopping 적용
  - Loss: `BinaryCrossentropy`
  - Optimizer: `Adam`
  - Batch size: `16`
  - Epoch: `50`

### 7.3.1 Layer 구조 실험
- 다양한 Layer 형태를 실험했으나 성능 차이는 제한적이었으며,  
- 기본 축소형 구조가 가장 효율적임을 확인.

### 7.3.2. RandomSearch 최적 하이퍼파라미터 조합

- CV: `3`, n_iter: `300`

**Best Params**
- Optimizer: `Adam`
- Layers: `[128, 64, 32]`
- Activation: `Tanh`
- Epoch: `30`
- Learning Rate: `0.001`
- Batch Size: `64`

#### 결론
- 최적화 기법(RandomSearch 포함)을 적용하더라도 기저 성능이 일정 수준 이상이면 성능 개선 폭이 크지 않음을 확인함. 
- 이 문제 설정에서는 복잡한 모델 설계나 고도화된 튜닝보다는, 데이터 전처리와 피처 품질이 성능에 더 큰 영향을 미친다는 점을 확인하였음. 
- MLP 구조를 다양한 프레임워크와 최적화 방법으로 적용하였으나, 성능 변동은 미미했으며 이는 본 문제에서 모델 복잡도보다 데이터 품질과 전처리 과정이 더 중요한 요인임을 의미함
**최종 선정 모델: LightGBM**
- 높은 예측 정확도, 빠른 학습 속도, 안정적인 ROC 성능으로 최종 고객 이탈 예측 모델로 선정함.
