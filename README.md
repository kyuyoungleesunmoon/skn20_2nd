# 🏦 **은행 고객 이탈 데이터 분석 프로젝트**

## 👥 팀 명 : **1조**

### 👨‍👩‍👧‍👦 **팀원 소개**
| 팀원 1 | 팀원 2 | 팀원 3 | 팀원 4 | 팀원 5 |
|:------:|:------:|:------:|:------:|:------:|
| 😎 김나현 | 🧐 문창교 | 😀 이승규 | 🥰 정래원 | 🤖 이경현 |

---

## 📆 **프로젝트 정보**
- **📅 개발 기간:** 2025.11.04 ~ 2025.11.05  
- **📘 주제:** 머신러닝을 이용한 **은행 고객 이탈률(Churn) 예측 모델 개발**

---

## 💡 **프로젝트 개요**
은행 고객의 **신용 점수, 잔액, 상품 보유 수, 활동 여부** 등의 데이터를 기반으로  
👉 **이탈 가능성이 높은 고객을 사전에 예측**하는 머신러닝 모델을 구축했습니다.  

본 프로젝트는 **금융권의 고객 유지 전략(Customer Retention)** 수립에  
데이터 기반 인사이트를 제공하는 것을 목표로 합니다.

---

## 🚀 **프로젝트 필요성**
> 신규 고객을 유치하는 비용보다 기존 고객을 유지하는 비용이 훨씬 낮습니다.

따라서 **이탈 고객을 조기 탐지하고 대응하는 것**은  
은행의 수익성과 직결되는 핵심 과제입니다.  

이번 프로젝트는 이를 **데이터 기반 예측 모델링으로 해결**하고자 진행되었습니다.



## 🎯 **프로젝트 목표**
✅ 고객 이탈 여부(Churn)를 예측하는 머신러닝 모델 개발  
✅ 주요 특성(Feature)별 이탈 영향 요인 분석  
✅ 데이터 불균형 해결(Class Weight)을 통한 해석 가능성 확보  
✅ 다양한 머신러닝 모델을 통한 최적의 모델 선택  
✅ 결론 및 이유 분석 향후 인사이트 도출출


## 🧩 **데이터 소개**
📂 **출처:** [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/)

| 컬럼명 | 설명 | 타입 |
|:-------------------|:----------------------------------------|:----------:|
| customer_id | 고객 ID | int64 |
| credit_score | 신용 점수 | int64 |
| country | 국가 | float64 |
| gender | 성별 | float64 |
| age | 나이 | int64 |
| tenure | 계좌 보유 연수 | int64 |
| balance | 잔고 | float64 |
| products_number | 금융상품 개수 | int64 |
| credit_card | 신용카드 보유 여부 | int64 |
| active_member | 활동회원 여부 | int64 |
| estimated_salary | 추정 연봉 | float64 |
| churn | 이탈 여부 (Target) | int64 |

---

## 🔍 **EDA (탐색적 데이터 분석)**
📈 변수 간 상관관계 시각화  
<img width="3179" height="3580" alt="churn_visual_summary" src="https://github.com/user-attachments/assets/1e325fb6-e26d-4713-be78-633ce98d19e0" />

---

## ⚙️ **이상치 처리 (Outlier Handling)**

**기준 설정**
- 연속형 변수: IQR 기반 (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
- 범주형 변수: 희귀 카테고리(<1%) 제거

**처리 결과**
| 항목 | 이상치 수 | 비율 |
|:---:|:---:|:---:|
| age | 359 | 3.59% |
| credit_score | 15 | 0.15% |
| products_number=4 | 60 | 0.006% |

➡ `374개(3.74%)` 행 삭제  
➡ products_number의 4번 카테고리는 3번과 통합

---

## 🧮 **전처리 및 학습 준비**
- 범주형 변수 인코딩  
- 연속형 변수 스케일링(StandardScaler)  
- 클래스 불균형 해결: `class_weight='balanced'` 적용  
  (SMOTE도 실험했으나 성능 저하로 미사용)

---

## 🤖 **모델 비교 및 평가**
💡 사용 모델  
`Logistic Regression`, `RandomForest`, `KNN`, `SVC`, `DecisionTree`, `Bagging`, `NN`, `AdaBoost`

📊 **주요 평가 지표**  
- ROC_AUC
- F1-score  
- Accuracy  
- Recall  
📈 모델 성능 비교 및 과적합 검증

- Train과 Test 성능 차이를 통해 과적합 여부 판단
  
📁Test

<img width="605" height="371" alt="테스트 모델평가8" src="https://github.com/user-attachments/assets/9869da34-8e4f-42b5-a261-92e9448c9025" />

📁Train

<img width="586" height="382" alt="트레인 모델평가8" src="https://github.com/user-attachments/assets/2b852e9c-2171-4ee0-9150-2d504f73853d" />

✒️AdaBoost,RandomForest,LogisticRegression 이 가장 안정적인 모델로 판단됨

- 비교 결과 과적합이 적고 최적에 결과를 낸 모델 3개를 선택 후 하이퍼파라메터를 적용후 결과

📂Test

<img width="595" height="222" alt="최종3 테스트" src="https://github.com/user-attachments/assets/1761885b-7cca-4448-a1d9-c0ad3fa84d9b" />

📂Train

<img width="600" height="225" alt="최종3 트레인" src="https://github.com/user-attachments/assets/18ffc21f-3974-4da9-8dd8-d5c6359404ef" />

<img width="695" height="475" alt="output" src="https://github.com/user-attachments/assets/d6b1459e-ca21-4032-a606-9bd40b4aa8e0" />


---

## 🧩 **모델 해석 요약**
🔹 **RandomForest**  
→ Train과 Test의 F1 차이가 적으며, 두 데이터셋 모두에서 높은 점수를 유지 → 균형 잡힌 성능

🔹 **SVM**  
→ 이탈 감지(Recall) 성능 높지만 False Positive 다소 많음  

🔹 **LogisticRegression**  
→ 간결한 모델로 안정적이고 과적합 위험이 가장 낮음

📈 **향후 개선 방향**  
- 파생변수 추가를 통한 고객 행동 패턴 심화 분석 추진.
- ENN, ADASYN 등 고급 샘플링 기법으로 데이터 불균형 개선.
- LightGBM / TabNet / PyTorch 기반 모델로 예측력 향상 실험.
- Optuna 등을 활용한 하이퍼파라미터 자동 최적화 적용.
- 장기적으로 LSTM 기반 시계열 분석으로 고객 변화 추세 반영 예정.

---

## 💻 Streamlit 구현 화면 (UI Showcase)
<img width="2850" height="1546" alt="combined_grid_1" src="https://github.com/user-attachments/assets/34ad7634-f1d3-4150-8385-0bab3d330ace" />
<img width="2850" height="1546" alt="combined_grid_2" src="https://github.com/user-attachments/assets/f9aadb59-e498-4a65-aa7a-c8631ab1ff30" />
<img width="2850" height="1546" alt="combined_grid_3" src="https://github.com/user-attachments/assets/ac2ac787-c5b3-464d-8d8b-79c73bba717a" />
<img width="2850" height="1546" alt="combined_grid_4" src="https://github.com/user-attachments/assets/d94893e1-8b54-4c53-8db4-7f4f3064de0a" />
<img width="2850" height="1546" alt="combined_grid_5" src="https://github.com/user-attachments/assets/54a783ad-6028-46fe-835e-7cd97aa3f3fc" />

## 📁 프로젝트 파일 구조
```
project/
│
├── final_model_pkl/                  # 최종 학습된 모델 파일들
│   ├── adaboost_model.pickl
│   ├── adaboost_model.pkl
│   ├── logisticregression_model.pickl
│   ├── logisticregression_model.pkl
│   ├── randomforest_model.pickl
│   └── randomforest_model.pkl
│
├── streamlit/                        # Streamlit 앱 관련 파일
│   ├── Bank Customer Churn Prediction...
│   ├── adaboost_model.pkl
│   ├── logisticregression_model.pkl
│   ├── randomforest_model.pkl
│   └── customer_churn_app.py
│
├── 산출물/                            # 프로젝트 결과물
│   ├── 모델 학습 결과서(1조).pdf
│   ├── 인공지능 데이터 전처리 결과서(1조).pdf
│   └── 학습된 인공지능 모델 결과서(1조).pdf
│
├── 성능 사진/                         # 모델 성능 비교 이미지
│   ├── after_tuning_test.png
│   ├── after_tuning_train.png
│   ├── base_test.png
│   ├── base_train.png
│   ├── Bank Customer Churn Prediction....png
│──── README.md
│──── churn_cleaned.csv
│──── modeling_final.ipynb
│──── requirements.txt
```

## 🏁 프로젝트 주요 결과 요약

1️⃣ 데이터 분석 결과, 고객의 활동 여부·잔액(balance)·신용점수(credit_score) 가
이탈률(Churn)에 가장 큰 영향을 미치는 핵심 변수로 확인됨.  
2️⃣ 데이터 불균형(class imbalance) 문제를 해결하기 위해
class_weight='balanced' 전략을 적용했으며, SMOTE는 오히려 성능 저하로 미사용.  
3️⃣ 총 8개 머신러닝 모델(Logistic Regression, RandomForest, SVM, AdaBoost 등)을 비교한 결과
RandomForest, AdaBoost, LogisticRegression 3개 모델이 과적합이 적고 안정적인 성능을 보임.  
4️⃣ RandomForest는 F1=0.65, Recall=0.65 수준으로
예측력과 일반화 성능이 가장 뛰어나 최적 모델로 선정됨.  
5️⃣  최종적으로 본 프로젝트는 은행 고객 이탈 가능성 상위 그룹을 식별하여  
선제적 리텐션(retention) 전략 수립 및 맞춤형 마케팅 활동에 활용할 수 있는 예측 모델을 구축함.

---

## 💡 인사이트 
- 고객의 활동 여부·잔액·신용 점수가 이탈에 가장 큰 영향을 주는 주요 변수로 확인됨.
- balance=0 고객군은 오히려 이탈률이 낮아, 단순 보유형 고객의 안정성이 드러남.
- RandomForest는 예측력과 안정성이 모두 높아 대표 모델로 선정됨.
- AdaBoost는 과적합이 적고, 안정적인 일반화 성능을 보여 실무 적용성이 높음.
- LogisticRegression은 단순 구조로 해석력이 뛰어나 인사이트 도출에 용이함.


---

## 🧰 **기술 스택**

| 분야 | 사용 기술 |
|:------|:------|
| 🐍 **언어** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white) |
| 🧮 **데이터 분석** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white) ![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white) |
| 📊 **시각화** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-43B6C7?style=for-the-badge&logo=seaborn&logoColor=white) |
| 🤖 **모델링** | ![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-1E90FF?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Random Forest](https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge&logo=scikitlearn&logoColor=white) ![KNN](https://img.shields.io/badge/KNN-20B2AA?style=for-the-badge&logo=scikitlearn&logoColor=white) ![SVM](https://img.shields.io/badge/SVM-800080?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Decision Tree](https://img.shields.io/badge/Decision%20Tree-FFD700?style=for-the-badge&logo=scikitlearn&logoColor=black) ![Bagging](https://img.shields.io/badge/Bagging-708090?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Neural Network](https://img.shields.io/badge/Neural%20Network-FF1493?style=for-the-badge&logo=pytorch&logoColor=white) ![AdaBoost](https://img.shields.io/badge/AdaBoost-F5B041?style=for-the-badge&logo=scikitlearn&logoColor=black) |
| 💻 **화면 구현** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) |

💬 **팀원 한줄 평**
- 😎 **김나현** : “프로젝트를 진행하며 수많은 질문이 생겼고, 그 답을 찾아가는 과정에서 많은 걸 배웠다. 특히 모델의 성능을 개선하기 위해 여러 접근을 시도하는 과정에서 데이터분석과 머신러닝의 개념을 깊이 이해할 수 있었다. 무엇보다 막혔을 때 함께 고민하고 해결해나간 팀원분들께 감사하고, 3일 동안 고생 많으셨다는 말을 전하고 싶다.”  
- 🧐 **문창교** : “프로젝트를 진행하며 알고 있던 것들도 너무 과신하지 말고 다시 한번 확인해야 한다는 것을 깨달았습니다. 모델의 성능을 개선하려고 팀원 분들과 함께 많은 노력을 기울였는데 고생한 1팀 모두 너무 수고하셨습니다. 다들 감사합니다.”  
- 😀 **이승규** : “처음엔 좋은 모델을 고르는 것이 머신러닝 성능의 전부라고 믿었습니다. 하지만 이탈률 분석을 직접 해보니, 성능을 좌우하는 것은 모델이 아니라 꼼꼼한 데이터 전처리 과정이었습니다. 결국 '좋은 데이터가 좋은 모델을 만든다'는 기본 원칙을 이번에야말로 체감하게 되었습니다.”  
- 🥰 **정래원** : “처음에는 머신러닝의 정확도를 높이기 위해 어떤 모델을 사용하는지가 가장 중요하다고 생각했습니다.하지만 이번 이탈률 분석 실습을 진행하면서, 모델보다도 데이터 전처리와 정규화 과정이 성능에 훨씬 더 큰 영향을 미친다는 점을 깨달았습니다 이번 경험을 통해 데이터 품질이 모델의 성능을 결정한다는 것을 실감했습니다”  
- 🤖 **이경현** : “1차 때와 다르게 많은걸 할 수 있을거라고 생각했지만 비슷했던거 같습니다. 팀원들과 소통을 많이 하려고 노력 하였고 많은 공부가 필요하다고 느꼈습니다. 제가 많은걸 잘하지 못했지만 배려해준 팀원들에게 너무 감사합니다.”



