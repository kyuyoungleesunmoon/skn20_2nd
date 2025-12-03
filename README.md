# 🍿Netflix 고객 이탈률 예측 모델
<img src="images/프로필2.png" width="100%"> 

## 팀명 : 나갈꺼면 프리미엄 해주시조

## 👥 팀원
| <img src="images/무도2.jpeg" width="150"> <br> 김태빈 |  <img src="images/무도1.jpeg" width="150"> <br> 오학성 |  <img src="images/무도6.jpeg" width="150"> <br> 황수현 |  <img src="images/무도4.jpeg" width="150"> <br> 박다정 |  <img src="images/무도5.jpeg" width="200"> <br> 조준상 | <img src="images/무도3.jpeg" width="150"> <br> 이성현 |
|:------:|:------:|:------:|:------:|:------:|:------:|

## 🎬 프로젝트 소개
본 프로젝트는 넷플릭스 사용자의 다양한 데이터를 기반으로 머신러닝을 활용하여 고객 이탈 가능성을 예측하는 시스템을 구현하고 이탈 가능성이 더 적은 구독 서비스를 추천해주는 마케팅 서비스입니다.

## 🎯 프로젝트 목표
- 사용자 데이터 기반 이탈 예측 모델 개발
- 다양한 머신러닝 모델의 성능 비교 분석
- 고객 맞춤형 구독 서비스 추천 시스템 구현

## 🧾 데이터셋 구성
- 사용자 개인정보: 나이, 성별, 지역
- 구독 정보: 구독 유형(Basic, Standard, Premium)
- 사용 패턴: 일일 시청 시간, 마지막 로그인 일수, 프로필 수
- 선호도: 선호 장르 ex).코미디, 액션 등
- 사용 기기: Desktop, Laptop, Mobile, TV, Tablet

----

## 💻 기술 스택

**Language & Libraries**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-EE4C2C?style=for-the-badge)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge)
![seaborn](https://img.shields.io/badge/seaborn-5A9?style=for-the-badge)
![joblib](https://img.shields.io/badge/joblib-4B8BBE?style=for-the-badge)
![tqdm](https://img.shields.io/badge/tqdm-FFD43B?style=for-the-badge)
![openpyxl](https://img.shields.io/badge/openpyxl-4B8BBE?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python-dotenv](https://img.shields.io/badge/python--dotenv-3776AB?style=for-the-badge)
![Pathlib](https://img.shields.io/badge/pathlib-3776AB?style=for-the-badge)
![os-sys](https://img.shields.io/badge/os/sys-333333?style=for-the-badge)
![time](https://img.shields.io/badge/time-999999?style=for-the-badge)

**Environment & Tools**

![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)

----


## ⚙️ 주요 기능
1. 사용자 이탈 예측
   - 개인 정보 입력을 통한 이탈 가능성 예측
   - 이탈 확률 수치화 제공

2. 맞춤형 구독 추천
   - 비이탈 고객 데이터 기반 최적 구독 유형 추천
   - K-최근접 이웃 알고리즘 활용
  
----

## 🧮 데이터 분석

#### 데이터 셋 정보
| 컬럼명 (Column Name) | 데이터 타입 (Data Type) | 한글 의미 (Description) |
|----------------------|-------------------------|---------------------------|
| `customer_id` | int64 | 고객 고유 식별 번호 |
| `age` | int64 | 고객 나이 |
| `gender` | object (string) | 고객 성별 (`Male`, `Female`, `other`) |
| `subscription_type` | object (string) | 구독 유형 (`Basic`, `Standard`, `Premium`) |
| `watch_hours` | float64 | 총 시청 시간(시간 단위) |
| `last_login_days` | int64 | 마지막 로그인 이후 경과 일수 |
| `region` | object (string) | 지역 구분 (예: `North America`, `Asia` 등) |
| `device` | object (string) | 주로 사용하는 기기 유형 (`Mobile`, `TV`, `PC` 등) |
| `monthly_fee` | float64 | 월 구독 요금 (달러 단위) |
| `churned` | int64 | 이탈 여부 (1 = 이탈, 0 = 유지) |
| `payment_method` | object | 결재방식(예: `gift card`, `crypto` 등) |
| `number_of_profiles` | int64 | 계정에 등록된 프로필 수 |
| `avg_watch_time_per_day` | float64 | 하루 평균 시청 시간 |
| `favorite_genre` | object | 장르(예: `action`, `horror` 등) |




<table>
<tr>
  <td align="center" width="50%">
    <img src="capture/1kaggle_data_collection.jpg" width="100%"><br>
    <sub><b>데이터 출처 (Kaggle)</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/2df_결측치확인.jpg" width="30%"><br>
    <sub><b>결측치 확인</b></sub>
  </td>
</tr>

<tr>
  <td align="center" width="50%">
    <img src="capture/2df_info.jpg" width="50%"><br>
    <sub><b>데이터 타입 확인</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/3heatmap_numeric feature.jpg" width="50%"><br>
    <img src="capture/4churn 상관관계 변수 절대값기준.jpg" width="50%"><br>
    <sub><b>데이터 상관관계 및 Churn 변수 분석</b></sub>
  </td>
</tr>
</table>

#### 탐색적 데이터 분석 (EDA)

<table>
<tr>
  <td align="center" width="33%">
    <img src="capture/D1-전체고객이탈비율(churn rate).jpg" width="90%"><br>
    <sub><b>전체 고객 이탈 비율</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/D2-연령대별평균이탈률.jpg" width="100%"><br>
    <sub><b>연령대별 이탈률</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/D3-이탈여부에 따른 시청시간 분포.jpg" width="100%"><br>
    <sub><b>이탈여부에 따른 시청시간 분포</b></sub>
  </td>
</tr>
<tr>
  <td align="center" width="33%">
    <img src="capture/D4-요금제별고객수.jpg" width="100%"><br>
    <sub><b>요금제별 고객수</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/D5-지역별 고객수.jpg" width="100%"><br>
    <sub><b>지역별 고객수</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/D5-지역별 평균 이탈률.jpg" width="100%"><br>
    <sub><b>지역별 평균 이탈률</b></sub>
  </td>
</tr>
</table>

</div>

----

## 🧹 데이터 전처리

<table>
<tr>
  <td align="center" width="50%">
    <img src="capture/5절대값이 가장높은 watch hours 이상치 확인_Z-score.round(2)_충성고객.jpg" width="100%"><br>
    <sub><b>1. 이상치 확인</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/6컬럼제거.jpg" width="100%"><br>
    <sub><b>2. 불필요한 데이터 컬럼 제거</b></sub>
  </td>
</tr>

<tr>
  <td align="center" width="50%">
    <img src="capture/7one-hot진행.jpg" width="100%"><br>
    <img src="capture/9age One-hot.jpg" width="90%"><br>
    <sub><b>3. One-Hot 인코딩 적용</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/8watch_hours _스케일링.jpg" width="70%"><br>
    <img src="capture/10age min_max.jpg" width="70%"><br>
    <sub><b>4. 필요 컬럼 스케일링</b></sub>
  </td>
</tr>

<tr>
  <td align="center" width="50%">
    <img src="capture/11One-hot info.jpg" width="60%"><br>
    <sub><b>5. 최종 전처리 데이터</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/12label encoding_treemodel.jpg" width="70%"><br>
    <sub><b>6. Tree 모델용 Label Encoding</b></sub><br>
    <img src="capture/13tree model data.jpg" width="100%"><br>
    <sub><b>7. Tree 모델용 최종 데이터</b></sub>
  </td>
</tr>
</table>

   
----

## 🤖 모델 구현

<div align="left">

### 🧩 기본 분류 모델 (Age One-Hot vs Min-Max Scale)

<table>
<tr>
  <td align="center" width="33%">
    <img src="capture/15knn(age_onehot)_classification_report.jpg" width="100%"><br>
    <sub><b>KNN (Age One-Hot)</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/16svc(age_onehot)_classification_report.jpg" width="100%"><br>
    <sub><b>SVM (Age One-Hot)</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/17log_reg(age_onehot)_classification_report.jpg" width="100%"><br>
    <sub><b>로지스틱 회귀 (Age One-Hot)</b></sub>
  </td>
</tr>

<tr>
  <td align="center" width="33%">
    <img src="capture/18knn(age_min_max)_classification_report.jpg" width="100%"><br>
    <sub><b>KNN (Min-Max Scaling)</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/19svc(age_min_max)_classification_report.jpg" width="100%"><br>
    <sub><b>SVM (Min-Max Scaling)</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/20log_reg(min_max)_classification_report.jpg" width="100%"><br>
    <sub><b>로지스틱 회귀 (Min-Max Scaling)</b></sub>
  </td>
</tr>
</table>

---

### 🌲 트리 기반 모델

<table>
<tr>
  <td align="center" width="50%">
    <img src="capture/21randomforest_classification_report.jpg" width="50%"><br>
    <sub><b>랜덤 포레스트 (Random Forest)</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/22gradient_classification_report.jpg" width="40%"><br>
    <sub><b>그래디언트 부스팅 (Gradient Boosting)</b></sub>
  </td>
</tr>
</table>

---

### 🧠 앙상블 모델

<table>
<tr>
  <td align="center" width="100%">
    <img src="capture/27앙상블 모델 4개 비교.jpg" width="60%"><br>
    <sub><b>Bagging / AdaBoost / Voting (Hard & Soft) 비교</b></sub>
  </td>
</tr>
</table>

---

### 🕸️ 딥러닝 모델

<table>
<tr align="center">
  <td align="center" width="100%">
    <img src="capture/29CNN딥러닝_classification_report.jpg" width="100%"><br>
    <sub><b>CNN</b></sub>
  </td>
</tr>
</table>

</div>

----

## 🎨 하이퍼 파라메터 튜닝
### AdaBoost
<table>
<tr>
  <td align="center" width="50%">
    <img src="capture/30-1AdaBoost_HyperParameter.jpg" width="80%"><br>
    <sub><b>AdaBoost 하이퍼 파라메터 튜닝</b></sub>
  </td>
  <td align="center" width="50%">
    <img src="capture/30-2AdaBoost_HyperParameter_결과.jpg" width="60%"><br>
    <sub><b>AdaBoost 하이퍼 파라메터 튜닝 결과</b></sub>
  </td>
</tr>
</table>
<table>
<tr>
  <td align="center" width="33%">
    <img src="capture/30-3AdaBoost_HyperParameter_confusion_matrix.jpg" width="60%"><br>
    <sub><b>AdaBoost 혼돈 행렬</b></sub>
  </td>
   
  <td align="center" width="33%">
    <img src="images/Roc.png" width="60%"><br>
    <sub><b>AdaBoost ROC 커브</b></sub>
  </td>
  
  <td align="center" width="33%">
    <img src="capture/30-4AdaBoost_HyperParameter_결과_시각화.jpg" width="60%"><br>
    <sub><b>AdaBoost 하이퍼 파라메터 튜닝 결과 시각화</b></sub>
  </td>
</tr>
</table>

### Soft-Voting
<table>
<tr>
  <td align="center" width="33%">
    <img src="capture/31-1softvoting-randomforest튜닝.jpg" width="90%"><br>
    <sub><b>softvoting-randomforest 튜닝</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/31-2softvoting-gradientboosting.jpg" width="100%"><br>
    <sub><b>softvoting-gradientboosting 튜닝</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/31-3softvoting-logisticReg튜닝.jpg" width="100%"><br>
    <sub><b>softvoting-logisticReg 튜닝</b></sub>
  </td>
</tr>
<tr>
  <td align="center" width="33%">
    <img src="capture/31-4softvoting-튜닝결과.jpg" width="100%"><br>
    <sub><b>softvoting 튜닝 결과</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/31-4softvoting-튜닝결과_시각화.jpg" width="100%"><br>
    <sub><b>softvoting 튜닝 결과 시각화</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/31-4softvoting-confusion_matrix.jpg" width="100%"><br>
    <sub><b>softvoting 혼돈 행렬</b></sub>
  </td>
</tr>
   
</table>

----

## 💯 프로젝트 결과
1. 모델 성능
   - AdaBoost: 가장 우수한 성능
   - Gradient Boosting: Random Forest와 유사한 성능
   - 앙상블 모델: 단일 모델 대비 개선된 예측력
   - CNN: 전통적 머신러닝 모델과 유사한 성능

2. 특징 엔지니어링
   - One-Hot Encoding vs Label Encoding 비교 분석
  
   <img src="capture/28-3RandomForest_Onehot과label비교_분석결과.jpg" width="40%">
   
   - Tree 기반 모델에서 Label Encoding 효율성 입증
   
   <img src="capture/28-9RandomForest_Onehot과label비교_결론_label_encoding_선택.jpg" width="40%">

3. 최종 모델 선택 :
   - Label Encoding 데이터를 활용한 AdaBoost모델

----

## 🌐 Streamlit 활용 시스템 구현
<div align="center">

### Netflix 이탈 예상자 요금제 추천

<table>
<tr>
  <td align="center" width="33%">
    <img src="capture/S1-메인페이지.jpg" width="100%"><br>
    <sub><b>메인화면</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/S2-맞춤구독추천.jpg" width="100%"><br>
    <sub><b>사용자 정보 입력</b></sub>
  </td>
  <td align="center" width="33%">
    <img src="capture/S3-분석결과.jpg" width="100%"><br>
    <sub><b>결과 확인 및 추천 시스템</b></sub>
  </td>
</tr>

</table>

</div>

----

## 💡 기대효과
### 1.개인화된 고객 대응

- 입력된 고객 데이터를 기반으로 이탈 위험과 맞춤 구독 플랜을 동시에 제공.

### 2.고객 맞춤형 추천 → 고객 경험 향상 가능.

- 이탈 방지 전략 지원

- 이탈 위험이 높은 고객을 선별하여 프로모션, 할인, 맞춤형 컨텐츠 제공 전략에 활용 가능.

### 3.간단한 데이터로 빠른 예측

- 단순 입력값만으로 예측 → 비전문가도 쉽게 사용할 수 있음.

- ML 모델을 활용한 데이터 기반 의사결정 지원.

### 4.시각적 직관성

- 위험도와 추천 플랜을 컬러/배너로 강조 → 의사결정 속도 향상.

- 확률 수치 제공 → 정량적 근거 확보.

----

## ⚠️ 한계점 및 개선 가능성

### 한계점 

#### 입력 데이터 단순화

- 현재 입력값은 기본적인 개인 정보와 시청 관련 데이터만 포함.

- 실제 고객 이탈에는 시청 패턴 변화, 콘텐츠 선호 변화, 결제 이력, 고객 지원 이용 기록 등이 더 중요.
  
### 개선 가능성

#### 1. 추천 신뢰도 제공

 - K근접 기반 이탈하지 않은 사용자 데이터 추천

 - 추천 근거: 최근접 고객 3~5명 정보, 유사도 점수 표시


#### 2. 복수 옵션 추천

 - Basic, Standard, Premium 중 상위 2~3개의 추천 옵션 제공

 - 가격 대비 혜택 비교 → 사용자 선택 폭 확대

----

---

## 💬 한 줄 회고

> #### 김태빈
> 다양한 전처리 과정을 복습하고 내것으로 만드는 과정이 흥미로웠고 다양한 모델 학습을 하면서 데이터에 따른 적합한 모델이 무엇인지 구분하는 능력을 기를 수 있었습니다.
---

> #### 오학성
> One-Hot과 Label Encoding의 trade-off를 직접 비교하며 Tree 모델의 효율성을 체감했고, 여러 모델을 실험하며 데이터 특성에 따른 최적 모델 선택의 중요성을 깨달았습니다.

---

> #### 황수현
> 데이터에 따라 모델들을 학습시키는 과정을 복습해보고 어떤 모델이 어떤 데이터에 어울리는지 알고 최종 선정된 모델의 하이퍼 파라메터를 튜닝하는 과정에서 여러가지 파라메터들을 써보고 최적의 값을 찾고 배울 수  있는 시간이었습니다.

---

> #### 박다정
> 데이터 전처리와 모델 성능 비교 등 수업에서 배운 개별 내용을 하나의 흐름으로 연결해, 전체적인 그림을 이해할 수 있었습니다.

---

> #### 조준상
> 머신러닝과 딥러닝에 대해서 이해 못하고 막연했던 점들이, 프로젝트를 통해 해소되었습니다.

---

> #### 이성현
> 이번 프로젝트를 통해 팀원들의 도움을 받아 딥러닝과 전처리 과정에 대해서 공부할 수 있게 되어서 팀원들에게 감사드리고, 앞으로도 배운 내용과 프로젝트 경험을 발판삼아 더욱 성장하고 싶습니다.

---
