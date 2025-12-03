"""
모델 로딩, 학습 및 예측/추천 함수 모듈
노트북 파일 (netflix_models_check.ipynb)의 모델 및 로직을 사용합니다.
"""

# =========================================================================
# 1. 라이브러리 및 경로 설정
# =========================================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cdist
from pathlib import Path
import os
import joblib
import warnings
import sys
import time # time 모듈 추가: 메시지 확인용 지연시간을 주기 위함
warnings.filterwarnings('ignore')


# 프로젝트 루트 경로 설정 (경로 문제 해결용)
PROJECT_ROOT = Path(__file__).parent.parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 데이터 파일 경로 설정 (최상위 폴더의 'data' 폴더)
DATA_PATH = PROJECT_ROOT / 'data' / 'netflix_customer_churn_tree_preprocessed.csv'

# 모델 파일 경로 설정 (최상위 폴더의 'models/models' 폴더)
MODEL_DIR = PROJECT_ROOT / 'models' / 'models'

# 노트북의 특성 순서 (9개)
FEATURE_COLS = ['age', 'watch_hours', 'last_login_days', 'number_of_profiles',
                'gender_encoded', 'subscription_type_encoded', 'region_encoded',
                'device_encoded', 'favorite_genre_encoded']

# 구독 타입 역방향 매핑
SUBSCRIPTION_INVERSE_MAP = {0: 'Basic', 1: 'Standard', 2: 'Premium'}


# =========================================================================
# 2. 더미 모델 생성 함수 (기존 create_dummy_models.py 파일 통합)
# =========================================================================
def generate_dummy_models(model_dir: Path, status_container: st.delta_generator.DeltaGenerator):
    """
    9개 특성 기반의 더미 모델 파일 (.pth) 2개를 생성하고 저장합니다.
    (status_container를 사용하여 메시지를 출력하고, 완료 후 지울 수 있게 합니다.)
    """
    status_container.info("⏳ 더미 모델 파일 생성 시작...")

    # 모델 저장 디렉토리가 없으면 생성
    os.makedirs(model_dir, exist_ok=True)

    # 더미 데이터 생성 (훈련용)
    np.random.seed(42)
    X_dummy = np.random.rand(100, 9)
    y_dummy = np.random.randint(0, 2, 100)

    # 1. AdaBoost 모델 - adaboost_model.pth
    status_container.text(" - [1/2] AdaBoost 모델 (adaboost_model.pth) 생성 중...")
    from sklearn.tree import DecisionTreeClassifier
    adaboost_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=10,
        random_state=42,
        algorithm='SAMME'
    )
    adaboost_model.fit(X_dummy, y_dummy)
    joblib.dump(adaboost_model, model_dir / 'adaboost_model.pth')

    # 2. RandomForest 모델 - random_forest_model.pth
    status_container.text(" - [2/2] RandomForest 모델 (random_forest_model.pth) 생성 중...")
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_dummy, y_dummy)
    joblib.dump(rf_model, model_dir / 'random_forest_model.pth')

    status_container.success("✅ 9개 특성 기반 더미 모델 파일 생성 완료!")
    # 사용자가 완료 메시지를 볼 수 있도록 잠시 대기
    time.sleep(2)
    # 메시지 컨테이너를 비워서 사라지게 함
    status_container.empty()


# =========================================================================
# 3. 모델 로딩 및 자동 생성 함수
# =========================================================================
@st.cache_resource
def load_models():
    """
    모델 파일을 로드하고, 실패 시 더미 모델을 생성하여 재시도합니다.
    """
    # 메시지 출력을 위한 임시 컨테이너를 만듭니다.
    status_container = st.empty()
    
    adaboost_model = None
    rf_proba_model = None
    tree_data = None
    
    # 1. 데이터 로드 (K-NN 추천용)
    try:
        tree_data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        status_container.error(f"데이터 파일 로드 오류: {DATA_PATH} 경로에 파일이 없습니다.")
        return None, None, None
    except Exception as e:
        status_container.error(f"데이터 파일 처리 중 오류 발생: {e}")
        return None, None, None

    # 2. 모델 로드 시도
    model_paths = {
        'adaboost': MODEL_DIR / 'adaboost_model.pth',
        'rf_proba': MODEL_DIR / 'random_forest_model.pth'
    }

    try:
        # 모델 로드 1차 시도
        adaboost_model = joblib.load(model_paths['adaboost'])
        rf_proba_model = joblib.load(model_paths['rf_proba'])
        
        # 특성 수 확인 (5개 특성 기반 구버전 모델 필터링)
        if adaboost_model.n_features_in_ != 9 or rf_proba_model.n_features_in_ != 9:
             raise ValueError("모델 파일이 9개 특성 기반이 아닙니다. (구 버전 모델)")

        status_container.success("모델 파일 로드 완료 (9개 특성 확인).")
        time.sleep(0.5)
        status_container.empty() # 성공 메시지 사라지게 함
        return adaboost_model, rf_proba_model, tree_data

    except (FileNotFoundError, ValueError, AttributeError) as e:
        status_container.warning(f"⚠️ 모델 로드 오류 ({type(e).__name__}). 더미 모델을 자동 생성합니다.")
        
        # 3. 더미 모델 생성 함수 호출 및 갱신 (status_container 전달)
        try:
            generate_dummy_models(MODEL_DIR, status_container)
            
            # 4. 모델 로드 2차 시도 (재시도)
            adaboost_model = joblib.load(model_paths['adaboost'])
            rf_proba_model = joblib.load(model_paths['rf_proba'])
            
            # 2차 시도 후 최종 검증
            if adaboost_model.n_features_in_ != 9 or rf_proba_model.n_features_in_ != 9:
                 st.error("자동 생성 후에도 모델 특성 수가 일치하지 않습니다. 코드를 확인하십시오.")
                 return None, None, None
                 
            # generate_dummy_models 내부에서 메시지가 지워지므로, 추가적인 지움 로직은 필요 없음.
            return adaboost_model, rf_proba_model, tree_data
            
        except Exception as e:
            status_container.error(f"더미 모델 생성 및 로드 재시도 중 치명적인 오류 발생: {e}")
            return None, None, None


# =========================================================================
# 4. 예측 및 추천 로직 함수
# =========================================================================

def get_churn_prediction(input_array, adaboost_model, rf_proba_model):
    """
    이탈 예측 (AdaBoost) 및 이탈 확률 (RandomForest)을 계산합니다.
    """
    churned_result = adaboost_model.predict(input_array)[0]
    churn_proba = rf_proba_model.predict_proba(input_array)[0, 1]
    
    return churned_result, churn_proba


def get_recommended_plan(input_array, tree_data_df):
    """
    K-NN 기반 추천: 이탈하지 않은 고객 중 가장 유사한 5명의 구독 타입을 추천합니다.
    """
    non_churned_customers = tree_data_df[tree_data_df['churned'] == 0].copy()
    
    new_customer_features = input_array
    non_churned_features = non_churned_customers[FEATURE_COLS].values
    
    K = 5 
    
    if non_churned_features.shape[0] < K:
        return "데이터 부족"
        
    distances = cdist(new_customer_features, non_churned_features, metric='euclidean')[0]
    
    non_churned_customers['distance_to_new_customer'] = distances
    
    nearest_neighbors = non_churned_customers.sort_values(by='distance_to_new_customer').head(K)
    
    recommended_type_encoded = nearest_neighbors['subscription_type_encoded'].mode()
    
    if recommended_type_encoded.empty:
        return "데이터 부족"
        
    recommended_plan_encoded = recommended_type_encoded.iloc[0]
    recommended_plan = SUBSCRIPTION_INVERSE_MAP.get(recommended_plan_encoded, "알 수 없음")
    
    return recommended_plan