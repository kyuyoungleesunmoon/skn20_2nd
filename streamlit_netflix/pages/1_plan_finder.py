import streamlit as st
import sys
from pathlib import Path
import time
import numpy as np

# =========================================================================
# 💡 ModuleNotFoundError 해결 로직
# 프로젝트 루트에 sys.path에 추가합니다.
# =========================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 이제 'utils' 폴더를 명시하여 모듈을 불러옵니다.
from streamlit_netflix.utils.styles import apply_netflix_style
from streamlit_netflix.utils.preprocessing_utils import (
    GENDER_MAP, SUBSCRIPTION_MAP, REGION_MAP, 
    DEVICE_MAP, FAVORITE_GENRE_MAP, prepare_model_input # 한글 맵핑을 불러옵니다.
)
from streamlit_netflix.utils.model_utils import load_models, get_churn_prediction, get_recommended_plan 

# 페이지 설정 및 스타일 적용
st.set_page_config(
    page_title="고객 이탈 예측 및 구독 추천",
    page_icon="🎯",
    layout="wide"
)
apply_netflix_style()
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# =========================================================================
# 1. 모델 및 데이터 로드 (Streamlit 캐싱을 사용하여 한 번만 로드)
# =========================================================================
adaboost_model, rf_proba_model, tree_data = load_models()

# 모델 로드에 실패했다면 이후 로직을 실행하지 않음
if adaboost_model is None or rf_proba_model is None or tree_data is None:
    st.error("⚠️ 모델 로드 오류로 인해 페이지 기능을 사용할 수 없습니다. `streamlit_netflix/utils/model_utils.py`의 경로를 확인해주세요.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# =========================================================================
# 2. 메인 페이지 UI
# =========================================================================

st.markdown('<h1 style="font-size: 3rem;">🎯 맞춤 플랜 찾기</h1>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.4rem; color: #333333;">AI 분석 시스템이  <b>고객 이탈 위험도</b>를 예측하고, <b>최적의 구독 타입</b>을 추천해드립니다.</div>', unsafe_allow_html=True)

st.markdown("---")

# =========================================================================
# 3. 입력 폼 (모든 카테고리 필드 한글 UI 적용)
# =========================================================================
with st.form(key='churn_prediction_form'):
    st.markdown('<h3 class="input-header" style="font-size: 2.2rem;">고객 정보 입력</h3>', unsafe_allow_html=True)
    
    # 1행: 나이, 평균 시청 시간 (number_input)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("나이", min_value=1, max_value=120, value=30, step=1)
    with col2:
        watch_hours = st.number_input("월 평균 시청 시간", min_value=0.0, max_value=600.0, value=7.0, step=0.1, format="%.1f")

    # 2행: 최종 로그인 경과일, 프로필 수 (number_input)
    col3, col4 = st.columns(2)
    with col3:
        last_login_days = st.number_input("최종 로그인 경과일", min_value=0, max_value=300, value=7, step=1)
    with col4:
        number_of_profiles = st.number_input("사용하는 프로필 수", min_value=1, max_value=5, value=1, step=1)
        
    # 3행: 성별, 현재 구독 타입
    col5, col6 = st.columns(2)
    with col5:
        # 한글 레이블을 옵션으로 사용
        gender_kr = st.selectbox("성별", options=list(GENDER_MAP.keys())) 
    with col6:
        # 한글 레이블을 옵션으로 사용
        subscription_type_kr = st.selectbox("현재 구독 타입", options=list(SUBSCRIPTION_MAP.keys()))

    # 4행: 지역, 디바이스, 선호 장르 
    col7, col8, col9 = st.columns(3)
    with col7:
        # 한글 레이블을 옵션으로 사용
        region_kr = st.selectbox("지역", options=list(REGION_MAP.keys()))
    with col8:
        # 한글 레이블을 옵션으로 사용
        device_kr = st.selectbox("주요 접속 디바이스", options=list(DEVICE_MAP.keys()))
    with col9:
        # 한글 레이블을 옵션으로 사용
        favorite_genre_kr = st.selectbox("선호 장르", options=list(FAVORITE_GENRE_MAP.keys()))
    
    st.markdown("---")
    
    # 버튼
    submitted = st.form_submit_button("🔥 이탈 위험도 예측 및 최적 플랜 찾기")


# =========================================================================
# 4. 결과 출력
# =========================================================================
if submitted:
    # 1. 사용자 입력 데이터 준비 및 한글 -> 영문 변환 (핵심 수정 부분)
    user_data = {
        'age': age,
        'watch_hours': watch_hours,
        'last_login_days': last_login_days,
        'number_of_profiles': number_of_profiles,
        
        # 한글 선택 값을 영문 값으로 변환하여 모델 입력용 딕셔너리 생성
        'gender': GENDER_MAP[gender_kr], 
        'subscription_type': SUBSCRIPTION_MAP[subscription_type_kr], 
        'region': REGION_MAP[region_kr],
        'device': DEVICE_MAP[device_kr],
        'favorite_genre': FAVORITE_GENRE_MAP[favorite_genre_kr]
    }
    
    # 2. 모델 입력 배열 생성 
    try:
        # prepare_model_input은 영문 레이블(Female, Basic 등)을 받아서 숫자로 인코딩합니다.
        input_array = prepare_model_input(user_data) 
    except Exception as e:
        st.error(f"데이터 전처리 중 오류가 발생했습니다: {e}")
        st.stop()


    # 3. 예측 및 추천 실행
    with st.spinner('모델 예측 및 추천 알고리즘 실행 중...'):
        time.sleep(0.5) # UX를 위한 딜레이
        
        # 3-1. 이탈 예측 및 확률 계산
        churned_result, churn_proba = get_churn_prediction(input_array, adaboost_model, rf_proba_model)
        
        # 3-2. 구독 플랜 추천 (K-NN 기반)
        recommended_plan = get_recommended_plan(input_array, tree_data)
        
    st.success('✅ 예측 완료! 아래 결과를 확인해주세요.')
    st.markdown("---")

    # 4. 결과 시각화 
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown('<h3 style="color:#1a1a1a;">이탈 위험도 분석 결과</h3>', unsafe_allow_html=True)
        
        if churned_result == 1:
            churn_text = "❌ 높음 (이탈 위험)"
            churn_color = "#e50914" # Netflix Red
        else:
            churn_text = "🟢 낮음 (잔류 예상)"
            churn_color = "#388E3C" # Green
            
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 5px; background-color: #f7f7f7; border-left: 5px solid {churn_color}; margin-bottom: 20px;">
                <p style="font-size: 1.2rem; margin: 0; font-weight: 600;">
                    고객님의 이탈 예측 결과는 <span style="color: {churn_color}; font-size: 1.5rem; font-weight: 700;">{churn_text}</span> 입니다.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(
                label="이탈 예측 결과", 
                value=churn_text
            )
            
        with col_metric2:
            st.metric(
                label="이탈 확률", 
                value=f"{churn_proba * 100:.2f}%"
            )

    with col_right:
        st.markdown('<h3 style="color:#e50914;">구독 타입 추천</h3>', unsafe_allow_html=True)
        
        plan_prices = { "Basic": "7,000원", "Standard": "13,500원", "Premium": "17,000원" }
        recommended_price = plan_prices.get(recommended_plan, "가격 정보 없음")

        st.markdown(f"""
            <div class="intro-section" style="background: linear-gradient(135deg, #e50914 0%, #b20710 100%); color: white; text-align: center; padding: 3rem 2rem;">
                <h2 style="color: white; font-size: 2.5rem; margin-bottom: 1rem;">✨ 추천 구독 타입</h2>
                <h1 style="color: white; font-size: 4.5rem; margin: 0.5rem 0; font-weight: 700;">{recommended_plan}</h1>
                <p style="color: white; font-size: 2rem; margin-top: 1rem; font-weight: 500;">월 {recommended_price}</p>
            </div>
        """, unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)