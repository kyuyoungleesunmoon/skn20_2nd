import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from utils.styles import apply_netflix_style, create_page_card

# 페이지 설정
st.set_page_config(
    page_title="Netflix 이탈율 예측 시스템",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Netflix 스타일 적용 (사이드바 포함 모든 스타일)
apply_netflix_style()

# 메인 컨테이너
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 헤더 섹션
st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title" style="font-size: 4rem;">🎬 나갈꺼면 프리미엄 해주시조</h1>
        <p class="hero-subtitle" style="font-size: 1.35rem;">Netflix 이탈율 예측 & 요금제 추천 시스템</p>
    </div>
""", unsafe_allow_html=True)

# 프로젝트 소개
st.markdown("""
    <div class="intro-section">
        <h2>📊 프로젝트 소개</h2>
        <p>
            본 프로젝트는 넷플릭스 사용자의 다양한 데이터를 기반으로 
            <br>머신러닝을 활용하여 고객 이탈 가능성을 예측하는 시스템을 구현하였습니다.
        </p>
    </div>
""", unsafe_allow_html=True)

# 주요 기능 섹션
st.markdown('<h2 class="section-title">&nbsp&nbsp&nbsp&nbsp&nbsp🎯 주요 기능</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    create_page_card(
        icon="🚀",
        title="맞춤 플랜 찾기",
        description="넷플릭스 사용 특징을 입력하시면 AI 분석시스템을 통해 <br>이탈률이 가장 낮은 최적의 요금제를 추천해드립니다.",
        page_link="1_plan_finder",
        is_active=True
    )

with col2:
    create_page_card(
        icon="🔧",
        title="준비중",
        description="추가 기능이 준비 중입니다. 곧 만나보실 수 있습니다!",
        page_link="2_coming_soon",
        is_active=False
    )

# 프로젝트 정보
st.markdown("""
    <div class="info-section">
        <h2>ℹ️ 프로젝트 정보</h2>
        <ul>
            <li><strong>목적:</strong> 머신러닝/딥러닝 학습 복습 프로젝트</li>
            <li><strong>데이터:</strong> Netflix 고객 이탈률 데이터셋</li>
            <li><strong>핵심 기술:</strong> 머신러닝 분류 모델, 이탈률 예측</li>
            <li><strong>주요 목표:</strong> 고객 맞춤형 요금제 추천으로 이탈률 최소화</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# 푸터
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ using Streamlit | Netflix Churn Prediction Project</p>
    </div>
""", unsafe_allow_html=True)
