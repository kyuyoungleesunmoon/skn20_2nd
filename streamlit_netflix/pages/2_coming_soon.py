import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from utils.styles import apply_netflix_style

# 페이지 설정
st.set_page_config(
    page_title="준비중 - Netflix",
    page_icon="🔧",
    layout="wide"
)

# Netflix 스타일 적용
apply_netflix_style()

# 메인 컨테이너
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 헤더
st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🔧 준비 중입니다</h1>
        <p class="hero-subtitle">새로운 기능을 준비하고 있습니다</p>
    </div>
""", unsafe_allow_html=True)

# 안내 메시지
st.markdown("""
    <div class="intro-section">
        <h2>🚧 Coming Soon!</h2>
        <p>
            이 페이지는 현재 개발 중입니다.
            곧 유용한 기능들을 만나보실 수 있습니다!
        </p>
    </div>
""", unsafe_allow_html=True)

# 중앙 정렬 메시지
st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 5rem; margin-bottom: 2rem;">⏳</div>
        <h2 style="color: #e50914; font-size: 2rem; margin-bottom: 1rem;">추가 기능 개발 중</h2>
        <p style="color: #b3b3b3; font-size: 1.2rem;">
            더 나은 서비스를 제공하기 위해 열심히 준비하고 있습니다.
        </p>
    </div>
""", unsafe_allow_html=True)

# 제안 사항
st.markdown("""
    <div class="cta-section">
        <h2>💡 제안하고 싶은 기능이 있나요?</h2>
        <p>
            추가되었으면 하는 기능이나 개선사항이 있다면 언제든지 알려주세요!
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 푸터
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ using Streamlit | Netflix Churn Prediction Project</p>
    </div>
""", unsafe_allow_html=True)
