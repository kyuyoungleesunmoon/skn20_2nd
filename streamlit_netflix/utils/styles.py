"""
Netflix 테마 스타일 모듈
2025 최신 디자인 트렌드 적용: 라이트 메인 + Netflix 다크 사이드바, 미니멀리즘, 반응형
"""

import streamlit as st


def apply_netflix_style():
    """Netflix 테마 CSS 스타일 적용"""
    st.markdown("""
        <style>
        /* ========== 사이드바 토글 완전 제거 (모든 방법 동원) ========== */
        button[kind="header"],
        button[data-testid="baseButton-header"],
        button[data-testid="collapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="baseButton-header"],
        [data-testid="stSidebarNavSeparator"],
        .css-1dp5vir,
        .css-17eq0hr {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            pointer-events: none !important;
        }
        
        header[data-testid="stHeader"] button[kind="header"] {
            display: none !important;
        }
        
        [data-testid="stSidebarNav"]::before {
            display: none !important;
        }

        /* ========== 전역 스타일 - 라이트 테마 유지 ========== */
        .stApp {
            background: #ffffff;
            color: #1a1a1a;
        }

        /* Streamlit 상단 헤더 */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #e50914 0%, #b20710 100%);
            border-bottom: 3px solid #8b0000;
        }

        /* 메인 컨테이너 */
        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* ========== 사이드바 Netflix 다크 스타일 (고정 너비 + 반응형) ========== */
        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            width: 200px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            transform: none !important;
            transition: none !important;
            background: #000000 !important;
            border-right: 2px solid #e50914;
            box-shadow: 4px 0 20px rgba(229, 9, 20, 0.3);
            padding-top: 0 !important;
        }

        section[data-testid="stSidebar"] > div:first-child {
            width: 200px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            background: transparent !important;
            padding-top: 0 !important;
        }

        /* 사이드바 상단 빨간색 헤더 (메인 헤더 왼쪽 색상과 동일한 단색) */
        section[data-testid="stSidebar"]::before {
            content: '';
            display: block;
            width: 100%;
            height: 3.75rem;
            background: #e50914 !important;
            border-bottom: 3px solid #8b0000 !important;
            position: sticky;
            top: 0;
            z-index: 999;
            box-sizing: border-box;
        }

        /* 사이드바 항상 표시 */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            display: block !important;
            margin-left: 0 !important;
        }

        /* 사이드바 내비게이션 스타일 */
        [data-testid="stSidebarNav"] {
            background: transparent !important;
            padding: 1.5rem 0.5rem !important;
            margin-top: 1rem;
        }

        /* 사이드바 링크 - 흰색 배경 버튼 스타일 */
        [data-testid="stSidebarNav"] a {
            color: #1a1a1a !important;
            font-weight: 500;
            padding: 1rem 1.2rem !important;
            margin: 0.4rem 0.3rem !important;
            border-radius: 8px;
            border-left: 3px solid transparent;
            background: #ffffff !important;
            transition: all 0.3s ease;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            font-size: 1rem;
        }

        /* 메뉴 텍스트 하드코딩 */
        [data-testid="stSidebarNav"] ul li:nth-child(1) a span {
            visibility: hidden;
            position: relative;
        }
        [data-testid="stSidebarNav"] ul li:nth-child(1) a span::before {
            visibility: visible;
            position: absolute;
            left: 0;
            content: "메인";
        }

        [data-testid="stSidebarNav"] ul li:nth-child(2) a span {
            visibility: hidden;
            position: relative;
        }
        [data-testid="stSidebarNav"] ul li:nth-child(2) a span::before {
            visibility: visible;
            position: absolute;
            left: 0;
            content: "맞춤플랜";
        }

        [data-testid="stSidebarNav"] ul li:nth-child(3) a span {
            visibility: hidden;
            position: relative;
        }
        [data-testid="stSidebarNav"] ul li:nth-child(3) a span::before {
            visibility: visible;
            position: absolute;
            left: 0;
            content: "coming soon";
        }

        [data-testid="stSidebarNav"] a:hover {
            background: #ffffff !important;
            border-left: 3px solid #e50914;
            color: #1a1a1a !important;
            transform: translateX(3px);
            box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
        }

        [data-testid="stSidebarNav"] a[aria-selected="true"] {
            background: #ffffff !important;
            border-left: 3px solid #e50914;
            color: #e50914 !important;
            font-weight: 600;
            box-shadow: 0 4px 16px rgba(229, 9, 20, 0.5);
        }

        [data-testid="stSidebarNav"] a[aria-selected="true"]:hover {
            box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
        }

        /* 사이드바 상단 타이틀 영역 제거 (빨간 헤더로 대체) */
        [data-testid="stSidebar"] > div:first-child > div:first-child {
            border-bottom: none;
            padding-bottom: 0;
            margin-bottom: 0;
        }

        /* 사이드바 반응형 디자인 */
        @media (max-width: 1200px) {
            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div:first-child {
                width: 160px !important;
                min-width: 160px !important;
                max-width: 160px !important;
            }
        }

        @media (max-width: 768px) {
            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div:first-child {
                width: 150px !important;
                min-width: 150px !important;
                max-width: 150px !important;
            }
            [data-testid="stSidebarNav"] a {
                font-size: 0.85rem;
                padding: 0.6rem 0.6rem !important;
            }
        }


        /* ========== 히어로 섹션 ========== */
        .hero-section {
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d0d0d 100%);
            border-radius: 16px;
            margin-bottom: 3rem;
            box-shadow: 0 8px 32px rgba(229, 9, 20, 0.2);
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #e50914 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            font-size: 1.3rem;
            color: #e0e0e0;
            font-weight: 300;
            margin-top: 1rem;
        }

        /* ========== 섹션 타이틀 ========== */
        .section-title {
            font-size: 2.2rem;
            font-weight: 600;
            margin: 2.5rem 0 1.5rem 0;
            color: #1a1a1a;
            border-left: 4px solid #e50914;
            padding-left: 1rem;
        }

        /* ========== 카드 스타일 ========== */
        .feature-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .feature-card:hover {
            transform: translateY(-5px);
            border-color: #e50914;
            box-shadow: 0 12px 40px rgba(229, 9, 20, 0.2);
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #e50914 0%, #ff6b6b 100%);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .feature-card:hover::before {
            transform: scaleX(1);
        }

        .card-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1.7rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            color: #1a1a1a;
        }

        .card-description {
            font-size: 1.3rem;
            color: #555555;
            line-height: 1.6;
        }

        .card-inactive {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .card-inactive:hover {
            transform: none;
            border-color: #e0e0e0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        /* ========== 정보 섹션 ========== */
        .intro-section, .info-section, .cta-section {
            background: #f9f9f9;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem 0;
            border-left: 4px solid #e50914;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        .intro-section h2, .info-section h2, .cta-section h2 {
            color: #e50914;
            font-size: 2.2rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .intro-section p, .info-section p, .cta-section p {
            color: #333333;
            font-size: 1.3rem;
            line-height: 1.8;
        }

        .info-section ul {
            list-style: none;
            padding-left: 0;
        }

        .info-section li {
            color: #333333;
            font-size: 1.3rem;
            margin: 0.8rem 0;
            padding-left: 1.5rem;
            position: relative;
        }

        .info-section li::before {
            content: '▸';
            color: #e50914;
            position: absolute;
            left: 0;
            font-weight: bold;
        }

        .info-section strong {
            color: #1a1a1a;
        }

        /* ========== 버튼 스타일 ========== */
        .stButton > button {
            background: linear-gradient(90deg, #e50914 0%, #b20710 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #b20710 0%, #e50914 100%);
            box-shadow: 0 6px 25px rgba(229, 9, 20, 0.5);
            transform: translateY(-2px);
        }

        /* ========== 입력 필드 스타일 ========== */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input {
            background-color: #ffffff;
            color: #1a1a1a;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 0.75rem;
        }

        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #e50914;
            box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.1);
        }

        /* ========== 푸터 ========== */
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 4rem;
            color: #888888;
            border-top: 2px solid #e0e0e0;
        }

        .footer p {
            margin: 0;
            font-size: 0.95rem;
        }

        /* ========== 반응형 디자인 ========== */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }

            .hero-subtitle {
                font-size: 1.1rem;
            }

            .section-title {
                font-size: 1.5rem;
            }

            .main-container {
                padding: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def create_page_card(icon, title, description, page_link, is_active=True):
    """페이지 이동 카드 컴포넌트 생성"""
    card_class = "feature-card" if is_active else "feature-card card-inactive"

    if is_active:
        card_html = f"""
        <a href="/{page_link}" target="_self" style="text-decoration: none; color: inherit;">
            <div class="{card_class}">
                <div class="card-icon">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="card-description">{description}</div>
            </div>
        </a>
        """
    else:
        card_html = f"""
        <div class="{card_class}">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-description">{description}</div>
        </div>
        """

    st.markdown(card_html, unsafe_allow_html=True)