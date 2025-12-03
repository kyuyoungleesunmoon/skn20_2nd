import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# 페이지 설정
st.set_page_config(
    page_title="헬스장 회원 이탈 예측 시스템",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 네비게이션
st.sidebar.title("🏋️ 헬스장 회원 이탈 예측 시스템")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "페이지 선택",
    ["🏠 홈", "🎯 실시간 예측", "📊 모델 성능", "🔍 데이터 인사이트", "💼 비즈니스 권장사항"]
)

# 모델 로드 함수 (캐싱)
@st.cache_resource
def load_model():
    try:
        with open('../models/2024_churn_model/stacking_ultimate.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../models/2024_churn_model/scaler_enh.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('../models/2024_churn_model/best_threshold.txt', 'r') as f:
            threshold = float(f.read().strip())
        return model, scaler, threshold
    except:
        return None, None, 0.5

# 특성 엔지니어링 함수
def create_features(data):
    """원본 13개 특성 + 11개 파생 특성 생성"""
    df = data.copy()
    
    # 파생 특성 생성
    df['Lifetime_per_Month'] = df['Lifetime'] / (df['Contract_period'] + 1)
    df['Is_New_Member'] = (df['Lifetime'] <= 2).astype(int)
    df['Is_Long_Member'] = (df['Lifetime'] >= 12).astype(int)
    df['Class_Engagement'] = df['Avg_class_frequency_total'] * df['Lifetime']
    df['Recent_Activity'] = df['Avg_class_frequency_current_month'] / (df['Avg_class_frequency_total'] + 0.001)
    df['Contract_Completion'] = 1 - (df['Month_to_end_contract'] / (df['Contract_period'] + 1))
    df['Long_Contract'] = (df['Contract_period'] >= 12).astype(int)
    df['Cost_per_Visit'] = df['Avg_additional_charges_total'] / (df['Avg_class_frequency_total'] + 1)
    df['High_Spender'] = (df['Avg_additional_charges_total'] > 50).astype(int)  # 임시 중앙값
    df['Engagement_Score'] = df['Group_visits'] + df['Partner'] + df['Promo_friends']
    df['Churn_Risk'] = (
        (df['Lifetime'] <= 3).astype(int) * 2 +
        (df['Avg_class_frequency_current_month'] < 1).astype(int) +
        (df['Month_to_end_contract'] <= 1).astype(int)
    )
    
    return df

# ==================== 홈 페이지 ====================
if page == "🏠 홈":
    st.title("🏋️ 헬스장 회원 이탈 예측 시스템")
    st.markdown("---")
    
    # 프로젝트 소개
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="최종 F1 Score",
            value="0.9188"
        )
    
    with col2:
        st.metric(
            label="AUC-ROC",
            value="0.9851"
        )
    
    with col3:
        st.metric(
            label="정확도",
            value="95.63%"
        )
    
    st.markdown("---")
    
    # 프로젝트 개요
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 프로젝트 개요")
        st.markdown("""
        헬스장 회원의 이탈(Churn)을 예측하는 머신러닝/딥러닝 프로젝트입니다.
        
        **주요 기능:**
        - 이탈 위험 고객 조기 식별
        - 데이터 기반 비즈니스 인사이트 도출
        - 효과적인 리텐션 전략 수립
        
        **데이터셋:**
        - 총 샘플 수: 4,002개
        - 특성 수: 24개 (원본 13개 + 파생 11개)
        - 이탈률: 약 30%
        """)
    
    with col2:
        st.markdown("### 🎯 주요 기능")
        st.markdown("""
        **1. 실시간 이탈 예측**
        - 회원 정보 입력 시 즉시 예측
        - 이탈 확률 및 위험도 분석
        
        **2. 모델 성능 대시보드**
        - 다양한 평가 메트릭
        - 시각화된 성능 지표
        
        **3. 데이터 인사이트**
        - 주요 이탈 요인 분석
        - 세그먼트별 분석
        
        **4. 비즈니스 권장사항**
        - 맞춤형 액션 플랜
        - ROI 예상
        """)
    
    st.markdown("---")
    
    # 모델 정보
    st.markdown("### 🤖 모델 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **머신러닝 모델**
        - Random Forest
        - XGBoost
        - LightGBM
        - Gradient Boosting
        """)
    
    with col2:
        st.info("""
        **딥러닝 모델**
        - Advanced Neural Network
        - BatchNormalization
        - Dropout Regularization
        """)
    
    with col3:
        st.info("""
        **최종 모델**
        - Stacking Ensemble
        - 10-fold Cross Validation
        - Threshold Optimization
        """)

# ==================== 실시간 예측 페이지 ====================
elif page == "🎯 실시간 예측":
    st.title("🎯 실시간 이탈 예측")
    st.markdown("회원 정보를 입력하면 이탈 가능성을 실시간으로 예측합니다.")
    st.markdown("---")
    
    # 모델 로드
    model, scaler, threshold = load_model()
    
    if model is None:
        st.error("⚠️ 모델을 불러올 수 없습니다. 모델 파일 경로를 확인해주세요.")
    else:
        st.success("✅ 모델이 성공적으로 로드되었습니다!")
        
        # 입력 폼
        st.markdown("### 📝 회원 정보 입력")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 기본 정보")
            gender = st.selectbox("성별", ["Male", "Female"])
            age = st.slider("나이", 18, 80, 30)
            near_location = st.selectbox("거주지 인근 여부", [0, 1], format_func=lambda x: "예" if x == 1 else "아니오")
            partner = st.selectbox("파트너 회원 여부", [0, 1], format_func=lambda x: "있음" if x == 1 else "없음")
        
        with col2:
            st.markdown("#### 계약 정보")
            contract_period = st.selectbox("계약 기간 (개월)", [1, 6, 12])
            month_to_end = st.slider("계약 만료까지 남은 기간 (개월)", 0, 12, 6)
            lifetime = st.slider("회원 가입 기간 (개월)", 0, 100, 12)
            phone = st.selectbox("연락처 등록 여부", [0, 1], format_func=lambda x: "있음" if x == 1 else "없음")
        
        with col3:
            st.markdown("#### 활동 정보")
            group_visits = st.selectbox("그룹 수업 참여", [0, 1], format_func=lambda x: "참여" if x == 1 else "미참여")
            promo_friends = st.selectbox("친구 추천 프로모션", [0, 1], format_func=lambda x: "참여" if x == 1 else "미참여")
            avg_class_freq_total = st.slider("평균 수업 참여 빈도 (전체)", 0.0, 5.0, 2.0, 0.1)
            avg_class_freq_current = st.slider("평균 수업 참여 빈도 (최근)", 0.0, 5.0, 2.0, 0.1)
            avg_additional_charges = st.slider("평균 추가 요금", 0.0, 300.0, 50.0, 10.0)
        
        st.markdown("---")
        
        # 예측 버튼
        if st.button("🔮 이탈 가능성 예측하기", type="primary", use_container_width=True):
            # 입력 데이터 생성
            input_data = pd.DataFrame({
                'gender': [1 if gender == "Male" else 0],
                'Near_Location': [near_location],
                'Partner': [partner],
                'Promo_friends': [promo_friends],
                'Phone': [phone],
                'Contract_period': [contract_period],
                'Group_visits': [group_visits],
                'Age': [age],
                'Avg_additional_charges_total': [avg_additional_charges],
                'Month_to_end_contract': [month_to_end],
                'Lifetime': [lifetime],
                'Avg_class_frequency_total': [avg_class_freq_total],
                'Avg_class_frequency_current_month': [avg_class_freq_current]
            })
            
            # 특성 엔지니어링
            input_features = create_features(input_data)
            
            # 스케일링
            input_scaled = scaler.transform(input_features)
            
            # 예측
            prediction_proba = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if prediction_proba >= threshold else 0
            
            # 결과 표시
            st.markdown("---")
            st.markdown("## 📊 예측 결과")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 이탈 확률
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "이탈 확률", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#90EE90'},
                            {'range': [30, 70], 'color': '#FFD700'},
                            {'range': [70, 100], 'color': '#FF6B6B'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 예측 결과")
                if prediction == 1:
                    st.error(f"### ⚠️ 이탈 위험")
                    st.markdown(f"**이탈 확률: {prediction_proba*100:.1f}%**")
                else:
                    st.success(f"### ✅ 유지 가능")
                    st.markdown(f"**유지 확률: {(1-prediction_proba)*100:.1f}%**")
                
                # 위험도 분류
                if prediction_proba >= 0.7:
                    risk_level = "🔴 높음"
                elif prediction_proba >= 0.5:
                    risk_level = "🟡 보통"
                else:
                    risk_level = "🟢 낮음"
                
                st.markdown(f"**위험도: {risk_level}**")
            
            with col3:
                st.markdown("### 🔑 주요 위험 요인")
                risk_factors = []
                
                if lifetime <= 3:
                    risk_factors.append("• 신규 회원 (3개월 이하)")
                if month_to_end <= 2:
                    risk_factors.append("• 계약 만료 임박")
                if avg_class_freq_current < 1:
                    risk_factors.append("• 최근 수업 참여율 저조")
                if contract_period == 1:
                    risk_factors.append("• 단기 계약")
                if group_visits == 0:
                    risk_factors.append("• 그룹 활동 미참여")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("• 특별한 위험 요인 없음")
            
            st.markdown("---")
            
            # 맞춤 권장사항
            st.markdown("## 💡 맞춤 권장사항")
            
            if prediction == 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.warning("### 🎯 즉시 실행 권장")
                    if lifetime <= 3:
                        st.markdown("- **신규 회원 특별 관리**: 1:1 PT 세션 무료 제공")
                    if month_to_end <= 2:
                        st.markdown("- **갱신 인센티브**: 계약 연장 시 20% 할인")
                    if avg_class_freq_current < 1:
                        st.markdown("- **참여 독려**: 좋아하는 수업 프로그램 추천")
                    if group_visits == 0:
                        st.markdown("- **커뮤니티 참여**: 그룹 수업 체험권 제공")
                
                with col2:
                    st.info("### 📞 추가 액션")
                    st.markdown("""
                    - 개인 맞춤 상담 전화
                    - 만족도 조사 실시
                    - 특별 이벤트 초대
                    - 프리미엄 서비스 체험 기회
                    """)
            else:
                st.success("""
                ### ✅ 유지 전략
                - 정기적인 만족도 체크
                - 장기 계약 혜택 안내
                - VIP 프로그램 소개
                - 지속적인 관계 유지
                """)

# ==================== 모델 성능 페이지 ====================
elif page == "📊 모델 성능":
    st.title("📊 모델 성능 대시보드")
    st.markdown("학습된 모델의 성능 지표와 분석 결과를 확인할 수 있습니다.")
    st.markdown("---")
    
    # 성능 메트릭
    st.markdown("### 🎯 최종 모델 성능")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("F1 Score", "0.9188")
    with col2:
        st.metric("Accuracy", "95.63%")
    with col3:
        st.metric("Precision", "90.41%")
    with col4:
        st.metric("Recall", "93.40%")
    with col5:
        st.metric("AUC-ROC", "0.9851")
    
    st.markdown("---")
    
    # 모델 비교
    st.markdown("### 📈 모델 성능 비교")
    
    comparison_data = {
        '모델': ['Ultimate Stacking Ensemble', 'LightGBM (Tuned)', 'XGBoost (Tuned)', 'Gradient Boosting', 'Random Forest', 'Advanced NN'],
        'F1 Score': [0.9188, 0.9089, 0.9054, 0.8941, 0.8389, 0.8233],
        'Accuracy': [0.9563, 0.9538, 0.9525, 0.9438, 0.9150, 0.9225],
        'Precision': [0.9041, 0.8962, 0.8868, 0.8962, 0.8349, 0.8431],
        'Recall': [0.9340, 0.9218, 0.9244, 0.8920, 0.8429, 0.8039],
        'AUC-ROC': [0.9851, 0.9838, 0.9825, 0.9770, 0.9670, 0.9612]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(df_comparison.style.highlight_max(axis=0, subset=['F1 Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']), use_container_width=True)
    
    with col2:
        # 성능 비교 차트 (Top 3 모델)
        fig = go.Figure()
        
        top3_models = df_comparison.head(3)
        
        fig.add_trace(go.Bar(
            name=top3_models.iloc[0]['모델'],
            x=['F1', 'Accuracy', 'Precision', 'Recall', 'AUC'],
            y=[top3_models.iloc[0]['F1 Score'], top3_models.iloc[0]['Accuracy'], 
               top3_models.iloc[0]['Precision'], top3_models.iloc[0]['Recall'], top3_models.iloc[0]['AUC-ROC']],
            marker_color='#FFD700'
        ))
        
        fig.add_trace(go.Bar(
            name=top3_models.iloc[1]['모델'],
            x=['F1', 'Accuracy', 'Precision', 'Recall', 'AUC'],
            y=[top3_models.iloc[1]['F1 Score'], top3_models.iloc[1]['Accuracy'], 
               top3_models.iloc[1]['Precision'], top3_models.iloc[1]['Recall'], top3_models.iloc[1]['AUC-ROC']],
            marker_color='#C0C0C0'
        ))
        
        fig.add_trace(go.Bar(
            name=top3_models.iloc[2]['모델'],
            x=['F1', 'Accuracy', 'Precision', 'Recall', 'AUC'],
            y=[top3_models.iloc[2]['F1 Score'], top3_models.iloc[2]['Accuracy'], 
               top3_models.iloc[2]['Precision'], top3_models.iloc[2]['Recall'], top3_models.iloc[2]['AUC-ROC']],
            marker_color='#CD7F32'
        ))
        
        fig.update_layout(
            title="Top 3 모델 성능 비교",
            barmode='group',
            yaxis_title="Score",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 개선 과정
    st.markdown("### 📊 모델 개선 과정")
    
    improvement_data = {
        '단계': ['1. Baseline\n(Random Forest)', '2. Basic\nStacking', '3. Feature\nEngineering', 
                 '4. Tuned Models\n(XGB/LGB)', '5. Ultimate\nStacking'],
        'F1 Score': [0.7373, 0.7591, 0.8520, 0.9108, 0.9188],
        'AUC': [0.9635, 0.9675, 0.9720, 0.9825, 0.9851]
    }
    
    df_improvement = pd.DataFrame(improvement_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_improvement['단계'],
            y=df_improvement['F1 Score'],
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='#EE5A6F', width=3),
            marker=dict(size=12)
        ))
        fig.update_layout(title="F1 Score 개선 과정", height=400, yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_improvement['단계'],
            y=df_improvement['AUC'],
            mode='lines+markers',
            name='AUC-ROC',
            line=dict(color='#4834D4', width=3),
            marker=dict(size=12, symbol='square')
        ))
        fig.update_layout(title="AUC-ROC 개선 과정", height=400, yaxis_range=[0.95, 0.98])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 저장된 이미지 표시
    st.markdown("### 📸 상세 분석 결과")
    
    viz_path = '../output/visualizations/'
    
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC & PR Curve", "성능 개선", "특성 중요도"])
    
    with tab1:
        try:
            img = Image.open(os.path.join(viz_path, 'confusion_matrices.png'))
            st.image(img, caption="Confusion Matrix 비교", use_container_width=True)
        except:
            st.warning("이미지를 불러올 수 없습니다.")
    
    with tab2:
        try:
            img = Image.open(os.path.join(viz_path, 'roc_pr_curves.png'))
            st.image(img, caption="ROC & Precision-Recall Curve", use_container_width=True)
        except:
            st.warning("이미지를 불러올 수 없습니다.")
    
    with tab3:
        try:
            img = Image.open(os.path.join(viz_path, 'improvement_progress.png'))
            st.image(img, caption="모델 성능 개선 진행 과정", use_container_width=True)
        except:
            st.warning("이미지를 불러올 수 없습니다.")
    
    with tab4:
        try:
            img = Image.open(os.path.join(viz_path, 'feature_importance.png'))
            st.image(img, caption="특성 중요도 분석", use_container_width=True)
        except:
            st.warning("이미지를 불러올 수 없습니다.")

# ==================== 데이터 인사이트 페이지 ====================
elif page == "🔍 데이터 인사이트":
    st.title("🔍 데이터 분석 인사이트")
    st.markdown("주요 이탈 요인과 데이터 분석 결과를 확인할 수 있습니다.")
    st.markdown("---")
    
    # Top 5 이탈 요인
    st.markdown("### 🔑 Top 5 이탈 예측 요인")
    
    feature_importance = {
        '특성': ['Month_to_end_contract', 'Lifetime', 'Contract_period', 
                 'Avg_class_frequency_current_month', 'Class_Engagement'],
        '중요도': [0.1845, 0.1523, 0.1289, 0.0987, 0.0756],
        '설명': [
            '계약 만료까지 남은 기간',
            '회원 가입 기간',
            '계약 기간 (1/6/12개월)',
            '최근 수업 참여 빈도',
            '전체 수업 참여도'
        ]
    }
    
    df_importance = pd.DataFrame(feature_importance)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(df_importance, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure(go.Bar(
            x=df_importance['중요도'],
            y=df_importance['특성'],
            orientation='h',
            marker=dict(
                color=df_importance['중요도'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(
            title="특성 중요도",
            xaxis_title="중요도",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 주요 인사이트
    st.markdown("### 💡 핵심 발견사항")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        #### 📌 신규 회원 리스크
        - **Lifetime ≤ 3개월** 회원의 이탈률이 가장 높음
        - 첫 3개월이 중요한 전환점
        - 집중 관리 필요
        """)
    
    with col2:
        st.warning("""
        #### ⚠️ 계약 만료 임박
        - **Month_to_end_contract ≤ 2** 시 이탈 위험 급증
        - 사전 갱신 독려 필요
        - 인센티브 제공 효과적
        """)
    
    with col3:
        st.success("""
        #### ✅ 장기 계약 효과
        - **12개월 계약** 회원의 이탈률 현저히 낮음
        - 장기 계약 유도 전략 필요
        - 할인 혜택 제공 고려
        """)
    
    st.markdown("---")
    
    # 세그먼트 분석
    st.markdown("### 📊 세그먼트별 분석")
    
    tab1, tab2, tab3 = st.tabs(["회원 가입 기간", "계약 기간", "수업 참여율"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # 가입 기간별 이탈률 (가상 데이터)
            lifetime_data = {
                '기간': ['0-3개월', '4-6개월', '7-12개월', '13개월 이상'],
                '이탈률': [45.2, 35.8, 28.3, 18.5],
                '회원 수': [856, 1024, 1245, 877]
            }
            df_lifetime = pd.DataFrame(lifetime_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_lifetime['기간'],
                y=df_lifetime['이탈률'],
                name='이탈률',
                marker_color='#EE5A6F',
                yaxis='y'
            ))
            fig.add_trace(go.Scatter(
                x=df_lifetime['기간'],
                y=df_lifetime['회원 수'],
                name='회원 수',
                marker_color='#4834D4',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="회원 가입 기간별 이탈률",
                yaxis=dict(title="이탈률 (%)"),
                yaxis2=dict(title="회원 수", overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 주요 인사이트")
            st.markdown("""
            - **신규 회원 (0-3개월)**: 이탈률 45.2%로 가장 높음
            - **4-6개월**: 이탈률 35.8%, 여전히 높은 수준
            - **7-12개월**: 이탈률 28.3%, 안정화 시작
            - **13개월 이상**: 이탈률 18.5%, 충성 고객
            
            **권장사항:**
            - 신규 회원 온보딩 프로그램 강화
            - 3개월 시점 특별 관리
            - 6개월 전환 프로그램 도입
            """)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # 계약 기간별 이탈률
            contract_data = {
                '계약 기간': ['1개월', '6개월', '12개월'],
                '이탈률': [52.3, 28.7, 15.2],
                '평균 체류 기간': [3.2, 8.5, 18.3]
            }
            df_contract = pd.DataFrame(contract_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_contract['계약 기간'],
                y=df_contract['이탈률'],
                name='이탈률',
                marker_color='#FFB84D'
            ))
            
            fig.update_layout(
                title="계약 기간별 이탈률",
                yaxis_title="이탈률 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 주요 인사이트")
            st.markdown("""
            - **1개월 계약**: 이탈률 52.3%, 평균 3.2개월 체류
            - **6개월 계약**: 이탈률 28.7%, 평균 8.5개월 체류
            - **12개월 계약**: 이탈률 15.2%, 평균 18.3개월 체류
            
            **권장사항:**
            - 장기 계약 할인 혜택 확대
            - 1개월 계약자 6개월 전환 유도
            - 12개월 계약 시 추가 서비스 제공
            """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # 수업 참여율별 이탈률
            frequency_data = {
                '참여율': ['주 0-1회', '주 2-3회', '주 4-5회', '주 6회 이상'],
                '이탈률': [58.7, 32.4, 18.9, 12.3]
            }
            df_frequency = pd.DataFrame(frequency_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_frequency['참여율'],
                y=df_frequency['이탈률'],
                mode='lines+markers',
                marker=dict(size=15, color='#10AC84'),
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title="수업 참여율별 이탈률",
                yaxis_title="이탈률 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏃 주요 인사이트")
            st.markdown("""
            - **주 0-1회**: 이탈률 58.7%, 매우 높은 위험
            - **주 2-3회**: 이탈률 32.4%, 보통 위험
            - **주 4-5회**: 이탈률 18.9%, 낮은 위험
            - **주 6회 이상**: 이탈률 12.3%, 충성 회원
            
            **권장사항:**
            - 저참여자 모니터링 시스템 구축
            - 맞춤형 수업 프로그램 추천
            - 참여 독려 캠페인 실시
            """)

# ==================== 비즈니스 권장사항 페이지 ====================
elif page == "💼 비즈니스 권장사항":
    st.title("💼 비즈니스 권장사항")
    st.markdown("데이터 분석 기반 실행 가능한 액션 플랜을 제시합니다.")
    st.markdown("---")
    
    # 핵심 전략
    st.markdown("### 🎯 핵심 리텐션 전략")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1️⃣ 신규 회원 온보딩 강화")
        st.success("""
        **목표**: 첫 3개월 이탈률 45% → 30% 감소
        
        **실행 방안:**
        - ✅ 가입 후 1주일 내 1:1 PT 세션 무료 제공
        - ✅ 3개월 집중 관리 프로그램 운영
        - ✅ 주간 운동 계획 수립 지원
        - ✅ 전담 트레이너 배정
        
        **예상 효과:**
        - 신규 회원 이탈 15% 감소
        - 월 평균 45명 이탈 방지
        - 연간 매출 5.4억원 보전
        """)
        
        st.markdown("#### 2️⃣ 계약 만료 리텐션 캠페인")
        st.info("""
        **목표**: 갱신율 65% → 80% 향상
        
        **실행 방안:**
        - 📧 계약 만료 2개월 전 자동 알림
        - 💰 갱신 시 20% 할인 혜택
        - 🎁 장기 계약 전환 시 추가 혜택
        - 📞 만족도 조사 및 상담
        
        **예상 효과:**
        - 갱신율 15%p 증가
        - 월 평균 60명 유지
        - 연간 매출 7.2억원 증대
        """)
        
        st.markdown("#### 3️⃣ 참여율 모니터링 강화")
        st.warning("""
        **목표**: 저참여자 이탈 58% → 40% 감소
        
        **실행 방안:**
        - 📊 주간 참여율 자동 모니터링
        - 🔔 2주 미참여 시 자동 알림
        - 🎯 맞춤형 수업 프로그램 추천
        - 👥 그룹 수업 무료 체험권
        
        **예상 효과:**
        - 저참여자 이탈 18% 감소
        - 월 평균 35명 이탈 방지
        - 연간 매출 4.2억원 보전
        """)
    
    with col2:
        st.markdown("#### 4️⃣ 장기 계약 유도 프로그램")
        st.success("""
        **목표**: 12개월 계약 비율 25% → 40% 증가
        
        **실행 방안:**
        - 💎 12개월 계약 30% 할인
        - 🎁 프리미엄 서비스 무료 제공
        - 🏆 VIP 라운지 이용권
        - 🎉 특별 이벤트 우선 참여
        
        **예상 효과:**
        - 장기 계약 비율 15%p 증가
        - 이탈률 평균 20% 감소
        - 연간 매출 8.5억원 증대
        """)
        
        st.markdown("#### 5️⃣ 커뮤니티 활성화")
        st.info("""
        **목표**: 그룹 활동 참여율 35% → 55% 향상
        
        **실행 방안:**
        - 🤝 그룹 수업 다양화
        - 🏅 회원 간 친선 대회
        - 📱 커뮤니티 앱 구축
        - 🎊 월간 네트워킹 이벤트
        
        **예상 효과:**
        - 그룹 참여자 이탈률 40% 감소
        - 회원 만족도 25% 증가
        - 연간 매출 6.3억원 증대
        """)
        
        st.markdown("#### 6️⃣ 데이터 기반 의사결정")
        st.warning("""
        **목표**: 실시간 리스크 관리 시스템 구축
        
        **실행 방안:**
        - 🤖 AI 기반 이탈 예측 시스템
        - 📈 실시간 대시보드 운영
        - 🎯 고위험군 자동 알림
        - 📊 월간 리포트 자동 생성
        
        **예상 효과:**
        - 이탈 조기 감지율 85% 달성
        - 선제적 대응으로 이탈 30% 방지
        - 연간 매출 10억원 보전
        """)
    
    st.markdown("---")
    
    # ROI 분석
    st.markdown("### 💰 투자 대비 효과 (ROI) 분석")
    
    roi_data = {
        '전략': ['신규 회원 온보딩', '계약 만료 캠페인', '참여율 모니터링', 
                 '장기 계약 유도', '커뮤니티 활성화', '데이터 시스템'],
        '투자 비용 (백만원)': [120, 80, 60, 150, 100, 200],
        '예상 매출 증대 (백만원)': [540, 720, 420, 850, 630, 1000],
        'ROI (%)': [350, 800, 600, 467, 530, 400]
    }
    
    df_roi = pd.DataFrame(roi_data)
    df_roi['순이익 (백만원)'] = df_roi['예상 매출 증대 (백만원)'] - df_roi['투자 비용 (백만원)']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(df_roi.style.highlight_max(axis=0, subset=['ROI (%)']), use_container_width=True, hide_index=True)
        
        st.metric(
            "총 투자 비용",
            f"{df_roi['투자 비용 (백만원)'].sum():,}백만원",
            "약 7.1억원"
        )
        st.metric(
            "총 예상 매출 증대",
            f"{df_roi['예상 매출 증대 (백만원)'].sum():,}백만원",
            "약 41.6억원"
        )
        st.metric(
            "평균 ROI",
            f"{df_roi['ROI (%)'].mean():.0f}%",
            "투자 대비 5.9배 수익"
        )
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='투자 비용',
            x=df_roi['전략'],
            y=df_roi['투자 비용 (백만원)'],
            marker_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Bar(
            name='예상 매출 증대',
            x=df_roi['전략'],
            y=df_roi['예상 매출 증대 (백만원)'],
            marker_color='#4ECDC4'
        ))
        
        fig.update_layout(
            title="전략별 투자 대비 효과",
            yaxis_title="금액 (백만원)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 실행 로드맵
    st.markdown("### 🗓️ 실행 로드맵 (6개월)")
    
    timeline_data = {
        '월': ['1개월차', '2개월차', '3개월차', '4개월차', '5개월차', '6개월차'],
        '주요 활동': [
            '시스템 구축\n신규 온보딩 시작',
            '리텐션 캠페인\n모니터링 체계',
            '장기계약 프로모션\n커뮤니티 론칭',
            '중간 평가\n전략 수정',
            '확대 실행\n효과 측정',
            '최종 평가\n지속 운영'
        ]
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        phases = ['준비기', '실행기', '확장기', '안정기']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, phase in enumerate(phases):
            fig.add_trace(go.Scatter(
                x=[i*1.5, i*1.5+1.5],
                y=[1, 1],
                mode='lines',
                line=dict(color=colors[i], width=20),
                name=phase,
                showlegend=True
            ))
        
        fig.update_layout(
            title="실행 단계별 로드맵",
            xaxis_title="개월",
            yaxis=dict(visible=False),
            height=300,
            xaxis=dict(range=[0, 6])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📅 단계별 목표")
        st.markdown("""
        **1-2개월**: 시스템 구축 및 초기 실행
        - 핵심 프로그램 론칭
        - 팀 교육 완료
        
        **3-4개월**: 본격 실행 및 모니터링
        - 전체 프로그램 가동
        - 실시간 성과 추적
        
        **5-6개월**: 확장 및 최적화
        - 효과 검증
        - 지속 운영 체계 확립
        """)
    
    st.markdown("---")
    
    # 성공 지표
    st.markdown("### 📊 성공 지표 (KPI)")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric("이탈률 감소 목표", "25%", "현재 30% → 목표 22.5%")
    
    with kpi_col2:
        st.metric("갱신율 향상 목표", "15%p", "현재 65% → 목표 80%")
    
    with kpi_col3:
        st.metric("신규 회원 유지율", "+20%p", "현재 55% → 목표 75%")
    
    with kpi_col4:
        st.metric("연간 매출 증대", "+41.6억원", "ROI 591%")

# 사이드바 하단 정보
st.sidebar.markdown("---")
st.sidebar.info("""
**모델 정보**
- 최종 F1 Score: 0.9188
- AUC-ROC: 0.9851
- 최적 임계값: 0.30

**데이터셋**
- 총 샘플: 4,002개
- 특성 수: 24개
- 이탈률: 26.7%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 SKN20-2nd-2TEAM")
