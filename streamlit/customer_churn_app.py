import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import os
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 네비게이션
st.sidebar.title("📋 목차")
menu = st.sidebar.radio(
    "분석 단계 선택",
    ["0. 프로젝트 개요", "1. 데이터 탐색", "2. 데이터 전처리", 
     "3. 모델 선정 단계", "4. 상위 모델 학습 및 평가", "5. 최종 결과"]
)

# 데이터 로드 함수
@st.cache_data
def load_data(filepath):
    """CSV 파일 로드"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# 이상치 탐지 함수
def detect_outliers_iqr(df, column):
    """IQR 방식으로 이상치 탐지"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# 범주형 변수 희귀 카테고리 탐지
def detect_rare_categories(df, column, threshold=0.01):
    """1% 미만 비율의 희귀 카테고리 탐지"""
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold]
    return rare_categories

# ====================
# 0. 프로젝트 개요
# ====================
if menu == "0. 프로젝트 개요":
    st.title("🏦 고객 이탈 예측 프로젝트")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📌 프로젝트 소개")
        st.markdown("""
        ### Bank Customer Churn Prediction
        
        본 프로젝트는 **은행 고객의 이탈을 예측**하는 머신러닝 분류 모델을 구축하는 것을 목표로 합니다.
        
        #### 🎯 프로젝트 목표
        - 고객 이탈 가능성을 사전에 예측하여 선제적 대응
        - 이탈 고객 특성 파악을 통한 고객 유지 전략 수립
        - 높은 Recall과 F1-score를 통한 균형잡힌 예측 모델 개발
        
        #### 📊 데이터셋 정보
        - **출처**: Kaggle - Bank Customer Churn Dataset
        - **규모**: 10,000개 행, 12개 열
        - **타겟 변수**: churn (0: 유지, 1: 이탈)
        """)
        
        st.markdown("---")
        
        st.subheader("🔍 주요 변수 설명")
        variable_info = pd.DataFrame({
            '변수명': ['credit_score', 'country', 'gender', 'age', 'tenure', 
                     'balance', 'products_number', 'credit_card', 'active_member', 
                     'estimated_salary', 'churn'],
            '설명': ['신용 점수', '국가', '성별', '나이', '거래 기간 (년)',
                   '계좌 잔액', '보유 상품 수', '신용카드 보유 여부', '활동 회원 여부',
                   '예상 연봉', '이탈 여부 (타겟)'],
            '타입': ['연속형', '범주형', '범주형', '연속형', '연속형',
                   '연속형', '범주형', '범주형', '범주형', '연속형', '범주형']
        })
        st.dataframe(variable_info, use_container_width=True)
    
    with col2:
        st.header("👥 팀 정보")
        st.info("""
        **팀명**: 1조
        
        **팀원**:
        - 김나현
        - 문창교
        - 이경현
        - 이승규
        - 정래원
        """)
        
        st.markdown("---")
        
        st.header("📈 분석 프로세스")
        st.markdown("""
        1️⃣ **데이터 탐색**
        - 결측치 및 이상치 확인
        
        2️⃣ **데이터 전처리**
        - 단변수/이변수/다변수 분석
        - 스케일링 및 인코딩
        
        3️⃣ **모델 선정**
        - 8개 모델 비교 평가
        
        4️⃣ **모델 최적화**
        - 하이퍼파라미터 튜닝
        
        5️⃣ **최종 결과**
        - 모델 해석 및 예측
        """)

# ====================
# 1. 데이터 탐색
# ====================
elif menu == "1. 데이터 탐색":
    st.title("🔍 데이터 탐색 (EDA)")
    st.markdown("---")
    
    # 파일 직접 로드
    st.subheader("📂 데이터 파일 로드")
    
    # 파일 경로 설정
    current_dir = Path(__file__).parent
    data_path = current_dir / "Bank Customer Churn Prediction.csv"
    
    try:
        df_raw = pd.read_csv(data_path)
        st.success(f"✅ 데이터 로드 성공: `{data_path}`")
    except FileNotFoundError:
        st.error(f"❌ 파일을 찾을 수 없습니다: `{data_path}`")
        st.info("파일 경로를 확인해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        st.stop()
    
    if df_raw is not None:
        
        # 기본 정보 표시
        st.success(f"✅ 데이터 로드 완료: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")
    
        # st.markdown("---")
        
        # 결측치 확인
        st.subheader("🔎 결측치 탐색")
        missing_data = df_raw.isnull().sum()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if missing_data.sum() == 0:
                st.success("✅ 결측치가 존재하지 않습니다!")
            else:
                st.warning(f"⚠️ 총 {missing_data.sum()}개의 결측치 발견")
            
            missing_df = pd.DataFrame({
                '변수': missing_data.index,
                '결측치 수': missing_data.values,
                '비율(%)': (missing_data.values / len(df_raw) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            fig = px.bar(
                missing_df,
                x='변수',
                y='결측치 수',
                title='변수별 결측치 분포',
                color='결측치 수',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 이상치 탐색
        st.subheader("🎯 이상치 탐색")
        
        st.info("""
        **이상치 탐색 기준**:
        - **연속형 변수**: IQR 기반 (Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과)
        - **범주형 변수**: 비율 1% 미만의 희귀 카테고리
        """)
        
        # 이상치 처리 방법 요약
        st.subheader("🛠️ 이상치 처리 방법")
        
        treatment_df = pd.DataFrame({
            '변수': ['age', 'credit_score', 'products_number (카테고리 4)'],
            '이상치 수': ['359', '15', '60'],
            '비율': ['3.59%', '0.15%', '0.006%'],
            '처리 방법': ['행 삭제', '행 삭제', '카테고리 3과 통합']
        })
        
        st.dataframe(treatment_df, use_container_width=True)
        
        st.info("""
        📌 **처리 결과**:
        - age와 credit_score의 이상치는 겹치지 않아 총 **374개 행(3.74%)** 삭제
        - products_number에서 4개 상품 보유 고객(60명)은 3개 상품 보유 그룹(266명)과 통합
        """)
    
    # df_raw 변수를 세션 상태에 저장 (다른 페이지에서 사용)
    st.session_state['df_raw'] = df_raw

# ====================
# 2. 데이터 전처리
# ====================
elif menu == "2. 데이터 전처리":
    st.title("⚙️ 데이터 전처리 및 EDA")
    st.markdown("---")
    
    # 파일 직접 로드
    current_dir = Path(__file__).parent
    data_path = current_dir / "Bank Customer Churn Prediction.csv"
    
    try:
        df = pd.read_csv(data_path)
        st.success(f"✅ 데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
    except FileNotFoundError:
        st.error(f"❌ 파일을 찾을 수 없습니다: `{data_path}`")
        st.stop()
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류 발생: {e}")
        st.stop()
    
    if df is not None:
        
        # 기본 전처리 (customer_id 제거, 인코딩)
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)
        
        # LabelEncoder 적용 (시각화용)
        df_encoded = df.copy()
        le = LabelEncoder()
        
        if 'gender' in df.columns and df['gender'].dtype == 'object':
            df_encoded['gender'] = le.fit_transform(df['gender'])
        if 'country' in df.columns and df['country'].dtype == 'object':
            df_encoded['country'] = le.fit_transform(df['country'])
        
        st.success(f"✅ 데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
        
        # Expander로 하위 목차 구성
        with st.expander("📊 단변수 분석", expanded=False):
            st.markdown("### 단변수 분석 (Univariate Analysis)")
            st.markdown("""
            각 변수의 개별 분포를 파악하여 데이터의 특성을 이해합니다.
            - 연속형 변수: 분포 형태, 중심 경향, 산포도
            - 범주형 변수: 각 범주의 빈도와 비율
            """)
            
            # 변수 선택
            all_columns = df_encoded.columns.tolist()
            if 'churn' in all_columns:
                all_columns.remove('churn')
            
            selected_col = st.selectbox("분석할 변수 선택", all_columns, key="univariate")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 시각화
                if df_encoded[selected_col].dtype in [np.float64, np.int64]:
                    # 연속형 변수 - 히스토그램과 박스플롯
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f'{selected_col} 히스토그램', f'{selected_col} 박스플롯'),
                        row_heights=[0.6, 0.4]
                    )
                    
                    # 히스토그램
                    fig.add_trace(
                        go.Histogram(x=df_encoded[selected_col], name='분포', 
                                   marker_color='skyblue'),
                        row=1, col=1
                    )
                    
                    # 박스플롯
                    fig.add_trace(
                        go.Box(x=df_encoded[selected_col], name='박스플롯',
                              marker_color='lightcoral'),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # 범주형 변수 - 막대 그래프
                    value_counts = df_encoded[selected_col].value_counts()
                    
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        labels={'x': selected_col, 'y': '빈도'},
                        title=f'{selected_col} 분포',
                        color=value_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 기초 통계량
                st.markdown("#### 📈 기초 통계")
                if df_encoded[selected_col].dtype in [np.float64, np.int64]:
                    stats_df = pd.DataFrame({
                        '통계량': ['평균', '중앙값', '표준편차', '최소값', '최대값', '왜도', '첨도'],
                        '값': [
                            f"{df_encoded[selected_col].mean():.2f}",
                            f"{df_encoded[selected_col].median():.2f}",
                            f"{df_encoded[selected_col].std():.2f}",
                            f"{df_encoded[selected_col].min():.2f}",
                            f"{df_encoded[selected_col].max():.2f}",
                            f"{df_encoded[selected_col].skew():.2f}",
                            f"{df_encoded[selected_col].kurtosis():.2f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    value_counts = df_encoded[selected_col].value_counts()
                    pct = (value_counts / len(df_encoded) * 100).round(2)
                    stats_df = pd.DataFrame({
                        '카테고리': value_counts.index,
                        '빈도': value_counts.values,
                        '비율(%)': pct.values
                    })
                    st.dataframe(stats_df, use_container_width=True)
            
            # 인사이트
            st.markdown("---")
            st.markdown("#### 💡 주요 인사이트")
            
            insights = {
                'credit_score': "- 신용 점수는 대체로 정규분포를 따르며, 400이 하한이고 이를 넘는 이상치가 존재\n- 신용 점수가 낮아질수록 사람 수가 적어지며, 감자기 증가하는 구간이 존재(600~850)",
                'age': "- 50%의 데이터가 32~44세에 밀집\n- 고령층의 수는 적음\n- 이상치가 많음(387개) ⇒ boxplot에서 62세 이상은 전부 이상치로 판단됨",
                'balance': "- histplot을 보니, 잔고가 0인 고객이 매우 많음\n- 잔고가 0인 고객이 전체 고객의 36.5%",
                'estimated_salary': "- Q2가 상자의 정중앙에 오고, 위 아래 수염 길이도 비슷함\n- 연봉 분포는 데이터가 대칭/고르게 분포되었음",
                'products_number': "- 1개(50.84%)나 2개(45.9%)의 상품을 이용하는 고객이 많음\n- 4개 상품 이용 고객은 매우 적음",
                'country': "- 프랑스 50.1%, 스페인 24.8%, 독일 25.1%\n- 고객층이 50%가 프랑스",
                'gender': "- 남자 54.6%, 여자 45.4%로 성비 비슷",
                'credit_card': "- 보유 70.5%, 미보유 29.5%",
                'active_member': "- 활동중인 회원 51.5%, 비활동중인 회원 48.5%로 비슷함",
                'tenure': "- 0년(거래 가입한 고객)이 10년은 400명대\n- 나머지 기간(1년~6년)은 대체로 비슷한 수준임(800~1000명대)"
            }
            
            if selected_col in insights:
                st.info(insights[selected_col])
            else:
                st.info("해당 변수에 대한 인사이트를 분석 중입니다.")
        
        with st.expander("🔗 이변수 분석", expanded=False):
            st.markdown("### 이변수 분석 (Bivariate Analysis)")
            st.markdown("""
            각 변수와 타겟 변수(churn) 간의 관계를 파악합니다.
            이를 통해 이탈에 영향을 미치는 요인을 식별할 수 있습니다.
            """)
            
            # 변수 선택 (churn 제외)
            feature_cols = [col for col in df_encoded.columns if col != 'churn']
            selected_feature = st.selectbox("비교할 변수 선택", feature_cols, key="bivariate")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if df_encoded[selected_feature].dtype in [np.float64, np.int64]:
                    # 연속형 변수 - 박스플롯
                    fig = px.box(
                        df_encoded,
                        x='churn',
                        y=selected_feature,
                        color='churn',
                        title=f'{selected_feature} vs Churn',
                        labels={'churn': '이탈 여부', selected_feature: selected_feature},
                        color_discrete_map={0: 'lightblue', 1: 'lightcoral'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # 범주형 변수 - 스택 바 차트
                    cross_tab = pd.crosstab(df_encoded[selected_feature], df_encoded['churn'])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=cross_tab.index,
                        y=cross_tab[0],
                        name='유지 (0)',
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=cross_tab.index,
                        y=cross_tab[1],
                        name='이탈 (1)',
                        marker_color='lightcoral'
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_feature} vs Churn',
                        xaxis_title=selected_feature,
                        yaxis_title='고객 수',
                        barmode='stack',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 이탈률 비교")
                
                if 'churn' in df_encoded.columns:
                    if df_encoded[selected_feature].dtype in [np.float64, np.int64]:
                        # 연속형 변수 - 이탈/유지 그룹별 통계
                        churn_stats = df_encoded.groupby('churn')[selected_feature].agg([
                            ('평균', 'mean'),
                            ('중앙값', 'median'),
                            ('표준편차', 'std')
                        ]).round(2)
                        churn_stats.index = ['유지 (0)', '이탈 (1)']
                        st.dataframe(churn_stats, use_container_width=True)
                    else:
                        # 범주형 변수 - 각 카테고리별 이탈률
                        churn_rate = df_encoded.groupby(selected_feature)['churn'].agg([
                            ('전체', 'count'),
                            ('이탈', 'sum'),
                            ('이탈률(%)', lambda x: (x.sum()/len(x)*100).round(2))
                        ])
                        st.dataframe(churn_rate, use_container_width=True)
            
            # 인사이트
            st.markdown("---")
            st.markdown("#### 💡 주요 인사이트")
            
            bivariate_insights = {
                'credit_score': "- 전반적 신용점수 분포 차이는 없어보임 (이탈인과 유지인의 IQR 범위가 유사)\n- 다만, 이탈 집단에서 신용점수가 4000이하의 이상치가 다수 발견됨 (특이점: 이탈집단의 credit_score < 4000 극단적 저신용 고객들이 존재하여 조기 이탈)",
                'age': "- 이탈 집단이 보자 전반보다 연령대가 높음(40중반~50초반)\n- 유지 집단은 연령대보다 젊음(30초~40초)",
                'balance': "- 이탈 집단의 잔고 평균이 조금 더 높음\n- 유지 집단이 이탈 집단보다 IQR 분포가 아래쪽으로 더 넓음\n- 유지 고객을 잔고하가 적은 사람이 많음\n- 유지 집단은 잔액이 0인 고객이 많음 ⇒ 잔액이 없으면 이탈 가능성 낮음",
                'estimated_salary': "- 전반적 연봉 분포는 비슷함",
                'products_number': "- 이탈집단의 이용상품수 1개 >>> 2개 > 3개 > 4개\n- 유지집단의 이용상품수 2개 > 1개\n- 유지집단 고객에서는 0인 국가로표시의 비율이 가장 높음\n- 이탈 집단은 2번(독일)>0번(프랑스)>1번(스페인) 순으로 많음",
                'country': "- 유지 집단 중에서는 0인 국가표시의 비율이 가장 높음\n- 이탈 집단은 2번 국가(독일)의 이탈률이 절반 정도로 높음",
                'gender': "- 유지 집단에서 남성과 비율이 높음\n- 이탈 집단에서 여자 비율이 높음",
                'credit_card': "- 유지집단/이탈집단 간 신용카드 보유 여부는 비슷함\n- 신용카드를 보유하면 따른 이탈 차이는 없음",
                'active_member': "- 이탈 집단일수록 비활동회원의 많음\n- 유지 집단일수록 활동회원이 높음\n- 비활동집단일수록 이탈률이 높음",
                'tenure': "- 카테고리로 10개로 많아서 그래프상 눈에 띄는 패턴은 없음"
            }
            
            if selected_feature in bivariate_insights:
                st.info(bivariate_insights[selected_feature])
            else:
                st.info("해당 변수에 대한 인사이트를 분석 중입니다.")
        
        with st.expander("🔢 다변수 분석", expanded=False):
            st.markdown("### 다변수 분석 (Multivariate Analysis)")
            st.markdown("""
            3개 이상의 변수 간 관계를 분석하여 복잡한 패턴을 파악합니다.
            - 상관관계 분석 (Correlation Analysis)
            - 다중공선성 검토 (VIF)
            """)
            
            st.markdown("---")
            st.markdown("#### 🔄 데이터 전처리")
            st.info("""
            **Scaling (표준화)**:
            - 연속형 변수의 평균을 0, 분산을 1로 변환
            - StandardScaler 적용
            
            **Encoding (인코딩)**:
            - 범주형 변수를 One-Hot Encoding으로 변환
            - 각 카테고리를 별도의 이진 변수로 분리
            """)
            
            # 스케일링 및 인코딩 수행
            from sklearn.preprocessing import StandardScaler
            
            # 연속형 변수 스케일링
            continuous_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
            if 'churn' in continuous_features:
                continuous_features.remove('churn')
            
            df_scaled = df_encoded.copy()
            scaler = StandardScaler()
            df_scaled[continuous_features] = scaler.fit_transform(df_encoded[continuous_features])
            
            # One-Hot Encoding (이미 인코딩된 경우 더미 변수 생성)
            categorical_features = []
            for col in ['gender', 'country', 'credit_card', 'active_member', 'products_number']:
                if col in df_scaled.columns:
                    categorical_features.append(col)
            
            if len(categorical_features) > 0:
                df_encoded_full = pd.get_dummies(df_scaled, columns=categorical_features, drop_first=False)
            else:
                df_encoded_full = df_scaled.copy()
            
            st.markdown("---")
            st.markdown("#### 📊 상관관계 분석 (Pearson Correlation)")
            
            # 상관관계 히트맵
            corr_matrix = df_encoded_full.corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="상관계수"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="변수 간 상관관계 히트맵"
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # # Churn과의 상관관계
            # st.markdown("#### 🎯 Churn과의 상관관계")
            
            if 'churn' in corr_matrix.columns:
                churn_corr = corr_matrix['churn'].drop('churn').sort_values(ascending=False)
            
            st.markdown("---")
            st.markdown("#### 🔍 다중공선성 검토 (VIF)")
            
            st.info("""
            **다중공선성: 특정 변수가 다른 변수들과 강한 선형관계를 가지는 현상. 다중공선성이 존재할 경우 모델 해석력 저하를 초래할 수 있음.
                           
            다중공선성 측정 지표 VIF (Variance Inflation Factor)**:
            - VIF = 1: 다른 변수들과 전혀 상관관계가 없음
            - 1 < VIF < 5: 약한~중간 정도의 상관관계
            - 5 < VIF < 10: 높은 상관관계, 주의 필요
            - VIF > 10: 심각한 다중공선성, 변수 제거 고려
            """)
            
            # VIF 계산은 시간이 오래 걸리므로 결과만 표시
            st.markdown("##### 📊 본 데이터셋의 VIF 측정 결과")
            
            vif_data = pd.DataFrame({
                'feature': ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary', 
                          'churn', 'country_1', 'country_2', 'gender_1', 'credit_card_1',
                          'active_member_1', 'products_number_2', 'products_number_3', 'products_number_4'],
                'VIF': [1.001658, 1.110699, 1.002220, 1.401081, 1.001055,
                       1.352674, 1.125197, 1.371610, 1.013243, 1.001673,
                       1.047173, 1.290294, 1.087827, 1.029537]
            })
            
            st.dataframe(vif_data, use_container_width=True)
            
            st.success("""
            ✅ **결과 해석**:
            - 모든 변수의 VIF 값이 1점대로 확인됨
            - 각 독립변수 간의 상관성이 낮고, 상호 독립성이 확보됨
            - **변수 제거나 차원 축소(PCA) 등의 추가 조치는 불필요**
            """)
        
        with st.expander("🔧 전처리 파이프라인", expanded=False):
            st.markdown("### 전처리 파이프라인")
            st.markdown("""
            모델링 단계에서 사용할 전처리 파이프라인을 구축합니다.
            - **ColumnTransformer**: 연속형/범주형 변수를 분리하여 처리
            - **Pipeline**: 전처리와 모델을 하나의 흐름으로 통합
            """)
            
            st.markdown("---")
            st.markdown("#### 📋 전처리 설정")
            
            pipeline_info = pd.DataFrame({
                '구분': ['연속형 변수', '범주형 변수', '데이터 분할', '클래스 불균형', '데이터 누수 방지'],
                '처리 내용': [
                    'StandardScaler / MinMaxScaler (모델 유형별 선택)',
                    "OneHotEncoder(drop='first', handle_unknown='ignore')",
                    "train_test_split(test_size=0.2, stratify=y, random_state=42)",
                    "Churn=1 비율 약 20.3% → Stratified Split 적용으로 유지",
                    "Pipeline 내부에서 fit은 train 데이터에만 수행 후 test에는 transform 적용"
                ]
            })
            
            st.dataframe(pipeline_info, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🔄 파이프라인 구조")
            
            st.code("""
# ColumnTransformer 구성
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Pipeline 구성
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# 학습 및 예측
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
            """, language='python')
            
            st.info("""
            💡 **장점**:
            - 데이터 누수(Data Leakage) 방지
            - 코드 재사용성 및 유지보수성 향상
            - 모델 변경 시에도 동일한 전처리 적용 보장
            """)
        
        # df를 세션 상태에 저장
        st.session_state['df_processed'] = df_encoded

# ====================
# 3. 모델 선정 단계
# ====================
elif menu == "3. 모델 선정 단계":
    st.title("🤖 모델 선정 단계")
    st.markdown("---")
    
    st.markdown("""
    ### 📌 모델 선정 목적
    
    본 프로젝트의 목표는 **은행 고객의 이탈을 예측**하는 것입니다.
    이를 위해 다양한 분류 모델을 학습하고 성능을 비교하여 최적의 모델을 선정합니다.
    
    #### 🎯 평가 기준
    - **Recall (재현율)**: 실제 이탈 고객을 얼마나 잘 찾아내는가?
    - **F1-score**: Precision과 Recall의 조화 평균 (균형잡힌 성능)
    - **일반화 성능**: Train/Test 점수 차이가 적은 모델
    """)
    
    st.markdown("---")
    
    st.subheader("🔬 학습 및 평가 모델 (총 8개)")
    
    models_info = pd.DataFrame({
        '모델명': [
            'Logistic Regression',
            'K-Nearest Neighbors (KNN)',
            'Support Vector Machine (SVM)',
            'Decision Tree',
            'Random Forest',
            'Bagging',
            'AdaBoost',
            'Neural Network (MLP)'
        ],
        '모델 유형': [
            '선형 모델',
            '거리 기반',
            '서포트 벡터',
            '트리 기반',
            '앙상블 (트리)',
            '앙상블 (배깅)',
            '앙상블 (부스팅)',
            '신경망'
        ],
        '특징': [
            '해석 가능, 빠른 학습',
            '거리 기반 분류, 단순',
            '비선형 경계, 고차원 적합',
            '해석 가능, 과적합 위험',
            '강력한 성능, 과적합 방지',
            '분산 감소, 안정성',
            '약한 학습기 결합',
            '복잡한 패턴 학습'
        ]
    })
    
    st.dataframe(models_info, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Baseline 모델 성능 비교")
    
    st.info("""
    **Baseline 설정**:
    - 모든 모델은 하이퍼파라미터를 기본값(default)으로 설정
    - 동일한 데이터 분할 적용 (train 80% / test 20%)
    - Stratified Split으로 클래스 비율 유지
    """)
    
    # 성능 데이터
    baseline_results = pd.DataFrame({
        'Model': ['AdaBoost', 'NN', 'Bagging', 'LogisticRegression', 
                  'RandomForest', 'SVM', 'KNN', 'DecisionTree'],
        'ROC_AUC': [0.8464, 0.8547, 0.8186, 0.8487, 0.8501, np.nan, 0.7995, 0.6919],
        'Accuracy': [0.8634, 0.8609, 0.8505, 0.8598, 0.8577, 0.8634, 0.8406, 0.7970],
        'F1': [0.6021, 0.5864, 0.5727, 0.5714, 0.5595, 0.5504, 0.5165, 0.5069],
        'Recall': [0.5103, 0.4872, 0.4949, 0.4615, 0.4462, 0.4128, 0.4205, 0.5154]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📋 성능 지표 (Test 데이터)")
        st.dataframe(baseline_results.style.highlight_max(subset=['F1', 'Recall'], color='lightgreen'), 
                    use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 모델별 F1-Score 및 Recall 비교")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=baseline_results['Model'],
            y=baseline_results['F1'],
            name='F1-Score',
            marker_color='skyblue'
        ))
        
        fig.add_trace(go.Bar(
            x=baseline_results['Model'],
            y=baseline_results['Recall'],
            name='Recall',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title='모델',
            yaxis_title='점수',
            height=400,
            legend=dict(x=0.7, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🏆 상위 3개 모델 선정")
    
    st.success("""
    **선정 기준**:
    - F1-Score와 Recall이 모두 양호
    - Train과 Test 점수 차이가 적음 (과적합 방지)
    
    **선정된 모델**:
    1. ⭐ **Logistic Regression** (F1: 0.5714, Recall: 0.4615)
    2. ⭐ **AdaBoost** (F1: 0.6021, Recall: 0.5103)
    3. ⭐ **Random Forest** (F1: 0.5595, Recall: 0.4462)
    """)
    
    # 선정 모델 비교
    top3_models = baseline_results[baseline_results['Model'].isin(['LogisticRegression', 'AdaBoost', 'RandomForest'])]
    
    fig = go.Figure()
    
    metrics = ['Accuracy', 'F1', 'Recall']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=top3_models['Model'],
            y=top3_models[metric],
            name=metric,
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title='상위 3개 모델 성능 비교',
        xaxis_title='모델',
        yaxis_title='점수',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    💡 **다음 단계**:
    선정된 3개 모델에 대해 하이퍼파라미터 튜닝을 진행하여 성능을 최적화합니다.
    """)

# ====================
# 4. 상위 모델 학습 및 평가
# ====================
elif menu == "4. 상위 모델 학습 및 평가":
    st.title("🎯 상위 모델 학습 및 평가")
    st.markdown("---")
    
    st.header("📦 학습된 모델 로드")
    
    # 모델 파일 경로 설정
    current_dir = Path(__file__).parent
    model_files = {
        'RandomForest': f'{current_dir}\\randomforest_model.pkl',
        'AdaBoost': f'{current_dir}\\adaboost_model.pkl',
        'LogisticRegression': f'{current_dir}\\logisticregression_model.pkl'
    }
    
    loaded_models = {}
    missing_models = []
    
    # 모델 로드 로그
    st.info("🔄 모델 로드를 시작합니다...")
    
    # 프로그레스 바
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 모델 로드
    for idx, (name, filepath) in enumerate(model_files.items()):
        status_text.text(f"로딩 중: {name}...")
        try:
            loaded_models[name] = joblib.load(filepath)
            st.success(f"✅ {name} 모델 로드 성공 - `{filepath}`")
        except FileNotFoundError:
            missing_models.append(name)
            st.error(f"❌ {name} 모델 파일을 찾을 수 없습니다: `{filepath}`")
        except Exception as e:
            missing_models.append(name)
            st.error(f"❌ {name} 모델 로드 중 오류 발생: {e}")
        
        # 프로그레스 바 업데이트
        progress_bar.progress((idx + 1) / len(model_files))
    
    status_text.text("모델 로드 완료!")
    
    if missing_models:
        st.warning(f"""
        ⚠️ 일부 모델 파일이 없습니다: {', '.join(missing_models)}
        
        노트북에서 모델을 먼저 학습하고 저장해주세요.
        """)
        st.stop()
    else:
        st.success(f"🎉 총 {len(loaded_models)}개의 모델이 성공적으로 로드되었습니다!")
    
    # 세션 상태에 저장
    st.session_state['loaded_models'] = loaded_models
    
    st.markdown("---")
    
    st.subheader("⚙️ 하이퍼파라미터 튜닝")
    
    st.markdown("""
    ### 🔧 튜닝 방법: RandomizedSearchCV
    
    - **탐색 방법**: 랜덤 서치 (효율적인 탐색)
    - **교차 검증**: 5-Fold Cross Validation
    - **평가 지표**: F1-Score (이탈 고객 탐지 중시)
    - **탐색 횟수**: n_iter=30
    """)
    
    st.markdown("---")
    
    # 튜닝 결과
    st.subheader("📊 하이퍼파라미터 튜닝 결과")
    
    tuning_params = pd.DataFrame({
        '모델': ['Logistic Regression', 'Random Forest', 'AdaBoost'],
        '주요 하이퍼파라미터': [
            "C=77.97, penalty='l1', solver='saga', max_iter=1544",
            "n_estimators=104, max_depth=17, max_features='log2', min_samples_split=13, min_samples_leaf=3",
            "n_estimators=285, learning_rate=1.23"
        ]
    })
    
    st.dataframe(tuning_params, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📈 튜닝 후 모델 성능")
    
    tuned_results = pd.DataFrame({
        'Model': ['RandomForest', 'AdaBoost', 'LogisticRegression'],
        'ROC_AUC': [0.856, 0.846, 0.847],
        'Accuracy': [0.858, 0.864, 0.784],
        'F1': [0.649, 0.593, 0.585],
        'Recall': [0.649, 0.487, 0.751]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📋 튜닝 후 성능 지표")
        st.dataframe(tuned_results.style.highlight_max(subset=['F1', 'Recall'], color='lightgreen'),
                    use_container_width=True)
        
        st.markdown("#### 📊 성능 개선 비교")
        
        # Baseline vs Tuned 비교
        comparison = pd.DataFrame({
            '모델': ['Random Forest', 'AdaBoost', 'Logistic Regression'],
            'Baseline F1': [0.560, 0.602, 0.571],
            'Tuned F1': [0.649, 0.593, 0.585],
            '개선': ['+0.089', '-0.009', '+0.014']
        })
        st.dataframe(comparison, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 튜닝 전후 성능 비교")
        
        # Baseline 데이터
        baseline_compare = pd.DataFrame({
            'Model': ['RandomForest', 'AdaBoost', 'LogisticRegression'],
            'Baseline_F1': [0.5595, 0.6021, 0.5714],
            'Tuned_F1': [0.649, 0.593, 0.585],
            'Baseline_Recall': [0.4462, 0.5103, 0.4615],
            'Tuned_Recall': [0.649, 0.487, 0.751]
        })
        
        fig = go.Figure()
        
        # F1 Score
        fig.add_trace(go.Bar(
            name='Baseline F1',
            x=baseline_compare['Model'],
            y=baseline_compare['Baseline_F1'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Tuned F1',
            x=baseline_compare['Model'],
            y=baseline_compare['Tuned_F1'],
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title='튜닝 전후 F1-Score 비교',
            xaxis_title='모델',
            yaxis_title='F1-Score',
            barmode='group',
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recall
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            name='Baseline Recall',
            x=baseline_compare['Model'],
            y=baseline_compare['Baseline_Recall'],
            marker_color='lightcoral'
        ))
        
        fig2.add_trace(go.Bar(
            name='Tuned Recall',
            x=baseline_compare['Model'],
            y=baseline_compare['Tuned_Recall'],
            marker_color='coral'
        ))
        
        fig2.update_layout(
            title='튜닝 전후 Recall 비교',
            xaxis_title='모델',
            yaxis_title='Recall',
            barmode='group',
            height=350
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🏆 최종 모델 선정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("최종 선정 모델", "Random Forest", "")
    with col2:
        st.metric("F1-Score", "0.649", "+0.089")
    with col3:
        st.metric("Recall", "0.649", "+0.203")
    
    st.success("""
    ### ✅ Random Forest 모델 선정 이유
    
    1. **최고의 성능**: F1-Score 0.649 최상위
    2. **대폭 개선**: Baseline 대비 F1-Score +0.089, Recall +0.203 향상
    3. **이탈 탐지 강화**: Recall 0.649로 이탈 고객의 64.9%를 정확히 탐지
    4. **일반화 성능**: Test 데이터에서도 안정적인 성능 유지
    5. **과적합 방지**: max_depth, min_samples_split 등의 파라미터 조정으로 일반화 성능 향상
    
    **핵심 성과**:
    - 이탈 고객을 놓치지 않으면서(높은 Recall)
    - 불필요한 오탐을 줄이는(균형잡힌 F1) 방향으로 최적화
    """)
    
    st.markdown("---")

# ====================
# 5. 최종 결과
# ====================
elif menu == "5. 최종 결과":
    st.title("🎉 최종 결과 및 모델 예측")
    st.markdown("---")
    
    # 프로젝트 요약
    st.header("📊 프로젝트 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("데이터 규모", "10,000건", "")
    with col2:
        st.metric("최종 모델", "Random Forest", "")
    with col3:
        st.metric("F1-Score", "0.649", "")
    with col4:
        st.metric("Recall", "0.649", "")
    
    st.markdown("---")
    
    # 프로세스 다이어그램
    st.subheader("🔄 분석 프로세스")
    
    st.markdown("""
    ```
    📊 데이터 수집
        ↓
    🔍 데이터 탐색 (EDA)
        ├─ 결측치 확인 ✅ (없음)
        ├─ 이상치 처리 (374건 제거)
        └─ 기본 통계 분석
        ↓
    ⚙️ 데이터 전처리
        ├─ 단변수 분석 (각 변수 분포 파악)
        ├─ 이변수 분석 (Churn과의 관계)
        ├─ 다변수 분석 (상관관계, VIF)
        └─ 파이프라인 구축
        ↓
    🤖 모델 학습
        ├─ 8개 모델 비교
        ├─ 상위 3개 선정
        └─ 하이퍼파라미터 튜닝
        ↓
    🏆 최종 모델: Random Forest
        ├─ F1-Score: 0.649
        ├─ Recall: 0.649
        └─ Accuracy: 0.858
    ```
    """)
    
    st.markdown("---")
    
    # 핵심 포인트
    st.subheader("💡 단계별 핵심 포인트")
    
    with st.expander("1️⃣ 데이터 탐색", expanded=False):
        st.markdown("""
        **주요 발견사항**:
        - ✅ 결측치 없음 (데이터 품질 양호)
        - ⚠️ age, credit_score에서 이상치 발견 (374건, 3.74%)
        - 📊 클래스 불균형: Churn=1이 약 20.3%
        
        **처리 방법**:
        - IQR 기반 이상치 제거
        - products_number의 희귀 카테고리(4개 상품) 통합
        - Stratified Split으로 클래스 비율 유지
        """)
    
    with st.expander("2️⃣ 데이터 전처리", expanded=False):
        st.markdown("""
        **단변수 분석**:
        - 각 변수의 분포 특성 파악
        - 왜도, 첨도 등 통계량 확인
        
        **이변수 분석**:
        - 이탈에 영향을 미치는 주요 변수 식별
        - age, products_number, active_member가 강한 영향
        
        **다변수 분석**:
        - 상관관계 분석: 강한 상관관계 변수 없음
        - VIF < 2: 다중공선성 문제 없음
        
        **전처리 파이프라인**:
        - StandardScaler: 연속형 변수
        - OneHotEncoder: 범주형 변수
        - 데이터 누수 방지
        """)
    
    with st.expander("3️⃣ 모델 선정 및 최적화", expanded=False):
        st.markdown("""
        **Baseline 비교 (8개 모델)**:
        - Logistic Regression, KNN, SVM, Decision Tree
        - Random Forest, Bagging, AdaBoost, Neural Network
        
        **상위 3개 선정**:
        1. AdaBoost (F1: 0.602)
        2. Logistic Regression (F1: 0.571)
        3. Random Forest (F1: 0.560)
        
        **하이퍼파라미터 튜닝**:
        - RandomizedSearchCV (n_iter=30, cv=5)
        - 평가 지표: F1-Score
        
        **최종 선정: Random Forest**
        - 튜닝 후 F1: 0.649 (+0.089)
        - 튜닝 후 Recall: 0.649 (+0.203)
        """)
    
    with st.expander("4️⃣ 최종 결과 및 인사이트", expanded=False):
        st.markdown("""
        **모델 성능**:
        - ✅ F1-Score: 0.649 (이탈/유지 균형)
        - ✅ Recall: 0.649 (이탈 고객 64.9% 탐지)
        - ✅ Accuracy: 0.858 (전체 정확도)
        
        **비즈니스 인사이트**:
        1. **나이**: 40대 중후반~50대 초반 고객이 이탈 위험 높음
        2. **상품 수**: 1개 상품만 이용하는 고객이 이탈 확률 높음
        3. **활동성**: 비활동 회원의 이탈률이 현저히 높음
        4. **잔액**: 잔액이 0인 고객은 오히려 이탈 확률 낮음
        5. **국가**: 독일 고객의 이탈률이 다른 국가 대비 높음
        
        **실무 활용**:
        - 고위험 고객 조기 식별 및 맞춤형 리텐션 전략
        - 40대 이상 + 1개 상품 + 비활동 회원 → 집중 관리
        - 추가 상품 가입 유도 및 활동성 증진 캠페인
        """)
    
    st.markdown("---")
    
    # 모델 예측 섹션
    st.header("🔮 고객 이탈 예측")
    st.markdown("학습된 Random Forest 모델을 사용하여 고객의 이탈 여부를 예측합니다.")
    
    # 모델 로드
    current_dir = Path(__file__).parent
    model_path = current_dir / 'randomforest_model.pkl'
    
    try:
        # 세션에서 모델 확인
        if 'loaded_models' in st.session_state and 'RandomForest' in st.session_state['loaded_models']:
            model = st.session_state['loaded_models']['RandomForest']
            st.success("✅ Random Forest 모델 로드 완료 (세션에서 가져옴)")
        else:
            # 직접 로드
            model = joblib.load(model_path)
            st.success(f"✅ Random Forest 모델 로드 완료 - `{model_path}`")
    except FileNotFoundError:
        st.error(f"❌ 모델 파일을 찾을 수 없습니다: `{model_path}`")
        st.info("먼저 '4. 상위 모델 학습 및 평가' 메뉴에서 모델을 로드하거나, 노트북에서 모델을 학습하고 저장해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        st.stop()
    
    st.markdown("---")
    
    # 입력 폼
    st.subheader("📝 고객 정보 입력")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("신용 점수", min_value=300, max_value=850, value=650, step=10)
        age = st.number_input("나이", min_value=18, max_value=100, value=40, step=1)
        tenure = st.number_input("거래 기간 (년)", min_value=0, max_value=10, value=5, step=1)
        balance = st.number_input("계좌 잔액 ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
    
    with col2:
        estimated_salary = st.number_input("예상 연봉 ($)", min_value=0.0, max_value=200000.0, value=80000.0, step=1000.0)
        products_number = st.selectbox("보유 상품 수", [1, 2, 3, 4], index=0)
        country = st.selectbox("국가", ["프랑스", "스페인", "독일"], index=0)
        gender = st.selectbox("성별", ["남성", "여성"], index=0)
    
    with col3:
        credit_card = st.selectbox("신용카드 보유", ["예", "아니오"], index=0)
        active_member = st.selectbox("활동 회원", ["예", "아니오"], index=0)
    
    # 예측 버튼
    if st.button("🔮 이탈 여부 예측하기", use_container_width=True):
        # 입력 데이터 전처리
        country_encoded = {"프랑스": 0, "스페인": 1, "독일": 2}[country]
        gender_encoded = 1 if gender == "남성" else 0
        credit_card_encoded = 1 if credit_card == "예" else 0
        active_member_encoded = 1 if active_member == "예" else 0
        
        # 입력 데이터 생성
        input_data = pd.DataFrame({
            'credit_score': [credit_score],
            'country': [country_encoded],
            'gender': [gender_encoded],
            'age': [age],
            'tenure': [tenure],
            'balance': [balance],
            'products_number': [products_number],
            'credit_card': [credit_card_encoded],
            'active_member': [active_member_encoded],
            'estimated_salary': [estimated_salary]
        })
        
        # 예측
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            st.markdown("---")
            
            # 결과 표시
            if prediction == 1:
                st.error("⚠️ **이탈 위험 고객입니다!**")
                churn_prob = prediction_proba[1] * 100
                st.metric("이탈 확률", f"{churn_prob:.1f}%", "위험")
                
                st.warning("""
                ### 🚨 권장 조치사항
                1. 고객 맞춤형 리텐션 캠페인 실시
                2. 추가 상품 가입 혜택 제공
                3. 전담 상담사 배정
                4. VIP 서비스 제공
                """)
            else:
                st.success("✅ **안정적인 고객입니다.**")
                stay_prob = prediction_proba[0] * 100
                st.metric("유지 확률", f"{stay_prob:.1f}%", "안정")
                
                st.info("""
                ### 💚 권장 조치사항
                1. 우수 고객 혜택 제공
                2. 정기적인 만족도 조사
                3. 신규 서비스 우선 안내
                """)
            
            # 확률 시각화
            fig = go.Figure(go.Bar(
                x=['유지', '이탈'],
                y=[prediction_proba[0]*100, prediction_proba[1]*100],
                marker_color=['lightgreen', 'lightcoral'],
                text=[f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='이탈 확률 분석',
                xaxis_title='예측 결과',
                yaxis_title='확률 (%)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.info("입력 데이터와 모델의 형식이 일치하는지 확인해주세요.")
    
    st.markdown("---")
    
    # 프로젝트 마무리
    st.subheader("🎓 프로젝트 결론")
    
    st.success("""
    ### ✅ 프로젝트 성과
    
    1. **데이터 품질**: 체계적인 전처리로 고품질 데이터셋 구축
    2. **모델 성능**: F1-Score 0.649로 균형잡힌 예측 성능 달성
    3. **실무 적용**: 64.9%의 이탈 고객을 사전에 탐지 가능
    4. **비즈니스 가치**: 선제적 리텐션 전략 수립 기반 마련
    """)
    
    st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏦 Bank Customer Churn Prediction Project</p>
    <p>Team 1 | 2024</p>
</div>
""", unsafe_allow_html=True)
