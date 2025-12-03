"""
데이터 전처리 함수 모듈
노트북 파일 (netflix_models_check.ipynb)의 인코딩 맵을 사용하며, Streamlit UI를 위해 한글 레이블을 사용합니다.
"""

import numpy as np

# =========================================================================
# 1. UI용 맵핑: {한글 레이블: 영문 값}
# =========================================================================
GENDER_MAP = {'여성': 'Female', '남성': 'Male', '기타': 'Other'}

SUBSCRIPTION_MAP = {'베이직': 'Basic', '스탠다드': 'Standard', '프리미엄': 'Premium'} 

REGION_MAP = {
    '아프리카': 'Africa',
    '아시아': 'Asia',
    '유럽': 'Europe',
    '북미': 'North America',
    '오세아니아': 'Oceania',
    '남미': 'South America'
}

DEVICE_MAP = {'데스크톱': 'Desktop', '노트북': 'Laptop', '모바일': 'Mobile', 'TV': 'TV', '태블릿': 'Tablet'}

FAVORITE_GENRE_MAP = {
    '액션': 'Action', 
    '코미디': 'Comedy', 
    '다큐멘터리': 'Documentary', 
    '드라마': 'Drama', 
    '호러': 'Horror', 
    '로맨스': 'Romance', 
    'SF': 'Sci-Fi'
}


def get_encoded_value(label, mapping):
    """주어진 맵핑에서 레이블에 해당하는 인코딩 값을 반환합니다."""
    # 레이블이 맵핑에 없을 경우 None 반환
    return mapping.get(label)


def prepare_model_input(user_data):
    """
    사용자 입력을 모델 입력 형식 (Label Encoded)으로 변환합니다.
    (user_data는 이미 영문 문자열을 포함하고 있다고 가정합니다.)

    Returns:
        numpy.array: 모델 입력용 배열 (1, 9)
    """
    
    # =========================================================================
    # 2. 모델 인코딩: {영문 값: 숫자 인코딩 값}
    # 이 영문 키는 위의 UI용 맵핑에서 생성된 영문 값과 완벽히 일치해야 합니다.
    # =========================================================================
    GENDER_ENCODE = {'Female': 0, 'Male': 1, 'Other': 2}
    SUBSCRIPTION_ENCODE = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    REGION_ENCODE = {
        'Africa': 0, 'Asia': 1, 'Europe': 2, 'North America': 3, 'Oceania': 4, 'South America': 5
    }
    DEVICE_ENCODE = {'Desktop': 0, 'Laptop': 1, 'Mobile': 2, 'TV': 3, 'Tablet': 4}
    FAVORITE_GENRE_ENCODE = {
        'Action': 0, 'Comedy': 1, 'Documentary': 2, 'Drama': 3, 'Horror': 4, 'Romance': 5, 'Sci-Fi': 6
    }
    
    
    # 3. 모델 입력 배열 생성
    input_vector = [
        user_data['age'],
        user_data['watch_hours'],
        user_data['last_login_days'],
        user_data['number_of_profiles'],
        
        # 영문 레이블을 인코딩 값으로 변환
        GENDER_ENCODE.get(user_data['gender']),
        SUBSCRIPTION_ENCODE.get(user_data['subscription_type']),
        REGION_ENCODE.get(user_data['region']),
        DEVICE_ENCODE.get(user_data['device']),
        FAVORITE_GENRE_ENCODE.get(user_data['favorite_genre'])
    ]
    
    # 4. 누락된 값이 없는지 확인 (None이 있으면 오류 발생)
    if None in input_vector:
        raise ValueError("입력 데이터에 누락된 카테고리 값이 있습니다.")
        
    return np.array([input_vector])