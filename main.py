import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np

# Streamlit 앱 설정
st.set_page_config(
    page_title="피트니스 데이터 상관관계 분석기",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(uploaded_file):
    """업로드된 CSV 파일을 읽어 DataFrame으로 반환합니다."""
    # 인코딩 문제 해결을 위해 'cp949'로 시도 후, 실패 시 'utf-8'로 시도
    try:
        df = pd.read_csv(uploaded_file, encoding='cp949')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None
    return df

def clean_and_prepare_data(df):
    """
    데이터를 정리하고 숫자형 데이터만 선택하여 상관관계를 분석할 준비를 합니다.
    - 불필요한 공백 제거
    - 숫자형이 아닌 컬럼 제거
    - 결측치 처리 (중앙값으로 대체)
    """
    # 1. 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 2. 숫자형 데이터만 선택 (상관관계 분석을 위함)
    numeric_df = df.select_dtypes(include=np.number)

    # 3. 결측치 처리: 각 컬럼의 중앙값으로 대체
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 4. 분석에 부적합한 단순 ID, 코드성 컬럼 제거 (필요시 수정)
    # '측정회차', '나이', '신장', '체중' 등은 분석에 유용하므로 유지
    
    return numeric_df

def calculate_top_correlations(df):
    """상관행렬을 계산하고 가장 높은 양의/음의 상관관계를 찾습니다."""
    # 상관 행렬 계산
    corr_matrix = df.corr().abs()
    
    # 자기 자신과의 상관관계 (1) 및 중복 쌍 (A-B와 B-A) 제거
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 양의 상관관계 (절댓값X)
    full_corr = df.corr()
    
    # 가장 높은 양의 상관관계 찾기 (절댓값 아님)
    positive_corr = full_corr.unstack().sort_values(ascending=False)
    # 자기 자신과의 관계(1.0) 제외
    positive_corr = positive_corr[positive_corr < 1.0] 
    top_positive = positive_corr.drop_duplicates().head(1)
    
    # 가장 높은 음의 상관관계 찾기
    negative_corr = full_corr.unstack().sort_values(ascending=True)
    top_negative = negative_corr.drop_duplicates().head(1)

    return full_corr, top_positive, top_negative

def display_correlation_pair(title, correlation_series):
    """가장 높은 상관관계 쌍을 시각화합니다."""
    if not correlation_series.empty:
        # 시리즈에서 인덱스(컬럼 쌍)와 값(상관계수) 추출
        pair_index = correlation_series.index[0]
        correlation_value = correlation_series.iloc[0]
        
        col1_name, col2_name = pair_index
        
        st.markdown(f"### {title}: **{col1_name}** 와 **{col2_name}**")
        st.info(f"상관계수: **{correlation_value:.4f}**")

        # 산점도 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=col1_name, y=col2_name, data=st.session_state.data, ax=ax, 
                    scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        
        ax.set_title(f'{col1_name} vs {col2_name} 산점도 (r = {correlation_value:.4f})')
        ax.set_xlabel(col1_name)
        ax.set_ylabel(col2_name)
        st.pyplot(fig)
        
        st.markdown("---")
    else:
        st.warning(f"{title}를 찾을 수 없습니다. 데이터를 확인해 주세요.")

# --- 메인 Streamlit 앱 ---
st.title("🏃‍♀️ 피트니스 데이터 상관관계 분석기")
st.markdown("---")

# 1. 파일 업로드 섹션
uploaded_file = st.file_uploader(
    "**CSV 파일을 업로드하세요** (예: `fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv`)", 
    type=['csv']
)

if uploaded_file is not None:
    # 2. 데이터 로드 및 준비
    df = load_data(uploaded_file)
    if df is not None:
        try:
            processed_df = clean_and_prepare_data(df.copy())
            st.session_state.data = processed_df # 산점도에 사용할 처리된 데이터 저장
            
            st.subheader("✅ 데이터 로드 및 전처리 완료")
            st.dataframe(processed_df.head())
            st.write(f"총 {len(processed_df)}개의 행과 {len(processed_df.columns)}개의 숫자형 컬럼이 분석에 사용됩니다.")

            # 3. 상관관계 분석
            full_corr, top_positive, top_negative = calculate_top_correlations(processed_df)

            st.markdown("---")
            st.header("📊 상관관계 분석 결과")
            
            # 4. 버튼 기반 결과 표시
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                if st.button("➕ 가장 높은 **양의 상관관계** 보기", use_container_width=True):
                    st.session_state['show_positive'] = True
                    st.session_state['show_negative'] = False
            
            with col_neg:
                if st.button("➖ 가장 높은 **음의 상관관계** 보기", use_container_width=True):
                    st.session_state['show_negative'] = True
                    st.session_state['show_positive'] = False

            # 초기 상태 또는 버튼 클릭에 따른 결과 표시
            if 'show_positive' in st.session_state and st.session_state['show_positive']:
                display_correlation_pair("🥇 가장 높은 양의 상관관계", top_positive)
            
            if 'show_negative' in st.session_state and st.session_state['show_negative']:
                display_correlation_pair("📉 가장 높은 음의 상관관계", top_negative)

            st.markdown("---")
            
            # 5. 전체 상관관계 행렬 히트맵 (추가 정보)
            st.subheader("🔍 전체 상관관계 행렬 히트맵")
            fig_corr, ax_corr = plt.subplots(figsize=(18, 15))
            # 마스크를 사용하여 중복된 부분 제거
            mask = np.triu(full_corr)
            sns.heatmap(
                full_corr, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                cbar=True, 
                mask=mask,
                linewidths=.5,
                linecolor='black',
                ax=ax_corr
            )
            ax_corr.set_title('데이터 속성 간의 상관관계 행렬')
            st.pyplot(fig_corr)


        except Exception as e:
            st.error(f"데이터 처리 중 예상치 못한 오류가 발생했습니다: {e}")

else:
    st.info("시작하려면 CSV 파일을 업로드해 주세요.")

st.markdown("---")
st.caption("© 2025 AI-Powered Data Analysis Tool")
