import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os


# --- 1. 모델 및 리소스 로드 ---
@st.cache_resource
def load_resources():
    # pkl 파일들이 실행 파일과 같은 위치에 있어야 합니다.
    model = joblib.load('eq_model.pkl')
    scaler = joblib.load('eq_scaler.pkl')
    return model, scaler


@st.cache_data
def get_data():
    # ⭐ [핵심 수정] 무조건 깃허브에 있는 파일명인 'earthquake_small.csv'만 읽습니다.
    file_name = 'earthquake_small.csv'
   
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
  
    # 컬럼명 한글 매핑
    df = df.rename(columns={
        'time': '발생시간', 'place': '발생지역', 'status': '검토상태',
        'tsunami': '쓰나미여부', 'significance': '영향도', 'data_type': '데이터유형',
        'magnitudo': '규모', 'state': '지역', 'longitude': '경도',
        'latitude': '위도', 'depth': '진원깊이', 'date': '발생일시'
    })
    return df


try:
    # 페이지 레이아웃 설정
    st.set_page_config(page_title="지진 위험도 분석", layout="wide")
   
    model, scaler = load_resources()
    df_master = get_data()


    # ⭐ [오류 해결] 모델이 요구한 3가지 특성 순서: 규모 -> 진원깊이 -> 영향도
    FEATURES_ORDER = ['규모', '진원깊이', '영향도']


    # --- 데이터 전처리 (Cluster 생성) ---
    if not df_master.empty and 'cluster' not in df_master.columns:
        X_all = df_master[FEATURES_ORDER]
        X_all_scaled = scaler.transform(X_all)
        df_master['cluster'] = model.predict(X_all_scaled)


    # --- UI 구성 ---
    st.title("🌍 지진 위험도 예측 시스템")
   
    st.sidebar.header("📍 데이터 입력")
    mag = st.sidebar.slider("규모", 0.0, 10.0, 5.0)
    dep = st.sidebar.number_input("진원깊이 (km)", value=10.0)
    sig = st.sidebar.number_input("영향도 (Significance)", value=100)
   
    st.sidebar.markdown("---")
    lat = st.sidebar.number_input("지도 위도", value=37.5)
    lon = st.sidebar.number_input("지도 경도", value=127.0)
   
    btn = st.sidebar.button("위험도 분석 실행")


    # 요청하신 군집 설명 매핑
    risk_dict = {
        0: '군집 0 - 규모가 크고 깊이가 얕음(위험)',
        1: '군집 1 - 규모, 깊이가 얕음(낮음)',
        2: '군집 2 - 규모가 크고 깊이가 깊음(중간)'
    }
    colors = {0: 'red', 1: 'blue', 2: 'green'}


    if btn:
        # 모델 예측 시 특성 순서 엄수
        input_df = pd.DataFrame([[mag, dep, sig]], columns=FEATURES_ORDER)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]


        st.success(f"### 분석 결과: {risk_dict.get(pred, '알 수 없음')}")
       
        col1, col2 = st.columns([1, 1.5])
       
        with col1:
            st.write("#### 📊 주변 지역 통계 (반경 5도)")
            near_df = df_master[
                (df_master['위도'].between(lat-5, lat+5)) &
                (df_master['경도'].between(lon-5, lon+5))
            ]
            if not near_df.empty:
                chart_data = near_df['cluster'].value_counts(normalize=True).rename(index=risk_dict)
                st.bar_chart(chart_data)
            else:
                st.write("인근에 과거 데이터가 없습니다.")


        with col2:
            st.write("#### 🗺️ 주변 지진 분포")
            m = folium.Map(location=[lat, lon], zoom_start=5)
           
            # 지도 안정성을 위해 1000개만 샘플링
            map_sample = df_master.sample(min(1000, len(df_master)), random_state=42)
           
            for _, row in map_sample.iterrows():
                folium.CircleMarker(
                    location=[row['위도'], row['경도']],
                    radius=3,
                    color=colors.get(row['cluster'], 'gray'),
                    fill=True, fill_opacity=0.6,
                    popup=risk_dict.get(row['cluster'], "")
                ).add_to(m)


            folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='star')).add_to(m)
           
            # 지도 크기를 화면에 맞게 100%로 설정
            st_folium(m, width="100%", height=500, returned_objects=[])
    else:
        st.info("왼쪽 사이드바에서 값을 입력하고 버튼을 눌러주세요.")


except Exception as e:
    st.error(f"🚨 시스템 오류 발생: {e}")

