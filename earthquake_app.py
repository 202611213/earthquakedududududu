import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib

# --- 1. 모델 및 리소스 로드 ---
@st.cache_resource
def load_resources():
    model = joblib.load('eq_model.pkl')
    scaler = joblib.load('eq_scaler.pkl')
    return model, scaler

@st.cache_data
def get_data():
    df = pd.read_csv('earthquake.csv')
    # 유저님이 지정한 한글 컬럼명 매핑
    df = df.rename(columns={
        'time': '발생시간', 'place': '발생지역', 'status': '검토상태',
        'tsunami': '쓰나미여부', 'significance': '영향도', 'data_type': '데이터유형',
        'magnitudo': '규모', 'state': '지역', 'longitude': '경도',
        'latitude': '위도', 'depth': '진원깊이', 'date': '발생일시'
    })
    return df

try:
    model, scaler = load_resources()
    df_master = get_data()

    # ⭐ 핵심: 모델이 학습되었을 때의 '정확한 순서'로 리스트 생성
    # 에러 메시지 순서에 따라 [규모, 영향도, 진원깊이, 위도, 경도] 순일 가능성이 가장 높습니다.
    FEATURES_ORDER = ['규모', '영향도', '진원깊이', '위도', '경도']

    # --- 데이터 전처리 (Cluster 생성) ---
    if 'cluster' not in df_master.columns:
        # 모델이 원하는 순서대로 컬럼을 뽑아서 전달
        X_all = df_master[FEATURES_ORDER] 
        X_all_scaled = scaler.transform(X_all)
        df_master['cluster'] = model.predict(X_all_scaled)

    # --- UI 구성 ---
    st.title("🌍 지진 위험도 예측 시스템 (순서 교정판)")
    
    st.sidebar.header("📍 데이터 입력")
    mag = st.sidebar.slider("규모", 0.0, 10.0, 5.0)
    sig = st.sidebar.number_input("영향도", value=100)
    dep = st.sidebar.number_input("진원깊이", value=10.0)
    lat = st.sidebar.number_input("위도", value=37.5)
    lon = st.sidebar.number_input("경도", value=127.0)
    
    btn = st.sidebar.button("위험도 분석 실행")

    risk_dict = {0: '높음', 1: '낮음', 2: '중간'}
    colors = {0: 'red', 1: 'blue', 2: 'green'}

    if btn:
        # 예측용 데이터프레임도 '순서'를 완벽하게 일치시킴
        input_df = pd.DataFrame([[mag, sig, dep, lat, lon]], columns=FEATURES_ORDER)
        
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        # 결과 출력
        st.success(f"### 분석 결과: 위험도 [{risk_dict.get(pred, '알수없음')}]")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("예측 군집", f"Cluster {pred}")
            st.write("#### 📊 인근 지역 통계 (반경 5도)")
            near_df = df_master[
                (df_master['위도'].between(lat-5, lat+5)) & 
                (df_master['경도'].between(lon-5, lon+5))
            ]
            if not near_df.empty:
                chart = near_df['cluster'].value_counts(normalize=True).rename(index=risk_dict)
                st.bar_chart(chart)

        with c2:
            st.write("#### 🗺️ 지진 분포 지도")
            m = folium.Map(location=[lat, lon], zoom_start=4)
            sample = df_master.sample(min(2000, len(df_master)), random_state=42)
            
            for _, row in sample.iterrows():
                folium.CircleMarker(
                    location=[row['위도'], row['경도']],
                    radius=2, color=colors.get(row['cluster'], 'gray'),
                    fill=True, fill_opacity=0.4
                ).add_to(m)

            folium.Marker([lat, lon], icon=folium.Icon(color='black', icon='star')).add_to(m)
            st_folium(m, width="100%", height=400)
    else:
        st.info("사이드바에 값을 입력하고 버튼을 눌러주세요.")

except Exception as e:
    st.error(f"🚨 오류 발생: {e}")
    # 만약 또 순서 에러가 나면, 모델이 기억하는 진짜 순서를 강제로 출력해 확인합니다.
    if 'scaler' in locals():
        st.write("모델이 기대하는 특성 순서:", scaler.feature_names_in_.tolist())