# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Case Fatality Rate'] = data['Total Deaths'] / data['Total Cases']
    return data.dropna(subset=['Longitude', 'Latitude'])

# Konfigurasi halaman
st.set_page_config(
    page_title="COVID-19 Indonesia Analytics",
    layout="wide",
    page_icon="🦠"
)

# Memuat data
data = load_data()
latest_data = data.sort_values('Date').groupby('Location').last().reset_index()

# Sidebar
st.sidebar.header("Konfigurasi")
n_clusters = st.sidebar.slider("Jumlah Cluster", 2, 5, 3)
start_date = st.sidebar.date_input("Tanggal Mulai", data['Date'].min())
end_date = st.sidebar.date_input("Tanggal Akhir", data['Date'].max())

# Tab utama
tab1, tab2, tab3 = st.tabs(["🗺 Peta Cluster", "📈 Analisis Tren", "📊 Ringkasan Risiko"])

with tab1:
    # Clustering dengan K-Means
    scaler = StandardScaler()
    features = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']
    X = scaler.fit_transform(latest_data[features].fillna(0))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    latest_data['Cluster'] = kmeans.fit_predict(X)
    
    # Visualisasi peta
    st.header("Peta Sebaran Cluster COVID-19")
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=latest_data,
        get_position=['Longitude', 'Latitude'],
        get_radius=50000,
        get_color="[Cluster*80, 200-Cluster*40, 150]",
        pickable=True
    )
    
    view_state = pdk.ViewState(
        latitude=-2.5489,
        longitude=118.0149,
        zoom=3.5,
        pitch=40
    )
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{Location}</b><br>"
                    "Total Kasus: {Total Cases}<br>"
                    "Total Kematian: {Total Deaths}<br>"
                    "Cluster: {Cluster}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    ))

with tab2:
    # Prediksi dengan Random Forest
    st.header("Prediksi Total Kasus")
    
    # Split data
    features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']
    target = 'Total Cases'
    
    X = data[features].fillna(0)
    y = data[target]
    
    # Model training
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # Prediksi
    prediction = model.predict(X.head(1))[0]
    st.metric("Prediksi Total Kasus untuk Contoh Data", f"{prediction:,.0f}")
    
    # Visualisasi tren
    st.header("Tren Kasus Harian")
    
    filtered = data[(data['Date'] >= pd.to_datetime(start_date)) & 
                   (data['Date'] <= pd.to_datetime(end_date))]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered['Date'], filtered['New Cases'], color='red', label='Kasus Baru')
    ax.plot(filtered['Date'], filtered['New Deaths'], color='black', label='Kematian Baru')
    ax.set_title("Tren Kasus Harian")
    ax.legend()
    st.pyplot(fig)

with tab3:
    # Analisis Risiko
    st.header("Klasifikasi Risiko Wilayah")
    
    # Hitung skor risiko
    latest_data['Risk Score'] = (
        0.4 * latest_data['Total Deaths'] +
        0.3 * latest_data['Total Cases'] +
        0.3 * latest_data['Population Density']
    )
    
    latest_data['Risk Level'] = pd.qcut(latest_data['Risk Score'], 
                                      q=3, 
                                      labels=["Rendah", "Sedang", "Tinggi"])
    
    # Tampilkan metrik
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Wilayah", len(latest_data))
    with col2:
        st.metric("Risiko Tinggi", latest_data[latest_data['Risk Level'] == "Tinggi"].shape[0])
    with col3:
        st.metric("Risiko Rendah", latest_data[latest_data['Risk Level'] == "Rendah"].shape[0])
    
    # Tabel data
    st.subheader("Detail Wilayah")
    st.dataframe(
        latest_data[['Location', 'Cluster', 'Risk Level', 'Total Cases', 'Total Deaths']]
        .sort_values('Risk Level', ascending=False),
        height=400,
        use_container_width=True
    )

# Menjalankan aplikasi: streamlit run app.py
