# dashboard/dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Judul Dashboard
st.title("Dashboard Analisis Kualitas Udara (Air Quality Dataset)")
st.markdown("""
**Nama:** Naufal Suryo Saputro  
**Email:** a008ybm371@devacademy.id  
**ID Dicoding:** suryonaufal  
""")

# Memuat data
data_path = "dashboard/main_data.csv"  # Path relatif di folder dashboard/
try:
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])  # Pastikan kolom datetime dalam format datetime
    st.write("Data berhasil dimuat!")

    # Sidebar untuk filter (setelah data dimuat)
    st.sidebar.header("Filter Data")
    selected_station = st.sidebar.multiselect("Pilih Stasiun", options=df['station'].unique(), default=df['station'].unique())
    selected_year = st.sidebar.slider("Pilih Tahun", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=(2013, 2017))

    # Filter data berdasarkan input pengguna
    filtered_df = df[df['station'].isin(selected_station) & (df['year'].between(selected_year[0], selected_year[1]))]

    # Pertanyaan 1: Bagaimana tren tingkat PM2.5 secara keseluruhan di semua stasiun dari tahun ke tahun, dan pada bulan apa tingkat polusi tertinggi biasanya terjadi?
    st.header("Pertanyaan 1: Tren PM2.5 Tahunan dan Polusi Bulanan")
    yearly_pm25 = filtered_df.groupby('year')['PM2.5'].mean()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    yearly_pm25.plot(kind='line', marker='o', ax=ax1)
    ax1.set_title('Tren Rata-rata PM2.5 per Tahun')
    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('Rata-rata PM2.5')
    ax1.grid(True)
    st.pyplot(fig1)

    # Pertanyaan 1: Bagaimana tren tingkat PM2.5 secara keseluruhan di semua stasiun dari tahun ke tahun, dan pada bulan apa tingkat polusi tertinggi biasanya terjadi?
    monthly_pm25 = filtered_df.groupby('month')['PM2.5'].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    monthly_pm25.plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_title('Rata-rata PM2.5 per Bulan')
    ax2.set_xlabel('Bulan')
    ax2.set_ylabel('Rata-rata PM2.5')
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'])
    st.pyplot(fig2)

    # Pertanyaan 2: Stasiun mana yang memiliki tingkat PM2.5 tertinggi, dan bagaimana hubungan antara kecepatan angin dan PM2.5 bervariasi antar stasiun?
    st.header("Pertanyaan 2: PM2.5 per Stasiun dan Hubungan dengan Kecepatan Angin")
    station_pm25 = filtered_df.groupby('station')['PM2.5'].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    unique_stations = filtered_df['station'].unique()
    palette = sns.color_palette('tab10', n_colors=len(unique_stations))
    sns.barplot(x=station_pm25.index, y=station_pm25.values, palette=palette, ax=ax3)
    ax3.set_title('Rata-rata PM2.5 per Stasiun')
    ax3.set_xlabel('Stasiun')
    ax3.set_ylabel('Rata-rata PM2.5')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Pertanyaan 2: Stasiun mana yang memiliki tingkat PM2.5 tertinggi, dan bagaimana hubungan antara kecepatan angin dan PM2.5 bervariasi antar stasiun?
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x='WSPM', y='PM2.5', hue='station', alpha=0.5, ax=ax4)
    ax4.set_title('Hubungan Kecepatan Angin (WSPM) dan PM2.5 per Stasiun')
    ax4.set_xlabel('Kecepatan Angin (m/s)')
    ax4.set_ylabel('PM2.5')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig4)

    # Analisis Lanjutan: Matriks Korelasi
    st.header("Analisis Lanjutan: Matriks Korelasi")
    corr_columns = ['PM2.5', 'WSPM', 'TEMP', 'PRES', 'DEWP']
    corr_df = filtered_df[corr_columns]
    correlation_matrix = corr_df.corr()
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax5)
    ax5.set_title('Matriks Korelasi Antar Variabel')
    st.pyplot(fig5)

    # Analisis Lanjutan: Clustering Stasiun
    st.header("Analisis Lanjutan: Pengelompokan Stasiun")
    # Menyiapkan data untuk clustering: rata-rata PM2.5 dan WSPM per stasiun
    cluster_data = filtered_df.groupby('station')[['PM2.5', 'WSPM']].mean().reset_index()

    # Standarisasi data agar skala seragam
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data[['PM2.5', 'WSPM']])

    # Menerapkan K-Means dengan 3 cluster
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Visualisasi clustering
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=cluster_data, x='WSPM', y='PM2.5', hue='Cluster', style='Cluster', 
                    s=100, palette='deep')
    for i, row in cluster_data.iterrows():
        ax6.text(row['WSPM'] + 0.01, row['PM2.5'], row['station'], fontsize=9)
    ax6.set_xlim(left=cluster_data['WSPM'].min() - 0.1, right=2)
    ax6.set_title('Pengelompokan Stasiun Berdasarkan PM2.5 dan Kecepatan Angin')
    ax6.set_xlabel('Rata-rata Kecepatan Angin (WSPM)')
    ax6.set_ylabel('Rata-rata PM2.5')
    ax6.legend(title='Cluster')
    st.pyplot(fig6)

    # Kesimpulan
    st.header("Kesimpulan")
    st.markdown("""
    - **Pertanyaan 1**: 
        - Tren PM2.5 dari tahun ke tahun menunjukkan fluktuasi, dengan puncak tertinggi terjadi pada tahun 2017.  
        - Tingkat polusi tertinggi terjadi pada bulan Desember dipengaruhi oleh faktor musiman seperti kecepatan angin dan suhu. Hal ini diperjelas dengan matriks korelasi antar variabel yang menunjukkan kecepatan angin yang memiliki efek moderat dalam mengurangi PM2.5 dan hubungan suhu yang memberikan pengaruh minimal terhadap konsentrasi PM2.5 (perlu analisis lebih lanjut).
    - **Pertanyaan 2**:
        - Stasiun Dongsi memiliki tingkat PM2.5 tertinggi secara rata-rata. Hal ini menunjukkan lokasi ini lebih terpapar polusi. Hal ini masuk akal karena stasiun dongsi termasuk sebagai stasiun urban dengan PM2.5 tinggi dan kecepatan angin lemah.  
        - Hubungan antara kecepatan angin dan PM2.5 bervariasi pada beberapa stasiun.Meskipun begitu dapat diketahui bahwa semakin kencang kecepatan angin, semakin sedikit stasiun yang memiliki kadar PM2.5 tinggi (tanpa hubungan variabel lain).
    """)

except FileNotFoundError:
    st.error("File main_data.csv tidak ditemukan. Pastikan file ada di folder dashboard/.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.stop()
