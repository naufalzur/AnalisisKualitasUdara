# dashboard/dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide")

# --- Judul Dashboard ---
st.title("Dashboard Analisis Kualitas Udara")
st.markdown("""
Ini adalah dashboard interaktif untuk menganalisis data kualitas udara dari berbagai stasiun pemantauan.
- **Nama:** Naufal Suryo Saputro
- **Email:** a008ybm371@devacademy.id
- **ID Dicoding:** suryonaufal
""")

# --- Fungsi untuk Memuat Data ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# --- Memuat Data ---
data_path = "dashboard/main_data.csv"
try:
    df = load_data(data_path)

    # --- Sidebar untuk Filter ---
    st.sidebar.header("Filter Data")
    
    # Filter Stasiun
    all_stations = df['station'].unique()
    selected_station = st.sidebar.multiselect("Pilih Stasiun", options=all_stations, default=all_stations)
    
    # Filter Rentang Tanggal
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # --- Filter Data Berdasarkan Input Pengguna ---
    start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
    filtered_df = df[
        (df['station'].isin(selected_station)) &
        (df['datetime'] >= start_date) &
        (df['datetime'] <= end_date)
    ]

    if filtered_df.empty:
        st.warning("Tidak ada data yang tersedia untuk filter yang dipilih. Silakan ubah pilihan filter Anda.")
        st.stop()

    # --- Layout Dashboard dengan Kolom ---
    col1, col2 = st.columns(2)

    with col1:
        # --- Pertanyaan 1: Tren PM2.5 Tahunan ---
        st.header("Tren PM2.5 Tahunan")
        st.markdown("Bagaimana tren tingkat PM2.5 dari tahun ke tahun?")
        
        yearly_pm25 = filtered_df.groupby('year')['PM2.5'].mean()
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        yearly_pm25.plot(kind='line', marker='o', ax=ax1, color='dodgerblue')
        ax1.set_title('Rata-rata PM2.5 per Tahun', fontsize=14)
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

    with col2:
        # --- Pertanyaan 1: Pola PM2.5 Bulanan ---
        st.header("Pola Polusi Bulanan")
        st.markdown("Pada bulan apa tingkat polusi tertinggi biasanya terjadi?")
        
        monthly_pm25 = filtered_df.groupby('month')['PM2.5'].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        monthly_pm25.plot(kind='bar', color='skyblue', ax=ax2)
        ax2.set_title('Rata-rata PM2.5 per Bulan', fontsize=14)
        ax2.set_xlabel('Bulan')
        ax2.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax2.set_xticks(range(12))
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'], rotation=45)
        st.pyplot(fig2)

    st.divider()

    # --- Pertanyaan 2: Perbandingan Antar Stasiun ---
    st.header("Perbandingan Antar Stasiun")
    st.markdown("Stasiun mana yang memiliki tingkat PM2.5 tertinggi dan bagaimana hubungannya dengan kecepatan angin?")
    
    col3, col4 = st.columns(2)
    with col3:
        station_pm25 = filtered_df.groupby('station')['PM2.5'].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=station_pm25.values, y=station_pm25.index, palette='viridis', ax=ax3)
        ax3.set_title('Rata-rata PM2.5 per Stasiun', fontsize=14)
        ax3.set_xlabel('Rata-rata PM2.5 (µg/m³)')
        ax3.set_ylabel('Stasiun')
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df.sample(min(1000, len(filtered_df))), x='WSPM', y='PM2.5', hue='station', alpha=0.6, ax=ax4)
        ax4.set_title('Hubungan Kecepatan Angin (WSPM) dan PM2.5', fontsize=14)
        ax4.set_xlabel('Kecepatan Angin (m/s)')
        ax4.set_ylabel('PM2.5 (µg/m³)')
        ax4.legend(title='Stasiun', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig4)

    st.divider()

    # --- Analisis Lanjutan ---
    st.header("Analisis Lanjutan")
    
    tab1, tab2, tab3 = st.tabs(["Matriks Korelasi", "Segmentasi Stasiun", "Peta Geospasial Polusi"])

    with tab1:
        # --- Matriks Korelasi ---
        st.subheader("Matriks Korelasi Antar Variabel Cuaca")
        corr_columns = ['PM2.5', 'WSPM', 'TEMP', 'PRES', 'DEWP']
        correlation_matrix = filtered_df[corr_columns].corr()
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax5, fmt=".2f")
        ax5.set_title('Matriks Korelasi', fontsize=14)
        st.pyplot(fig5)

    with tab2:
        # --- ANALISIS LANJUTAN BARU 1: SEGMENTASI MANUAL ---
        st.subheader("Segmentasi Stasiun Berdasarkan Karakteristik")
        
        # 1. Menyiapkan data: rata-rata PM2.5 dan WSPM per stasiun
        segment_data = filtered_df.groupby('station')[['PM2.5', 'WSPM']].mean().reset_index()

        if not segment_data.empty:
            # 2. Menentukan ambang batas menggunakan median
            pm25_threshold = segment_data['PM2.5'].median()
            wspm_threshold = segment_data['WSPM'].median()

            # 3. Fungsi untuk membuat segmen
            def create_segment(row):
                if row['PM2.5'] >= pm25_threshold and row['WSPM'] < wspm_threshold:
                    return 'Polusi Tinggi, Angin Rendah'
                elif row['PM2.5'] >= pm25_threshold and row['WSPM'] >= wspm_threshold:
                    return 'Polusi Tinggi, Angin Kencang'
                elif row['PM2.5'] < pm25_threshold and row['WSPM'] < wspm_threshold:
                    return 'Polusi Rendah, Angin Rendah'
                else:
                    return 'Polusi Rendah, Angin Kencang'

            # 4. Membuat kolom segmen
            segment_data['Segment'] = segment_data.apply(create_segment, axis=1)

            # 5. Visualisasi segmentasi
            fig6, ax6 = plt.subplots(figsize=(11, 7))
            sns.scatterplot(data=segment_data, x='WSPM', y='PM2.5', hue='Segment', style='Segment', 
                            s=200, palette='plasma', ax=ax6)
            
            # Menambahkan label nama stasiun
            for i, row in segment_data.iterrows():
                ax6.text(row['WSPM'] + 0.02, row['PM2.5'], row['station'], fontsize=9, ha='left')
            
            ax6.set_title('Segmentasi Stasiun Berdasarkan Rata-rata PM2.5 dan Kecepatan Angin', fontsize=14)
            ax6.set_xlabel('Rata-rata Kecepatan Angin (WSPM)')
            ax6.set_ylabel('Rata-rata PM2.5 (µg/m³)')
            ax6.legend(title='Segmen')
            ax6.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig6)
        else:
            st.warning("Tidak cukup data stasiun untuk membuat segmentasi.")


    with tab3:
        # --- ANALISIS LANJUTAN BARU 2: ANALISIS GEOSPASIAL ---
        st.subheader("Peta Sebaran Polusi PM2.5")
        
        # Data koordinat stasiun (harus dibuat manual)
        station_coords_data = {
            'station': ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 
                        'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong'],
            'latitude': [39.982, 40.217, 40.292, 39.929, 39.929, 39.914, 
                         40.328, 39.937, 40.127, 39.886, 39.942, 39.883],
            'longitude': [116.397, 116.225, 116.221, 116.417, 116.339, 116.184, 
                          116.628, 116.461, 116.655, 116.407, 116.287, 116.352]
        }
        station_coords = pd.DataFrame(station_coords_data)
        
        # Gabungkan data rata-rata PM2.5 dengan koordinat
        map_data = pd.merge(filtered_df.groupby('station')['PM2.5'].mean().reset_index(), station_coords, on='station')

        if not map_data.empty:
            # Buat peta yang berpusat di Beijing
            m = folium.Map(location=[39.9042, 116.4074], zoom_start=10)

            # Fungsi untuk memilih warna berdasarkan level PM2.5
            def get_color(pm25):
                if pm25 < 50: return 'green'
                elif pm25 < 100: return 'orange'
                else: return 'red'

            # Tambahkan penanda untuk setiap stasiun
            for idx, row in map_data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=10,
                    popup=f"<strong>{row['station']}</strong><br>Avg PM2.5: {row['PM2.5']:.2f} µg/m³",
                    color=get_color(row['PM2.5']),
                    fill=True,
                    fill_color=get_color(row['PM2.5']),
                    fill_opacity=0.7
                ).add_to(m)
            
            # Tampilkan peta di Streamlit
            folium_static(m)
        else:
            st.warning("Tidak ada data stasiun yang cocok dengan data koordinat untuk ditampilkan di peta.")


    # --- Kesimpulan ---
    st.divider()
    st.header("Kesimpulan")
    st.markdown("""
    - **Tren Polusi**: Secara umum, tingkat polusi PM2.5 menunjukkan pola musiman yang jelas, memuncak pada bulan-bulan musim dingin (November, Desember, Januari) dan menurun di musim panas. Hal ini kemungkinan besar disebabkan oleh kombinasi faktor emisi (pemanasan) dan kondisi meteorologi (angin lebih rendah, inversi suhu).
    - **Perbandingan Stasiun**: Terdapat variasi yang signifikan dalam tingkat polusi antar stasiun. Beberapa stasiun secara konsisten menunjukkan tingkat PM2.5 yang lebih tinggi, mengindikasikan adanya sumber polusi lokal yang dominan.
    - **Segmentasi Stasiun**:
        - Stasiun yang masuk dalam segmen **'Polusi Tinggi, Angin Rendah'** adalah yang paling mengkhawatirkan. Lokasi ini kemungkinan besar berada di area padat perkotaan dengan sirkulasi udara yang buruk, menyebabkan polutan terperangkap.
        - Stasiun di segmen **'Polusi Rendah, Angin Kencang'** menunjukkan kondisi terbaik, di mana kecepatan angin yang lebih tinggi efektif membantu menyebarkan polutan.
    - **Hubungan dengan Cuaca**: Analisis korelasi dan sebar menunjukkan bahwa **kecepatan angin (WSPM)** memiliki korelasi negatif yang paling signifikan dengan PM2.5. Artinya, angin yang lebih kencang cenderung menurunkan tingkat polusi udara.
    """)

except FileNotFoundError:
    st.error(f"File '{data_path}' tidak ditemukan. Pastikan file ada di lokasi yang benar.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses data: {e}")
    st.stop()

