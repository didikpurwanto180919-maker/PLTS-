import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Monitoring PLTS 1.5 MWp", layout="wide")

# Auto-refresh halaman setiap 60 detik agar otomatis mengikuti waktu real-time
st_autorefresh(interval=60 * 1000, key="plts_live_refresh")

st.title("☀️ Dashboard Monitoring Real-time PLTS 1.5 MWp")
st.markdown("**Lokasi:** Pasuruan (-7.6453, 112.9075) | **Sumber Data:** Global Solar Atlas & Open-Meteo")

# Parameter PLTS & Lokasi
LAT = -7.6453
LON = 112.9075
CAPACITY_MWP = 1.5  
TEMP_COEFF = -0.004 
NOCT = 45           
INVERTER_EFF = 0.85 

# Zona Waktu WIB (Asia/Jakarta)
wib_tz = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib_tz)
today_str = now_wib.strftime('%Y-%m-%d')

st.sidebar.header("Status Sistem Live")
st.sidebar.text(f"Waktu Server WIB:\n{now_wib.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.success("Auto-refresh aktif (tiap 60 detik)")

# Ambil Data Real-time dari API Open-Meteo
url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m,weathercode&timezone=auto&start_date={today_str}&end_date={today_str}"

response = requests.get(url)
data = response.json()

# Parsing Data ke DataFrame Pandas
hourly_data = data['hourly']
df = pd.DataFrame({
    'time': pd.to_datetime(hourly_data['time']),
    'ghi': hourly_data['shortwave_radiation'],
    'temp_ambient': hourly_data['temperature_2m'],
    'weather_code': hourly_data['weathercode']
})

if df['time'].dt.tz is None:
    df['time'] = df['time'].dt.tz_localize('Asia/Jakarta')
else:
    df['time'] = df['time'].dt.tz_convert('Asia/Jakarta')

# Kalkulasi Daya PLTS (MW)
def calculate_power(row):
    ghi = row['ghi']
    temp_amb = row['temp_ambient']
    
    if pd.isna(ghi) or ghi <= 0:
        return 0.0
    
    temp_cell = temp_amb + (NOCT - 20) * (ghi / 800.0)
    temp_factor = 1 + TEMP_COEFF * (temp_cell - 25)
    power_mw = CAPACITY_MWP * (ghi / 1000.0) * temp_factor * INVERTER_EFF
    
    return max(0.0, power_mw)

df['power_mw'] = df.apply(calculate_power, axis=1)
df['history_baseline_mw'] = df['ghi'] * (CAPACITY_MWP / 1000.0) * INVERTER_EFF

# Batasi Realisasi Grafik Hanya Sampai Jam Saat Ini (Mencegah titik masa depan muncul)
df_realtime = df.copy()
df_realtime.loc[df_realtime['time'].dt.hour > now_wib.hour, 'power_mw'] = None

# Visualisasi Grafik 1 Hari
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_realtime['time'].dt.strftime('%H:%M'), df_realtime['power_mw'], marker='o', color='orange', linewidth=2.5, label='Realisasi Real-time (MW)')
ax.plot(df['time'].dt.strftime('%H:%M'), df['history_baseline_mw'], linestyle='--', color='gray', alpha=0.7, label='Baseline History / Rencana (MW)')

ax.set_title(f'Grafik Produksi PLTS 1.5 MWp (Real-time WIB) - {today_str}', fontsize=12, fontweight='bold')
ax.set_xlabel('Waktu (Jam Lokal WIB)', fontsize=10)
ax.set_ylabel('Daya Output (MW)', fontsize=10)
plt.xticks(rotation=45)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')
plt.tight_layout()

st.pyplot(fig)

# Ringkasan Data Tabel Real-time
st.subheader(f"📋 Ringkasan Data Real-time (Update Terakhir: {now_wib.strftime('%H:%M')} WIB)")
df_display = df[df['time'].dt.hour <= now_wib.hour][['time', 'ghi', 'temp_ambient', 'power_mw']]
st.dataframe(df_display, use_container_width=True, hide_index=True)
