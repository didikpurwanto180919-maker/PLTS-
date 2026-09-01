import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Monitoring PLTS 1.5 MWp", layout="wide")

# Auto-refresh halaman setiap 1 detik
st_autorefresh(interval=1 * 1000, key="plts_live_refresh")

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
st.sidebar.success("Auto-refresh aktif (tiap 1 detik)")

# Ambil Data Real-time dari API Open-Meteo
url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m,weathercode&timezone=auto&start_date={today_str}&end_date={today_str}"

response = requests.get(url)
data = response.json()

# Parsing Data Hourly
hourly_data = data['hourly']
df_hourly = pd.DataFrame({
    'time': pd.to_datetime(hourly_data['time']),
    'ghi': hourly_data['shortwave_radiation'],
    'temp_ambient': hourly_data['temperature_2m'],
    'weather_code': hourly_data['weathercode']
})

if df_hourly['time'].dt.tz is None:
    df_hourly['time'] = df_hourly['time'].dt.tz_localize('Asia/Jakarta')
else:
    df_hourly['time'] = df_hourly['time'].dt.tz_convert('Asia/Jakarta')

# Interpolasi Menjadi Per Menit agar Garis Bergerak Halus Real-time
df_hourly = df_hourly.set_index('time')
df_minutely = df_hourly.resample('1min').interpolate(method='linear').reset_index()

# Kalkulasi Daya PLTS (MW) untuk Setiap Menit
def calculate_power(row):
    ghi = row['ghi']
    temp_amb = row['temp_ambient']
    
    if pd.isna(ghi) or ghi <= 0:
        return 0.0
    
    temp_cell = temp_amb + (NOCT - 20) * (ghi / 800.0)
    temp_factor = 1 + TEMP_COEFF * (temp_cell - 25)
    power_mw = CAPACITY_MWP * (ghi / 1000.0) * temp_factor * INVERTER_EFF
    
    return max(0.0, power_mw)

df_minutely['power_mw'] = df_minutely.apply(calculate_power, axis=1)
df_minutely['history_baseline_mw'] = df_minutely['ghi'] * (CAPACITY_MWP / 1000.0) * INVERTER_EFF

# Batasi Realisasi Hanya Sampai Menit Saat Ini
df_realtime = df_minutely.copy()
df_realtime.loc[df_realtime['time'] > now_wib, 'power_mw'] = None

# Visualisasi Grafik 1 Hari dengan Format Waktu Bersih (Jam:Menit)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_realtime['time'], df_realtime['power_mw'], color='orange', linewidth=2.5, label='Realisasi Real-time (MW)')
ax.plot(df_minutely['time'], df_minutely['history_baseline_mw'], linestyle='--', color='gray', alpha=0.7, label='Baseline History / Rencana (MW)')

# Mengatur Format Sumbu X agar Menampilkan Jam:Menit secara Rapi
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=wib_tz))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

ax.set_title(f'Grafik Produksi PLTS 1.5 MWp (Real-time WIB) - {today_str}', fontsize=12, fontweight='bold')
ax.set_xlabel('Waktu (Jam Lokal WIB)', fontsize=10)
ax.set_ylabel('Daya Output (MW)', fontsize=10)
plt.xticks(rotation=0)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')
plt.tight_layout()

st.pyplot(fig)

# Ringkasan Data Tabel Real-time
st.subheader(f"📋 Ringkasan Data Real-time (Update Terakhir: {now_wib.strftime('%H:%M:%S')} WIB)")
df_display_hourly = df_hourly.reset_index()
df_display_hourly['power_mw'] = df_display_hourly.apply(calculate_power, axis=1)
df_display = df_display_hourly[df_display_hourly['time'] <= now_wib][['time', 'ghi', 'temp_ambient', 'power_mw']]
st.dataframe(df_display, use_container_width=True, hide_index=True)
