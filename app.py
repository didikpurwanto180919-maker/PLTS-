import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import pytz

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Monitoring PLTS Real-time", layout="centered")

st.title("☀️ Monitoring Real-time PLTS 1.5 MWp")
st.markdown("**Lokasi:** Pasuruan (-7.6453, 112.9075)")

# 1. Konfigurasi Parameter PLTS
LAT = -7.6453
LON = 112.9075
CAPACITY_MWP = 1.5  
TEMP_COEFF = -0.004 
NOCT = 45           
INVERTER_EFF = 0.85 

# 2. Sinkronisasi Waktu Lokal WIB
wib_tz = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib_tz)
today_str = now_wib.strftime('%Y-%m-%d')

st.info(f"📅 Tanggal: {today_str} | 🕒 Waktu Saat Ini: {now_wib.strftime('%H:%M')} WIB")

# 3. Ambil Data Real-time dari Open-Meteo API
url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m,weathercode&timezone=auto&start_date={today_str}&end_date={today_str}"

@st.cache_data(ttl=600) # Cache data selama 10 menit agar tidak spam API
def fetch_data(api_url):
    response = requests.get(api_url)
    return response.json()

data = fetch_data(url)

if 'hourly' in data:
    hourly_data = data['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly_data['time']),
        'ghi': hourly_data['shortwave_radiation'],
        'temp_ambient': hourly_data['temperature_2m']
    })

    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('Asia/Jakarta')
    else:
        df['time'] = df['time'].dt.tz_convert('Asia/Jakarta')

    # 4. Kalkulasi Daya PLTS
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

    # 5. Filter Data Real-time Hanya Sampai Jam Saat Ini
    df['realtime_power'] = df.apply(lambda row: row['power_mw'] if row['time'] <= now_wib else None, axis=1)

    # Format kolom waktu untuk sumbu X grafik & tabel
    df['jam_str'] = df['time'].dt.strftime('%H:%M')

    # 6. Visualisasi Grafik Menggunakan Streamlit Line Chart
    chart_data = df.set_index('jam_str')[['realtime_power', 'history_baseline_mw']]
    chart_data.columns = ['Realisasi Real-time (MW)', 'Baseline History / Rencana (MW)']
    
    st.line_chart(chart_data)

    # 7. Tabel Log Data Real-time
    st.subheader("📋 Log Data Real-time (Sampai Jam Saat Ini)")
    df_filtered = df[df['time'] <= now_wib][['jam_str', 'ghi', 'temp_ambient', 'realtime_power']]
    df_filtered.columns = ['Waktu (WIB)', 'Irradiance (GHI W/m²)', 'Suhu Ambien (°C)', 'Realisasi (MW)']
    st.dataframe(df_filtered.tail(12), use_container_width=True)

else:
    st.error("Gagal memuat data dari server cuaca.")
