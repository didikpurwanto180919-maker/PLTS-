# 1. Install library pendukung & Git
!pip install requests pandas matplotlib pytz GitPython

import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import os
from git import Repo

# 2. Konfigurasi Parameter PLTS & Lokasi (Global Solar Atlas Pasuruan)
LAT = -7.6453
LON = 112.9075
CAPACITY_MWP = 1.5  # Kapasitas PLTS dalam MWp
TEMP_COEFF = -0.004 # Koefisien suhu panel
NOCT = 45           # Normal Operating Cell Temperature (°C)
INVERTER_EFF = 0.85 # Efisiensi total sistem

# 3. Sinkronisasi Zona Waktu ke WIB (Asia/Jakarta)
wib_tz = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(wib_tz)
today_str = now_wib.strftime('%Y-%m-%d')

print(f"Waktu Sistem Saat Ini (WIB): {now_wib.strftime('%Y-%m-%d %H:%M:%S')}")

# 4. Ambil Data Real-time / Forecast Hari Ini dari Open-Meteo API
url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m,weathercode&timezone=auto&start_date={today_str}&end_date={today_str}"

response = requests.get(url)
data = response.json()

# 5. Parsing Data ke DataFrame Pandas
hourly_data = data['hourly']
df = pd.DataFrame({
    'time': pd.to_datetime(hourly_data['time']),
    'ghi': hourly_data['shortwave_radiation'],
    'temp_ambient': hourly_data['temperature_2m'],
    'weather_code': hourly_data['weathercode']
})

# Konversi timezone DataFrame ke WIB
if df['time'].dt.tz is None:
    df['time'] = df['time'].dt.tz_localize('Asia/Jakarta')
else:
    df['time'] = df['time'].dt.tz_convert('Asia/Jakarta')

# 6. Kalkulasi Produksi Daya PLTS (MW)
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

# 7. Batasi Realisasi Hanya Sampai Jam Saat Ini (Mencegah jam masa depan muncul)
df_realtime = df.copy()
df_realtime.loc[df_realtime['time'] > now_wib, 'power_mw'] = None

# 8. Visualisasi Grafik 1 Hari & Simpan Gambar
plt.figure(figsize=(12, 6))
plt.plot(df_realtime['time'].dt.strftime('%H:%M'), df_realtime['power_mw'], marker='o', color='orange', linewidth=2.5, label='Realisasi Real-time (MW)')
plt.plot(df['time'].dt.strftime('%H:%M'), df['history_baseline_mw'], linestyle='--', color='gray', alpha=0.7, label='Baseline History / Rencana (MW)')

plt.title(f'Monitoring Produksi PLTS 1.5 MWp (Real-time WIB) - {today_str}\nLokasi: Pasuruan (-7.6453, 112.9075)', fontsize=14, fontweight='bold')
plt.xlabel('Waktu (Jam Lokal WIB)', fontsize=12)
plt.ylabel('Daya Output (MW)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()

# Simpan grafik ke file gambar untuk kebutuhan repository GitHub
output_image_path = 'plts_realtime_monitoring.png'
plt.savefig(output_image_path, dpi=300)
plt.show()

# 9. Tampilkan Ringkasan Data Tabel Real-time
print(f"\n--- RINGKASAN DATA REAL-TIME (Sampai Jam {now_wib.strftime('%H:%M')} WIB) ---")
df_display = df[df['time'] <= now_wib]
print(df_display[['time', 'ghi', 'temp_ambient', 'power_mw']].to_string(index=False))

# 10. (Opsional) Otomatis Push ke GitHub Repository Anda
# Pastikan Anda sudah melakukan autentikasi token GitHub di Colab jika ingin mengaktifkan blok ini
# REPO_URL = "https://github.com/didikpurwanto180919-maker/PLTS-.git"
# local_dir = "/content/PLTS-"
# if not os.path.exists(local_dir):
#     Repo.clone_from(REPO_URL, local_dir)
# repo = Repo(local_dir)
# os.system(f"cp {output_image_path} {local_dir}/")
# repo.git.add(update=True)
# repo.index.commit(f"Update real-time PLTS chart: {now_wib.strftime('%Y-%m-%d %H:%M')}")
# repo.remotes.origin.push()
# print("Berhasil memperbarui data dan grafik ke GitHub!")
