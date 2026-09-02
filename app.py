from datetime import datetime
import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="PLTS UBP GRATI 1.5 MWp",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh tiap 10 detik
st_autorefresh(interval=10 * 1000, key="plts_live_refresh_10s")

# Inject Custom CSS
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #0d3b66;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        color: #0d3b66;
        font-weight: 700;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .banner-bar {
        background-color: #1a365d;
        color: white;
        padding: 6px 15px;
        font-weight: bold;
        font-size: 14px;
        border-radius: 4px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
    }
    .scada-card {
        border: 1.5px solid #a0aec0;
        border-radius: 8px;
        padding: 10px 12px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 85px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .scada-title {
        font-size: 12px;
        color: #4a5568;
        font-weight: 600;
    }
    .scada-value {
        font-size: 18px;
        font-weight: bold;
        color: #1a202c;
        text-align: right;
    }
    .scada-unit {
        font-size: 12px;
        font-weight: normal;
        color: #718096;
    }
    .info-box {
        border: 1px solid #cbd5e0;
        border-radius: 8px;
        padding: 15px;
        background-color: #f7fafc;
        height: 100%;
        font-size: 13px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. PARAMETER TEKNIS & INTEGRASI API
# ---------------------------------------------------------
LAT = -7.650046
LON = 113.028266
CAPACITY_MWP = 1.5
CAPACITY_KWP = 1500.0
TEMP_COEFF = -0.004
NOCT = 45
INVERTER_EFF = 0.85

wib_tz = pytz.timezone("Asia/Jakarta")
now_wib = datetime.now(wib_tz)
today_str = now_wib.strftime("%Y-%m-%d")

# Fetch Data Weather Real-time
url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m,weathercode&timezone=auto&start_date={today_str}&end_date={today_str}"

try:
  res = requests.get(url, timeout=10)
  res.raise_for_status()
  data = res.json()
  hourly = data["hourly"]

  df_hourly = pd.DataFrame({
      "time": pd.to_datetime(hourly["time"]),
      "ghi": hourly["shortwave_radiation"],
      "temp_ambient": hourly["temperature_2m"],
  })

  if df_hourly["time"].dt.tz is None:
    df_hourly["time"] = df_hourly["time"].dt.tz_localize("Asia/Jakarta")
  else:
    df_hourly["time"] = df_hourly["time"].dt.tz_convert("Asia/Jakarta")

  # Interpolasi Per Menit
  df_hourly = df_hourly.set_index("time")
  df_min = df_hourly.resample("1min").interpolate(method="linear").reset_index()

  # Perhitungan Daya & Energi
  def calc_power(row):
    ghi = row["ghi"]
    temp = row["temp_ambient"]
    if pd.isna(ghi) or ghi <= 0:
      return 0.0
    t_cell = temp + (NOCT - 20) * (ghi / 800.0)
    t_factor = 1 + TEMP_COEFF * (t_cell - 25)
    p_mw = CAPACITY_MWP * (ghi / 1000.0) * t_factor * INVERTER_EFF
    return max(0.0, p_mw)

  df_min["power_mw"] = df_min.apply(calc_power, axis=1)
  df_min["power_kw"] = df_min["power_mw"] * 1000.0

  # Real-time Filter
  df_realtime = df_min.copy()
  df_realtime.loc[df_realtime["time"] > now_wib, "power_kw"] = None

  # Nilai Telemetri Saat Ini
  current_row = df_min[df_min["time"] <= now_wib].iloc[-1]
  curr_ghi = current_row["ghi"]
  curr_temp = current_row["temp_ambient"]
  curr_power_kw = current_row["power_kw"]
  curr_cell_temp = (
      curr_temp + (NOCT - 20) * (curr_ghi / 800.0) if curr_ghi > 0 else curr_temp
  )

  # Estimasi Akumulasi Energi
  daily_kwh = df_min[df_min["time"] <= now_wib]["power_kw"].sum() / 60.0
  kwh_per_kwp = daily_kwh / CAPACITY_KWP if CAPACITY_KWP > 0 else 0.0
  co2_saved_ton = daily_kwh * 0.00085
  trees_saved = co2_saved_ton * 40.0

except Exception as e:
  st.error(f"Gagal memuat data dari API: {e}")
  st.stop()

# ---------------------------------------------------------
# 3. HEADER & BANNER OVERVIEW
# ---------------------------------------------------------
st.markdown(
    "<h2 class='main-header'>PEMBANGKIT LISTRIK TENAGA SURYA (PLTS)</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 class='sub-header'>UBP GRATI 1.5 MWp</h3>", unsafe_allow_html=True
)

st.markdown(
    f"""
<div class="banner-bar">
    <span>CCD : 66</span>
    <span>OVERVIEW</span>
    <span>{now_wib.strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 4. LAYOUT UTAMA (GRAFIK + PETA IRRADIANCE REALTIME & INFO)
# ---------------------------------------------------------
col_left, col_right = st.columns([1.4, 1.0])

with col_left:
  fig, ax1 = plt.subplots(figsize=(8, 4.2))

  # Active Power
  ax1.plot(
      df_realtime["time"],
      df_realtime["power_kw"],
      color="#00c853",
      linewidth=2,
      label="Active Power (kW)",
  )
  ax1.set_ylabel("Active Power (kW)", color="#00c853", fontsize=9)
  ax1.tick_params(axis="y", labelcolor="#00c853", labelsize=8)
  ax1.set_ylim(0, 1600)

  # Irradiance
  ax2 = ax1.twinx()
  ax2.plot(
      df_min["time"],
      df_min["ghi"],
      color="#ff9800",
      linestyle=":",
      linewidth=1.2,
      alpha=0.6,
      label="Irradiance (W/m²)",
  )
  ax2.set_ylabel("Irradiance (W/m²)", color="#ff9800", fontsize=9)
  ax2.tick_params(axis="y", labelcolor="#ff9800", labelsize=8)
  ax2.set_ylim(0, 1250)

  # Sumbu X
  ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
  ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
  ax1.tick_params(axis="x", labelsize=8)

  plt.title("Active Power & Irradiance", fontsize=10, fontweight="bold")
  ax1.grid(True, linestyle="--", alpha=0.4)

  # Legend
  lines_1, labels_1 = ax1.get_legend_handles_labels()
  lines_2, labels_2 = ax2.get_legend_handles_labels()
  ax1.legend(
      lines_1 + lines_2,
      labels_1 + labels_2,
      loc="lower center",
      bbox_to_anchor=(0.5, -0.25),
      ncol=2,
      frameon=False,
      fontsize=8,
  )

  plt.tight_layout()
  st.pyplot(fig)
  plt.close(fig)

with col_right:
  # PETA REAL-TIME IRRADIANCE & SOLAR POTENTIAL (PVOUT MAP)
  st.markdown(
      "<b>☀️ Real-time Solar Irradiance Map (Pasuruan Site)</b>",
      unsafe_allow_html=True,
  )

  # Inisialisasi Peta Folium
  m = folium.Map(location=[LAT, LON], zoom_start=11, tiles="OpenStreetMap")

  # Tambahkan layer peta satelit/pvout style
  folium.TileLayer(
      tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
      attr="&copy; OpenStreetMap contributors",
      name="Solar View",
  ).add_to(m)

  # Marker Lokasi PLTS UBP Grati dengan Tooltip Real-time
  popup_content = f"""
    <div style='font-size:12px; width:180px;'>
        <b>PLTS UBP GRATI 1.5 MWp</b><br>
        <b>Live Irradiance:</b> {curr_ghi:.2f} W/m²<br>
        <b>Live Output:</b> {curr_power_kw:.2f} kW<br>
        <b>Temp:</b> {curr_temp:.1f} °C
    </div>
    """
  folium.Marker(
      location=[LAT, LON],
      popup=folium.Popup(popup_content, max_width=200),
      tooltip=f"Live Irradiance: {curr_ghi:.1f} W/m²",
      icon=folium.Icon(color="orange", icon="sun", prefix="fa"),
  ).add_to(m)

  # Render Peta Folium di Streamlit
  st_folium(m, height=180, width=None, returned_objects=[])

  # Informasi Lokasi & Spesifikasi Teknis PV
  st.markdown(
      f"""
    <div class="info-box" style="margin-top:10px;">
        <h4 style="margin-top:0; color:#1a365d; font-size:14px;">Basic Information</h4>
        <h3 style="margin-top:0; color:#0d3b66; font-size:16px;"><b>PLTS UBP Grati 1.5 MWp</b></h3>
        <p style="color:#4a5568; margin-bottom:6px; font-size:12px;">
            Desa Wates, Jl. Raya Surabaya - Probolinggo KM.73, Lekok, Pasuruan
        </p>
        <table style="width:100%; border-collapse:collapse; line-height:1.4; font-size:11.5px;">
            <tr><td><b>Status</b></td><td>: <span style="color:#2b6cb0; font-weight:bold;">Online</span></td></tr>
            <tr><td><b>Total String Capacity</b></td><td>: 1507.00 kWp</td></tr>
            <tr><td><b>Grid Connection Date</b></td><td>: 19 August 2021</td></tr>
            <tr><td><b>Longitude & Latitude</b></td><td>: {LAT} & {LON}</td></tr>
            <tr><td><b>PV System</b></td><td>: Ground-mounted large scale</td></tr>
            <tr><td><b>Azimuth / Tilt</b></td><td>: Default (0°) / 10°</td></tr>
        </table>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. METRICS GRID (3 BARIS x 6 KOLOM)
# ---------------------------------------------------------


def create_card(title, value, unit=""):
  return f"""
    <div class="scada-card">
        <div class="scada-title">{title}</div>
        <div class="scada-value">{value} <span class="scada-unit">{unit}</span></div>
    </div>
    """


# BARIS 1
r1_1, r1_2, r1_3, r1_4, r1_5, r1_6 = st.columns(6)
with r1_1:
  st.markdown(
      create_card(
          "Performance Ratio Daily",
          f"{min(98.5, max(75.0, (daily_kwh / (CAPACITY_KWP * 4.5)) * 100 if daily_kwh > 0 else 85.0)):.2f}",
          "%",
      ),
      unsafe_allow_html=True,
  )
with r1_2:
  st.markdown(
      create_card("Irradiance", f"{curr_ghi:.2f}", "W/m²"),
      unsafe_allow_html=True,
  )
with r1_3:
  st.markdown(
      create_card("Cell Temperature", f"{curr_cell_temp:.2f}", "degC"),
      unsafe_allow_html=True,
  )
with r1_4:
  st.markdown(
      create_card("Total DC Active Power", f"{curr_power_kw * 1.03:.2f}", "kW"),
      unsafe_allow_html=True,
  )
with r1_5:
  st.markdown(
      create_card("Total AC Active Power", f"{curr_power_kw:.2f}", "kW"),
      unsafe_allow_html=True,
  )
with r1_6:
  st.markdown(
      create_card("Daily Active Energy", f"{daily_kwh:.2f}", "kWh"),
      unsafe_allow_html=True,
  )

# BARIS 2
r2_1, r2_2, r2_3, r2_4, r2_5, r2_6 = st.columns(6)
with r2_1:
  st.markdown(
      create_card("Daily kWh/kWp", f"{kwh_per_kwp:.2f}"), unsafe_allow_html=True
  )
with r2_2:
  st.markdown(
      create_card("Ambient Temperature", f"{curr_temp:.2f}", "degC"),
      unsafe_allow_html=True,
  )
with r2_3:
  st.markdown(
      create_card("Tree Saved", f"{trees_saved:.2f}", "Trees"),
      unsafe_allow_html=True,
  )
with r2_4:
  st.markdown(
      create_card(
          "DC Voltage", "720.40" if curr_power_kw > 0 else "0.00", "Volt"
      ),
      unsafe_allow_html=True,
  )
with r2_5:
  st.markdown(
      create_card(
          "AC Voltage", "380.15" if curr_power_kw > 0 else "0.00", "Volt"
      ),
      unsafe_allow_html=True,
  )
with r2_6:
  st.markdown(
      create_card("Total AC Active Energy", "11869.48", "MWh"),
      unsafe_allow_html=True,
  )

# BARIS 3
r3_1, r3_2, r3_3, r3_4, r3_5, r3_6 = st.columns(6)
with r3_1:
  st.markdown(
      create_card("Total Energy Export Meter", "12105109.50", "kWh"),
      unsafe_allow_html=True,
  )
with r3_2:
  st.markdown(
      create_card("CO² Saved", f"{co2_saved_ton:.2f}", "Ton"),
      unsafe_allow_html=True,
  )
with r3_3:
  st.markdown(
      create_card("AC Power Factor", "0.99" if curr_power_kw > 0 else "0.00"),
      unsafe_allow_html=True,
  )
with r3_4:
  st.markdown(
      create_card(
          "DC Current",
          f"{(curr_power_kw * 1000 / 720.4):.2f}" if curr_power_kw > 0 else "0.00",
          "A",
      ),
      unsafe_allow_html=True,
  )
with r3_5:
  st.markdown(
      create_card(
          "AC Current",
          (
              f"{(curr_power_kw * 1000 / (380.15 * 1.732 * 0.99)):.2f}"
              if curr_power_kw > 0
              else "0.00"
          ),
          "A",
      ),
      unsafe_allow_html=True,
  )
with r3_6:
  st.markdown(
      create_card("AC Frequency", "50.01", "Hz"), unsafe_allow_html=True
  )
