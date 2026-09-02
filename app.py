from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS STYLING (DARK SCADA THEME)
# ---------------------------------------------------------
st.set_page_config(
    page_title="PLTS UBP GRATI 1.5 MWp",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh tiap 10 detik
st_autorefresh(interval=10 * 1000, key="plts_live_refresh_10s")

# Inject Custom CSS Warna Menarik (Dark High-Contrast SCADA)
st.markdown(
    """
<style>
    /* Background Utama */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    .main-header {
        text-align: center;
        color: #38bdf8;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }
    .sub-header {
        text-align: center;
        color: #f59e0b;
        font-weight: 700;
        margin-top: -5px;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
    }
    .banner-bar {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 50%, #1e293b 100%);
        border: 1px solid #334155;
        color: #38bdf8;
        padding: 8px 18px;
        font-weight: bold;
        font-size: 14px;
        border-radius: 6px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .scada-card {
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 10px 14px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        height: 85px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 12px;
        transition: transform 0.2s;
    }
    .scada-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .scada-title {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .scada-value {
        font-size: 20px;
        font-weight: bold;
        color: #f8fafc;
        text-align: right;
        font-family: 'Courier New', Courier, monospace;
    }
    .scada-unit {
        font-size: 12px;
        font-weight: normal;
        color: #38bdf8;
    }
    .info-box {
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        height: 100%;
        font-size: 13px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .info-table td {
        padding: 4px 0;
        color: #cbd5e1;
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

  df_hourly = df_hourly.set_index("time")
  df_min = df_hourly.resample("1min").interpolate(method="linear").reset_index()

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

  df_realtime = df_min.copy()
  df_realtime.loc[df_realtime["time"] > now_wib, "power_kw"] = None

  current_row = df_min[df_min["time"] <= now_wib].iloc[-1]
  curr_ghi = current_row["ghi"]
  curr_temp = current_row["temp_ambient"]
  curr_power_kw = current_row["power_kw"]
  curr_cell_temp = (
      curr_temp + (NOCT - 20) * (curr_ghi / 800.0) if curr_ghi > 0 else curr_temp
  )

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
    <span><span style="color:#94a3b8;">CCD :</span> <b style="color:#38bdf8;">66</b></span>
    <span style="color:#f8fafc; letter-spacing:2px;">OVERVIEW MONITORING</span>
    <span>{now_wib.strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 4. GRAFIK DENGAN WARNA NEON HIGH-CONTRAST
# ---------------------------------------------------------
col_left, col_right = st.columns([1.4, 1.0])

with col_left:
  # Konfigurasi Tema Gelap Matplotlib
  plt.style.use("dark_background")
  fig, ax1 = plt.subplots(figsize=(8, 4.2))
  fig.patch.set_facecolor("#0b0f19")
  ax1.set_facecolor("#0f172a")

  # Active Power (Cyan Neon)
  (line1,) = ax1.plot(
      df_realtime["time"],
      df_realtime["power_kw"],
      color="#00f2fe",
      linewidth=2.5,
      label="Active Power (kW)",
  )
  ax1.fill_between(
      df_realtime["time"],
      df_realtime["power_kw"],
      color="#00f2fe",
      alpha=0.15,
  )
  ax1.set_ylabel("Active Power (kW)", color="#00f2fe", fontsize=9, weight="bold")
  ax1.tick_params(axis="y", labelcolor="#00f2fe", labelsize=8)
  ax1.set_ylim(0, 1600)

  # Irradiance (Amber Gold Neon)
  ax2 = ax1.twinx()
  (line2,) = ax2.plot(
      df_min["time"],
      df_min["ghi"],
      color="#fbbf24",
      linestyle="--",
      linewidth=1.8,
      alpha=0.9,
      label="Irradiance (W/m²)",
  )
  ax2.set_ylabel(
      "Irradiance (W/m²)", color="#fbbf24", fontsize=9, weight="bold"
  )
  ax2.tick_params(axis="y", labelcolor="#fbbf24", labelsize=8)
  ax2.set_ylim(0, 1250)

  # Formatting Sumbu & Grid
  ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
  ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
  ax1.tick_params(axis="x", labelsize=8, colors="#94a3b8")
  ax1.spines["top"].set_visible(False)
  ax2.spines["top"].set_visible(False)
  ax1.spines["left"].set_color("#334155")
  ax2.spines["right"].set_color("#334155")

  plt.title(
      "Active Power & Irradiance Trend",
      fontsize=11,
      fontweight="bold",
      color="#f8fafc",
      pad=12,
  )
  ax1.grid(True, linestyle=":", alpha=0.2, color="#94a3b8")

  # Legend Style
  ax1.legend(
      [line1, line2],
      ["Active Power (kW)", "Irradiance (W/m²)"],
      loc="lower center",
      bbox_to_anchor=(0.5, -0.24),
      ncol=2,
      frameon=False,
      fontsize=8,
  )

  plt.tight_layout()
  st.pyplot(fig)
  plt.close(fig)

with col_right:
  st.markdown(
      f"""
    <div class="info-box">
        <h4 style="margin-top:0; color:#38bdf8; font-size:14px; text-transform:uppercase; letter-spacing:1px;">Basic Information</h4>
        <h3 style="margin-top:0; color:#f8fafc; font-size:16px;"><b>PLTS UBP Grati 1.5 MWp</b></h3>
        <p style="color:#94a3b8; margin-bottom:12px; font-size:12px; line-height:1.4;">
            Desa Wates, Jl. Raya Surabaya - Probolinggo KM.73<br>
            Lekok, Pasir Panjang, Wates, Kec. Lekok, Pasuruan<br>
            Jawa Timur 67186
        </p>
        <table class="info-table" style="width:100%; border-collapse:collapse; font-size:12px;">
            <tr><td><b>Status</b></td><td>: <span style="background-color:#166534; color:#4ade80; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:11px;">● ONLINE</span></td></tr>
            <tr><td><b>Total String Capacity</b></td><td>: <span style="color:#f8fafc;">1507.00 kWp</span></td></tr>
            <tr><td><b>Grid Connection Date</b></td><td>: <span style="color:#f8fafc;">19 August 2021</span></td></tr>
            <tr><td><b>Longitude & Latitude</b></td><td>: <span style="color:#f8fafc;">{LAT} & {LON}</span></td></tr>
            <tr><td><b>PV System</b></td><td>: <span style="color:#f8fafc;">Ground-mounted large scale</span></td></tr>
            <tr><td><b>Azimuth of PV Panels</b></td><td>: <span style="color:#f8fafc;">Default (0°)</span></td></tr>
            <tr><td><b>Tilt of PV Panels</b></td><td>: <span style="color:#f8fafc;">10°</span></td></tr>
        </table>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. METRICS GRID (DARK SCADA CARDS)
# ---------------------------------------------------------


def create_card(title, value, unit="", border_color="#38bdf8"):
  return f"""
    <div class="scada-card" style="border-left-color: {border_color};">
        <div class="scada-title">{title}</div>
        <div class="scada-value">{value} <span class="scada-unit">{unit}</span></div>
    </div>
    """


# BARIS 1
r1_1, r1_2, r1_3, r1_4, r1_5, r1_6 = st.columns(6)
with r1_1:
  st.markdown(
      create_card(
          "PR Daily",
          f"{min(98.5, max(75.0, (daily_kwh / (CAPACITY_KWP * 4.5)) * 100 if daily_kwh > 0 else 85.0)):.2f}",
          "%",
          "#4ade80",
      ),
      unsafe_allow_html=True,
  )
with r1_2:
  st.markdown(
      create_card("Irradiance", f"{curr_ghi:.2f}", "W/m²", "#fbbf24"),
      unsafe_allow_html=True,
  )
with r1_3:
  st.markdown(
      create_card(
          "Cell Temperature", f"{curr_cell_temp:.2f}", "°C", "#f97316"
      ),
      unsafe_allow_html=True,
  )
with r1_4:
  st.markdown(
      create_card(
          "Total DC Power", f"{curr_power_kw * 1.03:.2f}", "kW", "#38bdf8"
      ),
      unsafe_allow_html=True,
  )
with r1_5:
  st.markdown(
      create_card("Total AC Power", f"{curr_power_kw:.2f}", "kW", "#00f2fe"),
      unsafe_allow_html=True,
  )
with r1_6:
  st.markdown(
      create_card("Daily Energy", f"{daily_kwh:.2f}", "kWh", "#a855f7"),
      unsafe_allow_html=True,
  )

# BARIS 2
r2_1, r2_2, r2_3, r2_4, r2_5, r2_6 = st.columns(6)
with r2_1:
  st.markdown(
      create_card("Daily kWh/kWp", f"{kwh_per_kwp:.2f}", "", "#4ade80"),
      unsafe_allow_html=True,
  )
with r2_2:
  st.markdown(
      create_card("Ambient Temp", f"{curr_temp:.2f}", "°C", "#f97316"),
      unsafe_allow_html=True,
  )
with r2_3:
  st.markdown(
      create_card("Trees Saved", f"{trees_saved:.2f}", "Trees", "#22c55e"),
      unsafe_allow_html=True,
  )
with r2_4:
  st.markdown(
      create_card(
          "DC Voltage",
          "720.40" if curr_power_kw > 0 else "0.00",
          "V",
          "#38bdf8",
      ),
      unsafe_allow_html=True,
  )
with r2_5:
  st.markdown(
      create_card(
          "AC Voltage",
          "380.15" if curr_power_kw > 0 else "0.00",
          "V",
          "#00f2fe",
      ),
      unsafe_allow_html=True,
  )
with r2_6:
  st.markdown(
      create_card("Total AC Energy", "11869.48", "MWh", "#a855f7"),
      unsafe_allow_html=True,
  )

# BARIS 3
r3_1, r3_2, r3_3, r3_4, r3_5, r3_6 = st.columns(6)
with r3_1:
  st.markdown(
      create_card(
          "Export Meter", "12105109.50", "kWh", "#a855f7"
      ),
      unsafe_allow_html=True,
  )
with r3_2:
  st.markdown(
      create_card("CO² Saved", f"{co2_saved_ton:.2f}", "Ton", "#22c55e"),
      unsafe_allow_html=True,
  )
with r3_3:
  st.markdown(
      create_card(
          "AC Power Factor",
          "0.99" if curr_power_kw > 0 else "0.00",
          "",
          "#eab308",
      ),
      unsafe_allow_html=True,
  )
with r3_4:
  st.markdown(
      create_card(
          "DC Current",
          f"{(curr_power_kw * 1000 / 720.4):.2f}" if curr_power_kw > 0 else "0.00",
          "A",
          "#38bdf8",
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
          "#00f2fe",
      ),
      unsafe_allow_html=True,
  )
with r3_6:
  st.markdown(
      create_card("AC Frequency", "50.01", "Hz", "#eab308"),
      unsafe_allow_html=True,
  )
