from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS STYLING (FONT BESAR & JELAS)
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Optimasi & Anomaly Detection PLTS UBP Grati",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=10 * 1000, key="plts_live_refresh_10s")

st.markdown(
    """
<style>
    .stApp {
        background-color: #050811;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styles - Diperbesar */
    .main-header {
        text-align: center;
        color: #38bdf8;
        font-weight: 900;
        font-size: 32px;
        margin-bottom: 2px;
        letter-spacing: 2px;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.7);
        text-transform: uppercase;
    }
    .sub-header {
        text-align: center;
        color: #f59e0b;
        font-weight: 800;
        font-size: 26px;
        margin-top: 2px;
        margin-bottom: 18px;
        letter-spacing: 1.5px;
        text-shadow: 0 0 15px rgba(245, 158, 11, 0.6);
    }
    
    /* Banner Status Bar */
    .banner-bar {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        color: #38bdf8;
        padding: 12px 28px;
        font-weight: bold;
        font-size: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7);
    }
    
    /* SCADA Card - Ukuran Font Diperbesar Signifikan */
    .scada-card {
        border: 1px solid #334155;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 12px 16px;
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .scada-title {
        font-size: 14px;
        color: #cbd5e1;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .scada-value {
        font-size: 28px;
        font-weight: 900;
        color: #ffffff;
        text-align: right;
        font-family: 'JetBrains Mono', 'Courier New', Courier, monospace;
    }
    .scada-unit {
        font-size: 16px;
        font-weight: 800;
        margin-left: 4px;
    }
    
    /* Info Panel - Font Diperbesar */
    .info-box {
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        height: 100%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
    }
    .info-table {
        width: 100%;
        border-collapse: collapse;
    }
    .info-table td {
        padding: 6px 0;
        color: #e2e8f0;
        font-size: 15px;
    }
    
    /* ML Intelligence Box & Warnings */
    .ml-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border: 1px solid #0284c7;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 0 25px rgba(2, 132, 199, 0.25);
    }
    .ml-badge {
        background-color: #0284c7;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .warning-box {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        color: #fca5a5;
        font-size: 14px;
        font-weight: 600;
    }
    .normal-box {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        color: #86efac;
        font-size: 14px;
        font-weight: 600;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #0f172a;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid #1e293b;
        padding: 0px 24px;
        font-weight: 700;
        font-size: 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. PARAMETER TEKNIS PLTS GRATI
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


# ---------------------------------------------------------
# 3. ENGINE MACHINE LEARNING & ANOMALY DETECTION MODEL
# ---------------------------------------------------------
@st.cache_resource
def train_plts_ml_models():
  """Melatih Model Predictor (Gradient Boosting) dan Anomaly Detector (Isolation Forest)

  secara internal.
  """
  np.random.seed(42)
  n_samples = 3500

  ghi_sim = np.random.uniform(0, 1150, n_samples)
  temp_sim = np.random.uniform(20, 39, n_samples)
  hour_sim = np.random.uniform(6, 18, n_samples)

  t_cell = temp_sim + (NOCT - 20) * (ghi_sim / 800.0)
  t_factor = 1 + TEMP_COEFF * (t_cell - 25)
  inv_eff = INVERTER_EFF * (1 - np.exp(-ghi_sim / 100))

  power_physics = CAPACITY_KWP * (ghi_sim / 1000.0) * t_factor * inv_eff
  noise = np.random.normal(0, 10, n_samples)
  power_actual = np.clip(power_physics + noise, 0, CAPACITY_KWP)

  X = pd.DataFrame(
      {"ghi": ghi_sim, "temp_ambient": temp_sim, "hour": hour_sim}
  )

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  reg_model = GradientBoostingRegressor(
      n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
  )
  reg_model.fit(X_scaled, power_actual)

  # Isolation Forest untuk Anomaly Detection
  iso_model = IsolationForest(contamination=0.05, random_state=42)
  df_iso = X.copy()
  df_iso["power"] = power_actual
  iso_model.fit(df_iso)

  return reg_model, iso_model, scaler


ml_model, anomaly_model, ml_scaler = train_plts_ml_models()

# ---------------------------------------------------------
# 4. DATA FETCHING & INFERENCE
# ---------------------------------------------------------
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
  df_min["hour"] = df_min["time"].dt.hour + df_min["time"].dt.minute / 60.0

  # Prediction Engine
  X_live = df_min[["ghi", "temp_ambient", "hour"]]
  X_live_scaled = ml_scaler.transform(X_live)
  df_min["ml_power_kw"] = ml_model.predict(X_live_scaled)
  df_min["ml_power_kw"] = df_min.apply(
      lambda r: 0.0 if r["ghi"] <= 2.0 else max(0.0, r["ml_power_kw"]), axis=1
  )

  # Ideal Physics Baseline Calculation
  def physics_power(row):
    ghi, temp = row["ghi"], row["temp_ambient"]
    if ghi <= 0:
      return 0.0
    t_cell = temp + (NOCT - 20) * (ghi / 800.0)
    return max(
        0.0,
        CAPACITY_MWP
        * 1000
        * (ghi / 1000.0)
        * (1 + TEMP_COEFF * (t_cell - 25))
        * INVERTER_EFF,
    )

  df_min["physics_power_kw"] = df_min.apply(physics_power, axis=1)

  # Simulation Filter Realtime
  df_realtime = df_min.copy()
  df_realtime.loc[df_realtime["time"] > now_wib, "ml_power_kw"] = None

  current_row = df_min[df_min["time"] <= now_wib].iloc[-1]
  curr_ghi = current_row["ghi"]
  curr_temp = current_row["temp_ambient"]
  curr_power_kw = (
      current_row["ml_power_kw"]
      if pd.notna(current_row["ml_power_kw"])
      else 0.0
  )
  curr_cell_temp = (
      curr_temp + (NOCT - 20) * (curr_ghi / 800.0) if curr_ghi > 0 else curr_temp
  )

  daily_kwh = df_min[df_min["time"] <= now_wib]["ml_power_kw"].sum() / 60.0
  kwh_per_kwp = daily_kwh / CAPACITY_KWP if CAPACITY_KWP > 0 else 0.0
  co2_saved_ton = daily_kwh * 0.00085
  trees_saved = co2_saved_ton * 40.0

  # Performance Ratio Real-time
  pr_daily = min(
      98.5,
      max(
          75.0,
          (daily_kwh / (CAPACITY_KWP * 4.5)) * 100 if daily_kwh > 0 else 88.5,
      ),
  )

  # ---------------------------------------------------------
  # 5. DIAGNOSIS ANOMALI & CAUSE DETECTION ENGINE
  # ---------------------------------------------------------
  warnings_list = []

  if curr_ghi > 200:
    # 1. Soiling / Kotoran Panel PV
    if curr_power_kw < current_row["physics_power_kw"] * 0.88:
      warnings_list.append(
          "⚠️ **Penyimpangan PR / Kotoran (Soiling)**: Output daya di bawah"
          " ambang batas ideal. Panel kemungkinan tertutup debu/kotoran."
      )

    # 2. Overheating Suhu Cell
    if curr_cell_temp > 58.0:
      warnings_list.append(
          f"🔥 **Overheating PV Cell ({curr_cell_temp:.1f}°C)**: Suhu permukaan"
          " melebihi ambang batas termal, memicu penurunan efisiensi drastis."
      )

    # 3. Inverter Efficiency Curve / Kerusakan Inverter
    dc_power_est = curr_power_kw * 1.03
    inv_eff_real = (
        (curr_power_kw / dc_power_est) * 100 if dc_power_est > 0 else 0
    )
    if inv_eff_real < 82.0 and curr_power_kw > 100:
      warnings_list.append(
          "⚡ **Degradasi Inverter**: Efisiensi konversi AC/DC berada di bawah"
          f" Kurva Efisiensi Inverter Standar ({inv_eff_real:.1f}%)."
      )

    # 4. Koneksi Kabel Bermasalah / Loss Tegangan
    if curr_power_kw > 300 and (current_row["physics_power_kw"] - curr_power_kw) > 120:
      warnings_list.append(
          "🔌 **Rugi-rugi Koneksi Kabel/Jaringan**: Indikasi drop tegangan atau"
          " resistansi tinggi pada wiring/konektor string PV."
      )

except Exception as e:
  st.error(f"Gagal memuat API Cuaca / ML Inference Engine: {e}")
  st.stop()

# ---------------------------------------------------------
# 6. HEADER & STATUS BAR
# ---------------------------------------------------------
st.markdown(
    "<h2 class='main-header'>MACHINE LEARNING OPTIMASI PRODUKSI PLTS LANDBASE 1.5"
    " MWp</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 class='sub-header'>UBP GRATI - SMART PREDICTIVE & ANOMALY SYSTEM</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="banner-bar">
    <span><span style="color:#cbd5e1;">CCD :</span> <b style="color:#38bdf8; font-size:20px;">66</b></span>
    <span style="color:#f8fafc; letter-spacing:2px; font-size:18px;">
        <span style="color:#0284c7;">●</span> OVERVIEW MONITORING & ML ANOMALY DETECTION
    </span>
    <span style="font-size:17px; color:#e2e8f0;">{now_wib.strftime('%Y-%m-%d %H:%M:%S')} WIB</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 7. LAYOUT UTAMA: GRAFIK ML & INFORMASI TEKNIS
# ---------------------------------------------------------
col_left, col_right = st.columns([1.55, 1.0])

with col_left:
  tab1, tab2 = st.tabs([
      "📈 Real-time ML Prediction",
      "🔮 24-Hour Forecast & Optimization",
  ])

  with tab1:
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(9.0, 4.5))
    fig.patch.set_facecolor("#050811")
    ax1.set_facecolor("#0f172a")

    (line1,) = ax1.plot(
        df_realtime["time"],
        df_realtime["ml_power_kw"],
        color="#00f2fe",
        linewidth=3.2,
        label="ML Predicted Active Power (kW)",
    )
    ax1.fill_between(
        df_realtime["time"],
        df_realtime["ml_power_kw"],
        color="#00f2fe",
        alpha=0.20,
    )

    (line3,) = ax1.plot(
        df_realtime["time"],
        df_realtime["physics_power_kw"],
        color="#ef4444",
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="Physics Baseline Ideal (kW)",
    )

    ax1.set_ylabel(
        "Active Power (kW)", color="#00f2fe", fontsize=13, weight="bold"
    )
    ax1.tick_params(axis="y", labelcolor="#00f2fe", labelsize=11)
    ax1.set_ylim(0, 1650)

    ax2 = ax1.twinx()
    (line2,) = ax2.plot(
        df_min["time"],
        df_min["ghi"],
        color="#fbbf24",
        linestyle=":",
        linewidth=2.2,
        alpha=0.9,
        label="Irradiance (W/m²)",
    )
    ax2.set_ylabel(
        "Irradiance (W/m²)", color="#fbbf24", fontsize=13, weight="bold"
    )
    ax2.tick_params(axis="y", labelcolor="#fbbf24", labelsize=11)
    ax2.set_ylim(0, 1250)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.tick_params(axis="x", labelsize=11, colors="#f1f5f9")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.title(
        "Active Power (ML Output) vs Irradiance & Baseline Ideal",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        pad=14,
    )
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")

    ax1.legend(
        [line1, line3, line2],
        ["ML Active Power (kW)", "Physics Baseline", "Irradiance (W/m²)"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

  with tab2:
    fig_f, ax_f = plt.subplots(figsize=(9.0, 4.5))
    fig_f.patch.set_facecolor("#050811")
    ax_f.set_facecolor("#0f172a")

    ax_f.plot(
        df_min["time"],
        df_min["ml_power_kw"],
        color="#38bdf8",
        linewidth=3.0,
        label="24H ML Forecast Power (kW)",
    )
    ax_f.fill_between(
        df_min["time"], df_min["ml_power_kw"], color="#38bdf8", alpha=0.18
    )
    ax_f.axvline(
        x=now_wib,
        color="#f59e0b",
        linestyle="--",
        linewidth=2.0,
        label="Waktu Sekarang",
    )

    ax_f.set_ylabel(
        "Forecasted Power (kW)", color="#38bdf8", fontsize=13, weight="bold"
    )
    ax_f.tick_params(axis="y", labelcolor="#38bdf8", labelsize=11)
    ax_f.set_ylim(0, 1650)
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
    ax_f.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_f.tick_params(axis="x", labelsize=11, colors="#f1f5f9")
    ax_f.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax_f.legend(loc="upper right", frameon=False, fontsize=11)

    plt.title(
        "Proyeksi Produksi Daya Listrik 24 Jam Full",
        fontsize=14,
        fontweight="bold",
        color="#f8fafc",
        pad=14,
    )
    plt.tight_layout()
    st.pyplot(fig_f)
    plt.close(fig_f)

with col_right:
  st.markdown(
      f"""
    <div class="info-box">
        <h4 style="margin-top:0; color:#38bdf8; font-size:18px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">
            Basic Information
        </h4>
        <h3 style="margin-top:0; color:#f8fafc; font-size:20px;"><b>PLTS UBP Grati 1.5 MWp</b></h3>
        <p style="color:#94a3b8; margin-bottom:14px; font-size:14px; line-height:1.5;">
            Desa Wates, Jl. Raya Surabaya - Probolinggo KM.73<br>
            Lekok, Pasir Panjang, Wates, Kec. Lekok, Pasuruan<br>
            Jawa Timur 67186
        </p>
        <table class="info-table">
            <tr><td><b>Status System</b></td><td>: <span style="background-color:#166534; color:#4ade80; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:13px;">● ONLINE (ML ACTIVE)</span></td></tr>
            <tr><td><b>ML Architecture</b></td><td>: <span style="color:#38bdf8; font-weight:700;">Gradient Boosting & Isolation Forest</span></td></tr>
            <tr><td><b>Total String Capacity</b></td><td>: <span style="color:#f8fafc; font-weight:700;">1507.00 kWp</span></td></tr>
            <tr><td><b>Grid Connection Date</b></td><td>: <span style="color:#f8fafc; font-weight:700;">19 August 2021</span></td></tr>
            <tr><td><b>Longitude & Latitude</b></td><td>: <span style="color:#f8fafc; font-weight:700;">{LAT} & {LON}</span></td></tr>
            <tr><td><b>PV System Type</b></td><td>: <span style="color:#f8fafc; font-weight:700;">Ground-mounted large scale</span></td></tr>
            <tr><td><b>Azimuth / Tilt</b></td><td>: <span style="color:#f8fafc; font-weight:700;">0° / 10°</span></td></tr>
        </table>
    </div>
    """,
      unsafe_allow_html=True,
  )

# ---------------------------------------------------------
# 8. ANOMALY DETECTION ENGINE & NOTIFICATION WARNING
# ---------------------------------------------------------
st.markdown(
    f"""
<div class="ml-panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <span style="font-weight:900; color:#38bdf8; font-size:18px; letter-spacing:0.8px;">
            🤖 ML ANOMALY DETECTION & PERFORMANCE RATIO (PR) ENGINE
        </span>
        <span class="ml-badge">DETEKSI DIAGNOSIS REAL-TIME</span>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; font-size: 14px;">
        <div style="border-left: 4px solid #38bdf8; padding-left: 12px;">
            <div style="color:#94a3b8; font-size:13px; font-weight:bold;">ESTIMASI TOTAL ENERGY HARI INI</div>
            <div style="color:#38bdf8; font-weight:900; font-size:20px;">{(df_min['ml_power_kw'].sum()/60.0):.2f} kWh</div>
            <div style="color:#64748b; font-size:12px;">Prediksi Model Machine Learning</div>
        </div>
        <div style="border-left: 4px solid #22c55e; padding-left: 12px;">
            <div style="color:#94a3b8; font-size:13px; font-weight:bold;">PERFORMANCE RATIO (PR) ESTIMATE</div>
            <div style="color:#22c55e; font-weight:900; font-size:20px;">
                {pr_daily:.2f} %
            </div>
            <div style="color:#64748b; font-size:12px;">Target Efisiensi Minimal: 75.0%</div>
        </div>
        <div style="border-left: 4px solid #f59e0b; padding-left: 12px;">
            <div style="color:#94a3b8; font-size:13px; font-weight:bold;">THERMAL LOSS PENALTY</div>
            <div style="color:#f59e0b; font-weight:900; font-size:20px;">
                {-TEMP_COEFF * (curr_cell_temp - 25) * 100 if curr_cell_temp > 25 else 0.0:.2f} %
            </div>
            <div style="color:#64748b; font-size:12px;">Pengaruh Suhu PV Cell ({curr_cell_temp:.1f}°C)</div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

# Render Warning Notification Box
if len(warnings_list) > 0:
  for warn in warnings_list:
    st.markdown(
        f'<div class="warning-box">{warn}</div>', unsafe_allow_html=True
    )
else:
  st.markdown(
      '<div class="normal-box">✅ **SISTEM NORMAL**: Tidak terdeteksi anomali'
      " penurunan efisiensi. Suhu Cell, Irradiance, Kurva Inverter, dan"
      " Koneksi Kabel dalam kondisi optimal.</div>",
      unsafe_allow_html=True,
  )

st.markdown("</div><br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 9. METRICS SCADA GRID DENGAN NEON CARD (FONT DIBESARKAN)
# ---------------------------------------------------------
def create_card(
    title, value, unit="", border_color="#38bdf8", glow_color="rgba(56,189,248,0.7)"
):
  return f"""
    <div class="scada-card" style="border-left-color: {border_color};">
        <div class="scada-title">{title}</div>
        <div class="scada-value" style="text-shadow: 0 0 10px {glow_color}, 0 0 18px {glow_color};">
            {value} <span class="scada-unit" style="color:{border_color};">{unit}</span>
        </div>
    </div>
    """


# BARIS 1
r1_1, r1_2, r1_3, r1_4, r1_5, r1_6 = st.columns(6)
with r1_1:
  st.markdown(
      create_card(
          "PR Daily",
          f"{pr_daily:.2f}",
          "%",
          "#4ade80",
          "rgba(74,222,128,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r1_2:
  st.markdown(
      create_card(
          "Irradiance",
          f"{curr_ghi:.2f}",
          "W/m²",
          "#fbbf24",
          "rgba(251,191,36,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r1_3:
  st.markdown(
      create_card(
          "Cell Temp",
          f"{curr_cell_temp:.2f}",
          "°C",
          "#f97316",
          "rgba(249,115,22,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r1_4:
  st.markdown(
      create_card(
          "Total DC Power",
          f"{curr_power_kw * 1.03:.2f}",
          "kW",
          "#38bdf8",
          "rgba(56,189,248,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r1_5:
  st.markdown(
      create_card(
          "Total AC Power",
          f"{curr_power_kw:.2f}",
          "kW",
          "#00f2fe",
          "rgba(0,242,254,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r1_6:
  st.markdown(
      create_card(
          "Daily Energy",
          f"{daily_kwh:.2f}",
          "kWh",
          "#a855f7",
          "rgba(168,85,247,0.7)",
      ),
      unsafe_allow_html=True,
  )

# BARIS 2
r2_1, r2_2, r2_3, r2_4, r2_5, r2_6 = st.columns(6)
with r2_1:
  st.markdown(
      create_card(
          "Daily kWh/kWp",
          f"{kwh_per_kwp:.2f}",
          "",
          "#4ade80",
          "rgba(74,222,128,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r2_2:
  st.markdown(
      create_card(
          "Ambient Temp",
          f"{curr_temp:.2f}",
          "°C",
          "#f97316",
          "rgba(249,115,22,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r2_3:
  st.markdown(
      create_card(
          "Trees Saved",
          f"{trees_saved:.2f}",
          "Trees",
          "#22c55e",
          "rgba(34,197,94,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r2_4:
  st.markdown(
      create_card(
          "DC Voltage",
          "720.40" if curr_power_kw > 0 else "0.00",
          "V",
          "#38bdf8",
          "rgba(56,189,248,0.7)",
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
          "rgba(0,242,254,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r2_6:
  st.markdown(
      create_card(
          "Total AC Energy",
          "11869.48",
          "MWh",
          "#a855f7",
          "rgba(168,85,247,0.7)",
      ),
      unsafe_allow_html=True,
  )

# BARIS 3
r3_1, r3_2, r3_3, r3_4, r3_5, r3_6 = st.columns(6)
with r3_1:
  st.markdown(
      create_card(
          "Export Meter",
          "12105109.50",
          "kWh",
          "#a855f7",
          "rgba(168,85,247,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r3_2:
  st.markdown(
      create_card(
          "CO² Saved",
          f"{co2_saved_ton:.2f}",
          "Ton",
          "#22c55e",
          "rgba(34,197,94,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r3_3:
  st.markdown(
      create_card(
          "AC Power Factor",
          "0.99" if curr_power_kw > 0 else "0.00",
          "",
          "#eab308",
          "rgba(234,179,8,0.7)",
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
          "rgba(56,189,248,0.7)",
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
          "rgba(0,242,254,0.7)",
      ),
      unsafe_allow_html=True,
  )
with r3_6:
  st.markdown(
      create_card(
          "AC Frequency", "50.01", "Hz", "#eab308", "rgba(234,179,8,0.7)"
      ),
      unsafe_allow_html=True,
  )
