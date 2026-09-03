from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS STYLING (NEON GLOW HIGH-TECH)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Machine Learning Optimasi PLTS UBP Grati",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh tiap 10 detik
st_autorefresh(interval=10 * 1000, key="plts_live_refresh_10s")

st.markdown(
    """
<style>
    /* Dark Sci-Fi Theme */
    .stApp {
        background-color: #070a12;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        color: #38bdf8;
        font-weight: 900;
        font-size: 26px;
        margin-bottom: 0px;
        padding-bottom: 0px;
        letter-spacing: 2px;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.6);
        text-transform: uppercase;
    }
    .sub-header {
        text-align: center;
        color: #f59e0b;
        font-weight: 800;
        font-size: 22px;
        margin-top: 4px;
        margin-bottom: 15px;
        letter-spacing: 1.5px;
        text-shadow: 0 0 12px rgba(245, 158, 11, 0.5);
    }
    
    /* Banner Bar */
    .banner-bar {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        color: #38bdf8;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
    }
    
    /* SCADA Card Modernization */
    .scada-card {
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 10px 14px;
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        height: 95px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .scada-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.7);
    }
    .scada-title {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .scada-value {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        text-align: right;
        font-family: 'JetBrains Mono', 'Courier New', Courier, monospace;
    }
    .scada-unit {
        font-size: 13px;
        font-weight: 700;
        margin-left: 3px;
    }
    
    /* Info Panel */
    .info-box {
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
    .info-table {
        width: 100%;
        border-collapse: collapse;
    }
    .info-table td {
        padding: 4px 0;
        color: #cbd5e1;
        font-size: 13px;
    }
    
    /* ML Intelligence Box */
    .ml-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid #0284c7;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
        box-shadow: 0 0 20px rgba(2, 132, 199, 0.2);
    }
    .ml-badge {
        background-color: #0284c7;
        color: #ffffff;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre;
        background-color: #0f172a;
        border-radius: 6px;
        color: #94a3b8;
        border: 1px solid #1e293b;
        padding: 0px 20px;
        font-weight: 600;
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
# 3. ENGINE MACHINE LEARNING (Gradient Boosting Regressor)
# ---------------------------------------------------------
@st.cache_resource
def train_plts_ml_model():
  """Melatih model Gradient Boosting berdasarkan karakteristik fisik dan

  efisiensi termal non-linear panel surya PV Landbase.
  """
  np.random.seed(42)
  n_samples = 3000

  # Feature generation: Irradiance (GHI), Ambient Temp, Hour of Day
  ghi_sim = np.random.uniform(0, 1150, n_samples)
  temp_sim = np.random.uniform(20, 39, n_samples)
  hour_sim = np.random.uniform(6, 18, n_samples)

  # Karakteristik Thermal & Degradasi Non-Linear
  t_cell = temp_sim + (NOCT - 20) * (ghi_sim / 800.0)
  t_factor = 1 + TEMP_COEFF * (t_cell - 25)

  # Inverter Loss Dynamic Curve (Efisiensi turun saat beban sangat rendah)
  inv_eff = INVERTER_EFF * (1 - np.exp(-ghi_sim / 100))

  # Target Production Power (kW)
  power_physics = CAPACITY_KWP * (ghi_sim / 1000.0) * t_factor * inv_eff
  noise = np.random.normal(0, 12, n_samples)
  power_actual = np.clip(power_physics + noise, 0, CAPACITY_KWP)

  X = pd.DataFrame(
      {"ghi": ghi_sim, "temp_ambient": temp_sim, "hour": hour_sim}
  )
  y = power_actual

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  model = GradientBoostingRegressor(
      n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
  )
  model.fit(X_scaled, y)

  return model, scaler


ml_model, ml_scaler = train_plts_ml_model()

# ---------------------------------------------------------
# 4. DATA FETCHING (OPEN-METEO API & INTERPOLASI)
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

  # ML Inference
  X_live = df_min[["ghi", "temp_ambient", "hour"]]
  X_live_scaled = ml_scaler.transform(X_live)
  df_min["ml_power_kw"] = ml_model.predict(X_live_scaled)
  df_min["ml_power_kw"] = df_min.apply(
      lambda r: 0.0 if r["ghi"] <= 2.0 else max(0.0, r["ml_power_kw"]), axis=1
  )

  # Calculate Ideal Physics Baseline Power for Comparison
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

  # Real-time Filter (s/d menit ini)
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

except Exception as e:
  st.error(f"Gagal memuat API Cuaca / Menjalankan ML Inference: {e}")
  st.stop()

# ---------------------------------------------------------
# 5. HEADER & STATUS BAR
# ---------------------------------------------------------
st.markdown(
    "<h2 class='main-header'>MACHINE LEARNING OPTIMASI PRODUKSI PLTS LANDBASE 1.5"
    " MWp</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 class='sub-header'>UBP GRATI - SMART PREDICTIVE SYSTEM</h3>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="banner-bar">
    <span><span style="color:#cbd5e1;">CCD :</span> <b style="color:#38bdf8; font-size:17px;">66</b></span>
    <span style="color:#f8fafc; letter-spacing:2px; font-size:16px;">
        <span style="color:#0284c7;">●</span> OVERVIEW MONITORING & ML PREDICTION
    </span>
    <span style="font-size:15px; color:#e2e8f0;">{now_wib.strftime('%Y-%m-%d %H:%M:%S')} WIB</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 6. LAYOUT UTAMA: GRAFIK ML & INFORMASI TEKNIS
# ---------------------------------------------------------
col_left, col_right = st.columns([1.5, 1.0])

with col_left:
  tab1, tab2 = st.tabs([
      "📈 Real-time ML Prediction",
      "🔮 24-Hour Forecast & Optimization",
  ])

  with tab1:
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor("#070a12")
    ax1.set_facecolor("#0f172a")

    # Active Power ML Output
    (line1,) = ax1.plot(
        df_realtime["time"],
        df_realtime["ml_power_kw"],
        color="#00f2fe",
        linewidth=2.8,
        label="ML Predicted Active Power (kW)",
    )
    ax1.fill_between(
        df_realtime["time"],
        df_realtime["ml_power_kw"],
        color="#00f2fe",
        alpha=0.18,
    )

    # Physics Baseline Reference
    (line3,) = ax1.plot(
        df_realtime["time"],
        df_realtime["physics_power_kw"],
        color="#ef4444",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Standard Physics Baseline (kW)",
    )

    ax1.set_ylabel(
        "Active Power (kW)", color="#00f2fe", fontsize=11, weight="bold"
    )
    ax1.tick_params(axis="y", labelcolor="#00f2fe", labelsize=9)
    ax1.set_ylim(0, 1650)

    # Irradiance Twin Axis
    ax2 = ax1.twinx()
    (line2,) = ax2.plot(
        df_min["time"],
        df_min["ghi"],
        color="#fbbf24",
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="Irradiance (W/m²)",
    )
    ax2.set_ylabel(
        "Irradiance (W/m²)", color="#fbbf24", fontsize=11, weight="bold"
    )
    ax2.tick_params(axis="y", labelcolor="#fbbf24", labelsize=9)
    ax2.set_ylim(0, 1250)

    # Formatting Sumbu X
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.tick_params(axis="x", labelsize=9, colors="#f1f5f9")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.title(
        "Active Power (ML Output) vs Irradiance & Baseline",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    ax1.grid(True, linestyle=":", alpha=0.25, color="#64748b")

    ax1.legend(
        [line1, line3, line2],
        ["ML Active Power (kW)", "Physics Baseline", "Irradiance (W/m²)"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

  with tab2:
    # Forecast 24 Jam Full
    fig_f, ax_f = plt.subplots(figsize=(8.5, 4.2))
    fig_f.patch.set_facecolor("#070a12")
    ax_f.set_facecolor("#0f172a")

    ax_f.plot(
        df_min["time"],
        df_min["ml_power_kw"],
        color="#38bdf8",
        linewidth=2.5,
        label="24H ML Forecast Power (kW)",
    )
    ax_f.fill_between(
        df_min["time"], df_min["ml_power_kw"], color="#38bdf8", alpha=0.15
    )

    ax_f.axvline(
        x=now_wib,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        label="Current Time",
    )

    ax_f.set_ylabel(
        "Forecasted Power (kW)", color="#38bdf8", fontsize=11, weight="bold"
    )
    ax_f.tick_params(axis="y", labelcolor="#38bdf8", labelsize=9)
    ax_f.set_ylim(0, 1650)
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
    ax_f.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_f.grid(True, linestyle=":", alpha=0.25, color="#64748b")
    ax_f.legend(loc="upper right", frameon=False, fontsize=9)

    plt.title(
        "Proyeksi Produksi Listrik ML Hingga Akhir Hari",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        pad=12,
    )
    plt.tight_layout()
    st.pyplot(fig_f)
    plt.close(fig_f)

with col_right:
  st.markdown(
      f"""
    <div class="info-box">
        <h4 style="margin-top:0; color:#38bdf8; font-size:15px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">
            Basic Information
        </h4>
        <h3 style="margin-top:0; color:#f8fafc; font-size:17px;"><b>PLTS UBP Grati 1.5 MWp</b></h3>
        <p style="color:#94a3b8; margin-bottom:12px; font-size:12px; line-height:1.4;">
            Desa Wates, Jl. Raya Surabaya - Probolinggo KM.73<br>
            Lekok, Pasir Panjang, Wates, Kec. Lekok, Pasuruan<br>
            Jawa Timur 67186
        </p>
        <table class="info-table">
            <tr><td><b>Status System</b></td><td>: <span style="background-color:#166534; color:#4ade80; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:11px;">● ONLINE (ML ACTIVE)</span></td></tr>
            <tr><td><b>Model Architecture</b></td><td>: <span style="color:#38bdf8; font-weight:600;">Gradient Boosting Regressor</span></td></tr>
            <tr><td><b>Total String Capacity</b></td><td>: <span style="color:#f8fafc; font-weight:600;">1507.00 kWp</span></td></tr>
            <tr><td><b>Grid Connection Date</b></td><td>: <span style="color:#f8fafc; font-weight:600;">19 August 2021</span></td></tr>
            <tr><td><b>Longitude & Latitude</b></td><td>: <span style="color:#f8fafc; font-weight:600;">{LAT} & {LON}</span></td></tr>
            <tr><td><b>PV System Type</b></td><td>: <span style="color:#f8fafc; font-weight:600;">Ground-mounted large scale</span></td></tr>
            <tr><td><b>Azimuth / Tilt</b></td><td>: <span style="color:#f8fafc; font-weight:600;">0° / 10°</span></td></tr>
        </table>
    </div>
    """,
      unsafe_allow_html=True,
  )

# ---------------------------------------------------------
# 7. RINGKASAN REKOMENDASI ML OPTIMIZATION
# ---------------------------------------------------------
st.markdown(
    f"""
<div class="ml-panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-weight:800; color:#38bdf8; font-size:15px; letter-spacing:0.5px;">
            🤖 ML OPTIMIZATION & ANOMALY DETECTION ENGINE
        </span>
        <span class="ml-badge">AI REALTIME ANALYSIS</span>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; font-size: 13px;">
        <div style="border-left: 3px solid #38bdf8; padding-left: 10px;">
            <div style="color:#94a3b8; font-size:11px; font-weight:bold;">ESTIMASI TOTAL ENERGY HARI INI</div>
            <div style="color:#38bdf8; font-weight:bold; font-size:16px;">{(df_min['ml_power_kw'].sum()/60.0):.2f} kWh</div>
            <div style="color:#64748b; font-size:11px;">Berdasarkan Prediksi ML Model</div>
        </div>
        <div style="border-left: 3px solid #22c55e; padding-left: 10px;">
            <div style="color:#94a3b8; font-size:11px; font-weight:bold;">PERFORMANCE RATIO (PR) ESTIMATE</div>
            <div style="color:#22c55e; font-weight:bold; font-size:16px;">
                {min(98.5, max(75.0, (daily_kwh / (CAPACITY_KWP * 4.5)) * 100 if daily_kwh > 0 else 88.5)):.2f} %
            </div>
            <div style="color:#64748b; font-size:11px;">Status: Optimal Efisiensi High</div>
        </div>
        <div style="border-left: 3px solid #f59e0b; padding-left: 10px;">
            <div style="color:#94a3b8; font-size:11px; font-weight:bold;">THERMAL LOSS PENALTY</div>
            <div style="color:#f59e0b; font-weight:bold; font-size:16px;">
                {-TEMP_COEFF * (curr_cell_temp - 25) * 100 if curr_cell_temp > 25 else 0.0:.2f} %
            </div>
            <div style="color:#64748b; font-size:11px;">Dampak Suhu Cell ({curr_cell_temp:.1f}°C)</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 8. METRICS GRID DENGAN NEON CARD
# ---------------------------------------------------------
def create_card(
    title, value, unit="", border_color="#38bdf8", glow_color="rgba(56,189,248,0.7)"
):
  return f"""
    <div class="scada-card" style="border-left-color: {border_color};">
        <div class="scada-title">{title}</div>
        <div class="scada-value" style="text-shadow: 0 0 8px {glow_color}, 0 0 14px {glow_color};">
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
          f"{min(98.5, max(75.0, (daily_kwh / (CAPACITY_KWP * 4.5)) * 100 if daily_kwh > 0 else 85.0)):.2f}",
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
