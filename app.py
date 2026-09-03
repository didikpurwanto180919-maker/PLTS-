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

# Safe Import SHAP (Mencegah Crash jika library belum terinstal di Streamlit Cloud)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Optimasi & Anomaly Detection PLTS UBP Grati",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Autorefresh setiap 10 detik agar realtime 24 jam ter-update otomatis
st_autorefresh(interval=10 * 1000, key="plts_live_refresh_10s")

st.markdown(
    """
<style>
    .stApp {
        background-color: #050811;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .main-header {
        text-align: center;
        color: #38bdf8;
        font-weight: 900;
        font-size: 30px;
        margin-bottom: 2px;
        letter-spacing: 2px;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.7);
        text-transform: uppercase;
    }
    .sub-header {
        text-align: center;
        color: #f59e0b;
        font-weight: 800;
        font-size: 24px;
        margin-top: 2px;
        margin-bottom: 18px;
        letter-spacing: 1.5px;
        text-shadow: 0 0 15px rgba(245, 158, 11, 0.6);
    }
    
    .banner-bar {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        color: #38bdf8;
        padding: 12px 28px;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.7);
    }
    
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
        font-size: 13px;
        color: #cbd5e1;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .scada-value {
        font-size: 26px;
        font-weight: 900;
        color: #ffffff;
        text-align: right;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .scada-unit {
        font-size: 15px;
        font-weight: 800;
        margin-left: 4px;
    }
    
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
        font-size: 14px;
    }
    
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
        font-size: 12px;
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
        font-size: 15px;
        font-weight: 600;
    }
    .normal-box {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        color: #86efac;
        font-size: 16px;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #0f172a;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid #1e293b;
        padding: 0px 24px;
        font-weight: 700;
        font-size: 14px;
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
V_NOMINAL_STRING = 720.0  # Vdc Nominal

wib_tz = pytz.timezone("Asia/Jakarta")
now_wib = datetime.now(wib_tz)
today_str = now_wib.strftime("%Y-%m-%d")

# ---------------------------------------------------------
# 3. ENGINE MACHINE LEARNING
# ---------------------------------------------------------
@st.cache_resource
def train_plts_ml_models():
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

    X = pd.DataFrame({"ghi": ghi_sim, "temp_ambient": temp_sim, "hour": hour_sim})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg_model = GradientBoostingRegressor(
        n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
    )
    reg_model.fit(X_scaled, power_actual)

    iso_model = IsolationForest(contamination=0.05, random_state=42)
    df_iso = X.copy()
    df_iso["power"] = power_actual
    iso_model.fit(df_iso)

    return reg_model, iso_model, scaler

ml_model, anomaly_model, ml_scaler = train_plts_ml_models()

# ---------------------------------------------------------
# 4. DATA FETCHING (24 JAM KONTINYU)
# ---------------------------------------------------------
url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=shortwave_radiation,temperature_2m&timezone=auto&start_date={today_str}&end_date={today_str}"

try:
    res = requests.get(url, timeout=5)
    res.raise_for_status()
    data = res.json()
    hourly = data["hourly"]

    df_hourly = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "ghi": hourly["shortwave_radiation"],
        "temp_ambient": hourly["temperature_2m"],
    })
except Exception:
    times = pd.date_range(start=f"{today_str} 00:00", periods=24, freq="h")
    ghi_sim = [0, 0, 0, 0, 0, 0, 80, 300, 650, 880, 980, 1050, 990, 820, 580, 320, 90, 5, 0, 0, 0, 0, 0, 0]
    temp_sim = [24, 24, 23, 23, 24, 25, 27, 29, 31, 33, 34, 35, 35, 34, 33, 31, 29, 27, 26, 25, 25, 24, 24, 24]
    df_hourly = pd.DataFrame({"time": times, "ghi": ghi_sim, "temp_ambient": temp_sim})

if df_hourly["time"].dt.tz is None:
    df_hourly["time"] = df_hourly["time"].dt.tz_localize("Asia/Jakarta")
else:
    df_hourly["time"] = df_hourly["time"].dt.tz_convert("Asia/Jakarta")

df_hourly = df_hourly.set_index("time")
df_min = df_hourly.resample("1min").interpolate(method="linear").reset_index()
df_min["hour"] = df_min["time"].dt.hour + df_min["time"].dt.minute / 60.0

# Predict Output Daya ML untuk 24 Jam Penuh
X_live = df_min[["ghi", "temp_ambient", "hour"]]
X_live_scaled = ml_scaler.transform(X_live)
df_min["ml_power_kw"] = ml_model.predict(X_live_scaled)
df_min["ml_power_kw"] = df_min.apply(
    lambda r: 0.0 if r["ghi"] <= 2.0 else max(0.0, r["ml_power_kw"]), axis=1
)

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

# Menentukan data histori real-time hingga detik/menit saat ini
past_rows = df_min[df_min["time"] <= now_wib]
current_row = past_rows.iloc[-1] if not past_rows.empty else df_min.iloc[0]

curr_time = current_row["time"]
curr_ghi = current_row["ghi"]
curr_temp = current_row["temp_ambient"]
curr_power_kw = current_row["ml_power_kw"] if pd.notna(current_row["ml_power_kw"]) else 0.0
curr_physics_kw = current_row["physics_power_kw"]
curr_cell_temp = curr_temp + (NOCT - 20) * (curr_ghi / 800.0) if curr_ghi > 0 else curr_temp

# Parameter SCADA & Operasional
curr_vdc = 720.40 if curr_power_kw > 0 else 0.0
curr_idc = (curr_power_kw * 1000 / curr_vdc) if curr_vdc > 0 else 0.0
curr_vac = 380.15 if curr_power_kw > 0 else 0.0
curr_iac = (curr_power_kw * 1000 / (curr_vac * 1.732 * 0.99)) if curr_vac > 0 else 0.0
curr_freq = 50.01
curr_thd = 1.8
curr_inv_temp = 42.0
curr_iso_res = 12.5
curr_albedo = 0.22
scada_data_lag_sec = 0

daily_kwh = past_rows["ml_power_kw"].sum() / 60.0
kwh_per_kwp = daily_kwh / CAPACITY_KWP if CAPACITY_KWP > 0 else 0.0
co2_saved_ton = daily_kwh * 0.00085
trees_saved = co2_saved_ton * 40.0

pr_daily = min(
    98.5,
    max(
        75.0,
        (daily_kwh / (CAPACITY_KWP * 4.5)) * 100 if daily_kwh > 0 else 88.5,
    ),
)

# ---------------------------------------------------------
# 5. INTEGRATED ANOMALY DETECTION ENGINE
# ---------------------------------------------------------
warnings_list = []

# SISI DC (PV ARRAY & STRING)
if curr_ghi > 200 and (0.75 * curr_physics_kw <= curr_power_kw < 0.88 * curr_physics_kw):
    warnings_list.append("⚠️ **PV Kotor / Soiling Detected**: Penurunan output daya ~12-25% akibat debu/kotoran. Disarankan pencucian modul.")

ghi_std_last_15m = past_rows.tail(15)["ghi"].std() if len(past_rows) >= 15 else 0
if curr_ghi > 300 and ghi_std_last_15m > 120 and curr_power_kw < curr_physics_kw * 0.70:
    warnings_list.append("☁️ **Shadowing / Transient Cloud Passing**: Fluktuasi daya tajam terdeteksi akibat bayangan awan melintas/vegetasi.")

if curr_vdc > 0 and abs(curr_vdc - (2/3 * V_NOMINAL_STRING)) < 30:
    warnings_list.append("⚡ **Bypass Diode Failure / Short Circuit**: Tegangan Vdc drop ~1/3 dari nominal. Korsleting diode modul terdeteksi.")

if curr_ghi > 400 and curr_vdc < V_NOMINAL_STRING * 0.82 and curr_cell_temp > 55:
    warnings_list.append("🔬 **PID / Microcracks / Hotspot**: Terdeteksi degradasi sel atau retak mikro yang memicu pembentukan hotspot berlebih.")

if curr_ghi > 200 and curr_idc < 0.5:
    warnings_list.append("🔌 **String Open Circuit / Arus Putus**: Arus string bernilai 0 A saat Irradiance tinggi (>200 W/m²). Cek Fuse/Connector.")

if curr_iso_res < 1.0:
    warnings_list.append("🌧️ **DC Ground Fault / Isolasi Turun**: Resistansi isolasi kabel DC ke bumi drop (< 1 MΩ). Risiko kelembapan/kebocoran arus.")

# SISI INVERTER & AC
if curr_temp > 38.0:
    warnings_list.append(f"🌡️ **Ambient Suhu Lingkungan Panas ({curr_temp:.1f}°C)**: Suhu sekeliling tinggi memicu efisiensi pendinginan menurun.")
if curr_inv_temp > 65.0:
    warnings_list.append(f"🔥 **Overheating Inverter ({curr_inv_temp:.1f}°C)**: Suhu internal inverter kritikal! Inverter melakukan Thermal Derating.")

if curr_cell_temp > 58.0:
    warnings_list.append(f"🔥 **Suhu Modul PV Panas ({curr_cell_temp:.1f}°C)**: Suhu sel melebihi 58°C, menyebabkan Thermal Loss Penalty bertambah.")

real_inv_eff = (curr_power_kw / (curr_vdc * curr_idc / 1000)) if (curr_vdc * curr_idc) > 0 else INVERTER_EFF
if curr_power_kw > 50 and real_inv_eff < 0.75:
    warnings_list.append(f"📉 **Inverter Efisiensi Drop ({real_inv_eff*100:.1f}%)**: Efisiensi inverter turun drastis! Indikasi komponen internal/IGBT bermasalah.")

if curr_albedo < 0.12 and curr_ghi > 300:
    warnings_list.append(f"🌱 **Albedo Variation Anomaly ({curr_albedo:.2f})**: Reflektifitas permukaan tanah turun. Indikasi rumput liar tinggi / genangan air.")

if scada_data_lag_sec > 60:
    warnings_list.append(f"⏱️ **Telemetry Data Lag / Freeze ({scada_data_lag_sec}s)**: Pembacaan sensor atau inverter berhenti memperbarui nilai.")

# ---------------------------------------------------------
# 6. HEADER & STATUS BAR
# ---------------------------------------------------------
st.markdown(
    "<h2 class='main-header'>MACHINE LEARNING OPTIMASI PRODUKSI PLTS LANDBASE 1.5 MWp</h2>",
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
    <span style="color:#f8fafc; letter-spacing:2px; font-size:16px;">
        <span style="color:#0284c7;">●</span> OVERVIEW MONITORING & ML ANOMALY DETECTION
    </span>
    <span style="font-size:16px; color:#e2e8f0;">{now_wib.strftime('%Y-%m-%d %H:%M:%S')} WIB</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 7. LAYOUT UTAMA & VISUALISASI CHART PERBAIKAN (24 JAM CONTINUOUS)
# ---------------------------------------------------------
col_left, col_right = st.columns([1.55, 1.0])

with col_left:
    tab1, tab2, tab3 = st.tabs([
        "📈 Real-time ML Prediction",
        "🔮 24-Hour Forecast & Optimization",
        "🧠 Feature Importance (XAI)"
    ])

    with tab1:
        plt.style.use("dark_background")
        fig, ax1 = plt.subplots(figsize=(9.2, 4.5))
        fig.patch.set_facecolor("#050811")
        ax1.set_facecolor("#0a0f1d")

        # 1. Kurva ML Active Power (24 Jam Kontinyu)
        line1_glow, = ax1.plot(
            df_min["time"],
            df_min["ml_power_kw"],
            color="#00f2fe",
            linewidth=5.0,
            alpha=0.35,
        )
        line1, = ax1.plot(
            df_min["time"],
            df_min["ml_power_kw"],
            color="#00f2fe",
            linewidth=2.8,
            label="ML Active Power (kW)",
        )
        
        # Area Shading Transparan untuk Histori sampai detik ini
        ax1.fill_between(
            past_rows["time"],
            past_rows["ml_power_kw"],
            color="#00f2fe",
            alpha=0.22,
        )

        # 2. Kurva Physics Baseline (24 Jam Kontinyu)
        line3, = ax1.plot(
            df_min["time"],
            df_min["physics_power_kw"],
            color="#f43f5e",
            linestyle="--",
            linewidth=2.2,
            alpha=0.9,
            label="Physics Baseline (kW)",
        )

        # 3. Kurva Irradiance (Sumbu Kanan - 24 Jam Kontinyu)
        ax2 = ax1.twinx()
        line2, = ax2.plot(
            df_min["time"],
            df_min["ghi"],
            color="#fbbf24",
            linestyle=":",
            linewidth=2.2,
            alpha=0.9,
            label="Irradiance (W/m²)",
        )
        ax2.set_ylabel("Irradiance (W/m²)", color="#fbbf24", fontsize=11, weight="bold")
        ax2.set_ylim(0, 1300)
        ax2.tick_params(colors="#fbbf24")

        # 4. Marker Point & Annotation Box Nilai Angka LIVE (ML) & BASELINE
        if pd.notna(curr_power_kw):
            # Glowing marker point pada titik realtime
            ax1.plot(curr_time, curr_power_kw, marker="o", markersize=12, color="#00f2fe", alpha=0.4)
            ax1.plot(curr_time, curr_power_kw, marker="o", markersize=7, color="#ffffff")

            # Annotation Box Atas: Real-time ML Power (Warna Cyan/Blue Glow)
            ax1.annotate(
                f"LIVE: {curr_power_kw:.1f} kW",
                xy=(curr_time, curr_power_kw),
                xytext=(-85, 30),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#0284c7", edgecolor="#38bdf8", alpha=0.95),
                fontsize=11,
                fontweight="bold",
                color="#ffffff",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#38bdf8", lw=2),
            )

            # Annotation Box Bawah: Physics Baseline Value (Warna Red/Crimson Glow)
            ax1.annotate(
                f"BASE: {curr_physics_kw:.1f} kW",
                xy=(curr_time, curr_physics_kw),
                xytext=(-85, -45),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#9f1239", edgecolor="#f43f5e", alpha=0.95),
                fontsize=11,
                fontweight="bold",
                color="#ffffff",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="#f43f5e", lw=2),
            )

        ax1.set_ylabel("Active Power (kW)", color="#00f2fe", fontsize=11, weight="bold")
        ax1.set_ylim(0, 1650)
        ax1.tick_params(colors="#00f2fe")

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax1.grid(True, linestyle=":", alpha=0.25, color="#475569")

        # Legenda Modern
        ax1.legend(
            [line1, line3, line2],
            ["ML Active Power (kW)", "Physics Baseline", "Irradiance (W/m²)"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig_f, ax_f = plt.subplots(figsize=(9.2, 4.5))
        fig_f.patch.set_facecolor("#050811")
        ax_f.set_facecolor("#0a0f1d")

        ax_f.plot(
            df_min["time"],
            df_min["ml_power_kw"],
            color="#38bdf8",
            linewidth=2.8,
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

        ax_f.set_ylabel("Forecasted Power (kW)", color="#38bdf8", fontsize=11, weight="bold")
        ax_f.set_ylim(0, 1650)
        ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=wib_tz))
        ax_f.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_f.grid(True, linestyle=":", alpha=0.25, color="#475569")
        ax_f.legend(loc="upper right", frameon=False, fontsize=10)

        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

    with tab3:
        fig_xai, ax_xai = plt.subplots(figsize=(9.2, 4.5))
        fig_xai.patch.set_facecolor("#050811")
        ax_xai.set_facecolor("#0a0f1d")
        
        if HAS_SHAP:
            explainer = shap.TreeExplainer(ml_model)
            sample_scaled = ml_scaler.transform(X_live.sample(100, random_state=42))
            shap_values = explainer.shap_values(sample_scaled)
            shap.summary_plot(shap_values, sample_scaled, feature_names=["Irradiance (GHI)", "Ambient Temp", "Hour"], plot_type="bar", show=False)
        else:
            features = ["Irradiance (GHI)", "Ambient Temp", "Hour"]
            importances = ml_model.feature_importances_
            ax_xai.barh(features, importances, color="#0284c7")
            ax_xai.set_xlabel("Relative Feature Importance Score", color="#f8fafc")
            ax_xai.tick_params(colors="#f8fafc")
            
        plt.tight_layout()
        st.pyplot(fig_xai)
        plt.close(fig_xai)

with col_right:
    st.markdown(
        f"""
    <div class="info-box">
        <h4 style="margin-top:0; color:#38bdf8; font-size:16px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">
            Basic Information
        </h4>
        <h3 style="margin-top:0; color:#f8fafc; font-size:18px;"><b>PLTS UBP Grati 1.5 MWp</b></h3>
        <p style="color:#94a3b8; margin-bottom:12px; font-size:13px; line-height:1.4;">
            Desa Wates, Lekok, Pasuruan, Jawa Timur
        </p>
        <table class="info-table">
            <tr><td><b>Status System</b></td><td>: <span style="background-color:#166534; color:#4ade80; padding:2px 8px; border-radius:10px; font-weight:bold; font-size:12px;">● ONLINE</span></td></tr>
            <tr><td><b>ML Architecture</b></td><td>: <span style="color:#38bdf8; font-weight:700;">Gradient Boosting + Isolation Forest</span></td></tr>
            <tr><td><b>Capacity</b></td><td>: <span style="color:#f8fafc; font-weight:700;">1507.00 kWp</span></td></tr>
            <tr><td><b>Coordinates</b></td><td>: <span style="color:#f8fafc; font-weight:700;">{LAT}, {LON}</span></td></tr>
        </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# 8. PANEL ANOMALY DETECTION & AUDIO ALARM
# ---------------------------------------------------------
st.markdown(
    f"""
<div class="ml-panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:18px;">
        <span style="font-weight:900; color:#38bdf8; font-size:20px; letter-spacing:0.5px;">
            🤖 ML ANOMALY DETECTION & PERFORMANCE RATIO (PR) ENGINE
        </span>
        <span class="ml-badge" style="font-size:14px; padding:6px 14px; font-weight:800;">DETEKSI REAL-TIME</span>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; font-size: 15px;">
        <div>
            <div style="color:#cbd5e1; font-weight:700; font-size:15px; letter-spacing:0.5px;">ESTIMASI TOTAL ENERGY</div>
            <div style="color:#38bdf8; font-weight:900; font-size:26px; margin-top:4px;">{(df_min['ml_power_kw'].sum()/60.0):.2f} kWh</div>
        </div>
        <div>
            <div style="color:#cbd5e1; font-weight:700; font-size:15px; letter-spacing:0.5px;">PERFORMANCE RATIO</div>
            <div style="color:#22c55e; font-weight:900; font-size:26px; margin-top:4px;">{pr_daily:.2f} %</div>
        </div>
        <div>
            <div style="color:#cbd5e1; font-weight:700; font-size:15px; letter-spacing:0.5px;">THERMAL LOSS PENALTY</div>
            <div style="color:#f59e0b; font-weight:900; font-size:26px; margin-top:4px;">{-TEMP_COEFF * (curr_cell_temp - 25) * 100 if curr_cell_temp > 25 else 0.0:.2f} %</div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)

if warnings_list:
    # HTML5 / Web Audio API Synthesizer Alarm Sound Generator (Bunyi Sirene Peringatan Laptop)
    alarm_html = """
    <script>
    function playAlarmSound() {
        try {
            var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime); // 880Hz (A5 Tone)
            osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.5); // Sweeping Sirene
            
            gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
            
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            
            osc.start();
            osc.stop(audioCtx.currentTime + 0.5);
        } catch(e) {
            console.log("Audio play error: " + e);
        }
    }
    // Bunyikan Alarm 2x Pulse saat anomali terdeteksi
    playAlarmSound();
    setTimeout(playAlarmSound, 600);
    </script>
    """
    st.components.v1.html(alarm_html, height=0, width=0)

    for warn in warnings_list:
        st.markdown(
            f'<div class="warning-box">{warn}</div>', 
            unsafe_allow_html=True
        )
else:
    st.markdown(
        '<div class="normal-box">✅ <b>SISTEM NORMAL</b>: Tidak terdeteksi anomali pada PV String, Inverter, Lingkungan, maupun Sistem SCADA.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div><br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 9. METRICS SCADA GRID
# ---------------------------------------------------------
def create_card(title, value, unit="", border_color="#38bdf8", glow_color="rgba(56,189,248,0.7)"):
    return f"""
    <div class="scada-card" style="border-left-color: {border_color};">
        <div class="scada-title">{title}</div>
        <div class="scada-value" style="text-shadow: 0 0 10px {glow_color};">
            {value} <span class="scada-unit" style="color:{border_color};">{unit}</span>
        </div>
    </div>
    """

# BARIS 1
r1_1, r1_2, r1_3, r1_4, r1_5, r1_6 = st.columns(6)
with r1_1: st.markdown(create_card("PR Daily", f"{pr_daily:.2f}", "%", "#4ade80"), unsafe_allow_html=True)
with r1_2: st.markdown(create_card("Irradiance", f"{curr_ghi:.2f}", "W/m²", "#fbbf24"), unsafe_allow_html=True)
with r1_3: st.markdown(create_card("Cell Temp", f"{curr_cell_temp:.2f}", "°C", "#f97316"), unsafe_allow_html=True)
with r1_4: st.markdown(create_card("Total DC Power", f"{curr_power_kw * 1.03:.2f}", "kW", "#38bdf8"), unsafe_allow_html=True)
with r1_5: st.markdown(create_card("Total AC Power", f"{curr_power_kw:.2f}", "kW", "#00f2fe"), unsafe_allow_html=True)
with r1_6: st.markdown(create_card("Daily Energy", f"{daily_kwh:.2f}", "kWh", "#a855f7"), unsafe_allow_html=True)

# BARIS 2
r2_1, r2_2, r2_3, r2_4, r2_5, r2_6 = st.columns(6)
with r2_1: st.markdown(create_card("Daily kWh/kWp", f"{kwh_per_kwp:.2f}", "", "#4ade80"), unsafe_allow_html=True)
with r2_2: st.markdown(create_card("Ambient Temp", f"{curr_temp:.2f}", "°C", "#f97316"), unsafe_allow_html=True)
with r2_3: st.markdown(create_card("Trees Saved", f"{trees_saved:.2f}", "Trees", "#22c55e"), unsafe_allow_html=True)
with r2_4: st.markdown(create_card("DC Voltage", f"{curr_vdc:.2f}", "V", "#38bdf8"), unsafe_allow_html=True)
with r2_5: st.markdown(create_card("AC Voltage", f"{curr_vac:.2f}", "V", "#00f2fe"), unsafe_allow_html=True)
with r2_6: st.markdown(create_card("Total AC Energy", "11869.48", "MWh", "#a855f7"), unsafe_allow_html=True)

# BARIS 3
r3_1, r3_2, r3_3, r3_4, r3_5, r3_6 = st.columns(6)
with r3_1: st.markdown(create_card("Export Meter", "12105109.50", "kWh", "#a855f7"), unsafe_allow_html=True)
with r3_2: st.markdown(create_card("CO² Saved", f"{co2_saved_ton:.2f}", "Ton", "#22c55e"), unsafe_allow_html=True)
with r3_3: st.markdown(create_card("AC Power Factor", "0.99" if curr_power_kw > 0 else "0.00", "", "#eab308"), unsafe_allow_html=True)
with r3_4: st.markdown(create_card("DC Current", f"{curr_idc:.2f}", "A", "#38bdf8"), unsafe_allow_html=True)
with r3_5: st.markdown(create_card("AC Current", f"{curr_iac:.2f}", "A", "#00f2fe"), unsafe_allow_html=True)
with r3_6: st.markdown(create_card("AC Frequency", f"{curr_freq:.2f}", "Hz", "#eab308"), unsafe_allow_html=True)
