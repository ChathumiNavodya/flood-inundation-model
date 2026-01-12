import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# PAGE SETTINGS
# ==========================================================
st.set_page_config(page_title="Flood Relief Priority System - Sri Lanka", layout="wide")

# ==========================================================
# THEME SWITCHER
# ==========================================================
st.sidebar.title("🎨 Theme Settings")
theme = st.sidebar.radio("Select Theme", ["🌙 Dark Dashboard", "☀️ Light Professional"])

# ==========================================================
# APPLY CSS BASED ON THEME
# ==========================================================
def apply_theme(theme_name):
    if "Dark" in theme_name:
        st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: white; }
            section[data-testid="stSidebar"] { background-color: #161b22; }
            section[data-testid="stSidebar"] * { color: white; }

            h1, h2, h3, h4 { color: white; }
            div[data-testid="stDataFrame"] { background: #1f2937; border-radius: 12px; padding: 10px; }

            .card {
                background-color: #1f2937;
                border-radius: 16px;
                padding: 20px;
                margin: 10px 0px;
                box-shadow: 0px 2px 10px rgba(0,0,0,0.3);
                color: white;
            }

            div.stButton > button {
                background-color: #ff4b4b;
                color: white;
                border-radius: 10px;
                padding: 10px 18px;
                border: none;
                font-size: 16px;
            }
            div.stButton > button:hover {
                background-color: #ff1c1c;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <style>
            .stApp { background-color: #f6f8fa; color: black; }
            section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #ddd; }
            section[data-testid="stSidebar"] * { color: black; }

            h1, h2, h3, h4 { color: #0f172a; }

            div[data-testid="stDataFrame"] {
                background: white;
                border-radius: 12px;
                padding: 10px;
                border: 1px solid #e5e7eb;
            }

            .card {
                background-color: white;
                border-radius: 16px;
                padding: 20px;
                margin: 10px 0px;
                box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
                color: black;
                border: 1px solid #e5e7eb;
            }

            div.stButton > button {
                background-color: #2563eb;
                color: white;
                border-radius: 10px;
                padding: 10px 18px;
                border: none;
                font-size: 16px;
            }
            div.stButton > button:hover {
                background-color: #1d4ed8;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

apply_theme(theme)

# ==========================================================
# TITLE SECTION (Web App Style)
# ==========================================================
st.markdown("""
<div class="card">
    <h1>🌊 Flood Impact Prediction & Relief Priority Dashboard (Sri Lanka)</h1>
    <p style="font-size:16px;">
        Upload new location dataset → Predict flood probability + inundation impact → Compute relief priority → Download output.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD TRAINED MODELS
# ==========================================================
@st.cache_resource
def load_models():
    cls_model = joblib.load("flood_occurrence_model.pkl")
    reg_model = joblib.load("inundation_area_model.pkl")
    return cls_model, reg_model

cls_model, reg_model = load_models()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def safe_fill_missing(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def normalize(col):
    denom = col.max() - col.min()
    return (col - col.min()) / denom if denom != 0 else 0

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider(
    "Flood Probability Threshold (predict inundation only above this)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.success("✅ Upload CSV → Predict → View Maps → Download Output")

# ==========================================================
# FILE UPLOADER
# ==========================================================
st.subheader("📂 Upload Input Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file containing new location features", type=["csv"])

if uploaded_file is None:
    st.warning("📌 Please upload a CSV file to begin prediction.")
    st.stop()

data = pd.read_csv(uploaded_file)
st.success("✅ File uploaded successfully!")

st.write("🔍 Preview of Uploaded Dataset:")
st.dataframe(data.head())

# ==========================================================
# REQUIRED COLUMNS CHECK
# ==========================================================
required_cols = [
    "district", "latitude", "longitude",
    "population_density_per_km2",
    "infrastructure_score",
    "nearest_hospital_km",
    "nearest_evac_km"
]

missing_cols = [c for c in required_cols if c not in data.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# ==========================================================
# DROP NON-FEATURE / TARGET COLUMNS IF PRESENT
# ==========================================================
drop_cols = [
    "record_id", "place_name", "generation_date",
    "is_good_to_live", "reason_not_good_to_live",
    "flood_occurrence_current_event", "inundation_area_sqm"
]

data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

# ==========================================================
# CLEAN DATA
# ==========================================================
data = safe_fill_missing(data)

# ==========================================================
# PREDICT FLOOD PROBABILITY
# ==========================================================
st.subheader("🧠 Flood Prediction Results")
data["flood_probability"] = cls_model.predict_proba(data)[:, 1]

# ==========================================================
# PREDICT INUNDATION AREA FOR HIGH-RISK LOCATIONS
# ==========================================================
data["pred_inundation_area"] = 0.0
candidates = data["flood_probability"] >= threshold

if candidates.any():
    pred_log_area = reg_model.predict(data.loc[candidates])
    data.loc[candidates, "pred_inundation_area"] = np.expm1(pred_log_area)

st.success("✅ Flood probability + inundation predictions completed!")

# ==========================================================
# KPI SUMMARY CARDS
# ==========================================================
total_locations = len(data)
high_risk_locations = int(candidates.sum())
avg_prob = round(data["flood_probability"].mean(), 3)

col1, col2, col3 = st.columns(3)
col1.metric("📍 Total Locations", total_locations)
col2.metric("⚠️ High Risk Locations", high_risk_locations)
col3.metric("📊 Avg Flood Probability", avg_prob)

# ==========================================================
# RELIEF PRIORITY SCORE
# ==========================================================
st.subheader("🚑 Relief Priority Ranking")

data["pop_norm"] = normalize(data["population_density_per_km2"])
data["infra_norm"] = normalize(data["infrastructure_score"])
data["hospital_norm"] = normalize(data["nearest_hospital_km"])
data["evac_norm"] = normalize(data["nearest_evac_km"])
data["inund_norm"] = normalize(data["pred_inundation_area"])

data["relief_priority_score"] = (
    0.30 * data["flood_probability"] +
    0.35 * data["inund_norm"] +
    0.20 * data["pop_norm"] +
    0.10 * (1 - data["infra_norm"]) +
    0.03 * data["hospital_norm"] +
    0.02 * data["evac_norm"]
)

data["priority_level"] = pd.cut(
    data["relief_priority_score"],
    bins=[-1, 0.4, 0.7, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
)

data = data.sort_values("relief_priority_score", ascending=False)

# ==========================================================
# TOP 20 TABLE
# ==========================================================
st.write("✅ Top 20 Priority Locations")
st.dataframe(data[[
    "district", "latitude", "longitude",
    "flood_probability", "pred_inundation_area",
    "relief_priority_score", "priority_level"
]].head(20))

# ==========================================================
# MAP 1 — PRIORITY SCORE MAP
# ==========================================================
st.subheader("🗺️ Map 1: Relief Priority Score Map")

fig1, ax1 = plt.subplots(figsize=(9, 6))
sc = ax1.scatter(
    data["longitude"], data["latitude"],
    c=data["relief_priority_score"], s=15
)
plt.colorbar(sc, ax=ax1, label="Relief Priority Score")
ax1.set_title("Relief Priority Score Map")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
st.pyplot(fig1)

# ==========================================================
# DOWNLOAD OUTPUT
# ==========================================================
st.subheader("⬇️ Download Prediction Output")

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Predicted Results CSV",
    data=csv,
    file_name="predicted_relief_priority_output.csv",
    mime="text/csv"
)
