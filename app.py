
from pathlib import Path
from datetime import date, time, datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AERONEX | Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "flight_weather_safety_model.joblib"
DATA_PATH = BASE_DIR / "weather_flight_safety.csv"

# ============================================================
# FUTURISTIC UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(0,245,255,.12), transparent 35%),
        radial-gradient(ellipse at 90% 12%, rgba(168,85,247,.14), transparent 36%),
        linear-gradient(145deg,#050812,#0a1021 55%,#070b16);
    color: #edf4ff;
    font-family: Inter, sans-serif;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b1225,#070b16);
    border-right: 1px solid rgba(0,245,255,.22);
}

h1, h2, h3 {
    font-family: Orbitron, sans-serif !important;
    letter-spacing: .7px;
}

.stApp p,
.stApp label,
div[data-testid="stWidgetLabel"] p {
    font-size: 16px;
}

.hero {
    padding: 38px;
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        rgba(0,245,255,.10),
        rgba(168,85,247,.15)
    ), rgba(12,20,39,.92);
    border: 1px solid rgba(0,245,255,.30);
    box-shadow: 0 0 35px rgba(0,245,255,.08);
    margin-bottom: 28px;
}

.eyebrow {
    color: #00f5ff;
    font-size: 14px;
    letter-spacing: 4px;
    font-weight: 800;
}

.hero-title {
    font-family: Orbitron, sans-serif;
    font-size: clamp(34px,4.2vw,58px);
    font-weight: 800;
    line-height: 1.35;
    margin: 18px 0;
    color: #ffffff;
}

.gradient-text {
    background: linear-gradient(90deg,#00f5ff,#a855f7,#ff61d8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-desc {
    color: #b3c2dc;
    font-size: 18px;
    line-height: 1.8;
    max-width: 950px;
}

.section-label {
    font-family: Orbitron, sans-serif;
    color: #00f5ff;
    font-size: 17px;
    letter-spacing: 2px;
    margin: 28px 0 18px;
}

.metric-card {
    background: linear-gradient(145deg,#14203b,#0a1122);
    border: 1px solid rgba(116,153,210,.2);
    border-radius: 18px;
    padding: 22px;
    min-height: 140px;
}

.metric-label {
    color: #a2b2cd;
    font-size: 13px;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}

.metric-value {
    font-family: Orbitron, sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin-top: 12px;
}

.metric-note {
    color: #00f5ff;
    font-size: 14px;
    margin-top: 8px;
}

.panel {
    padding: 22px;
    border-radius: 18px;
    background: rgba(14,23,43,.8);
    border: 1px solid rgba(123,155,205,.18);
    margin-bottom: 18px;
}

/* Minimal prediction result: statement only */
.result-card {
    padding: 35px 20px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(168,85,247,.13),
        rgba(0,245,255,.07)
    );
    border: 1px solid rgba(0,245,255,.30);
    text-align: center;
    margin-top: 25px;
}

.result-value {
    font-family: Orbitron, sans-serif;
    font-size: clamp(25px,3.4vw,43px);
    font-weight: 800;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg,#00d9f5,#8065ff);
    color: #050812;
    border: 0;
    border-radius: 11px;
    min-height: 52px;
    font-size: 16px !important;
    font-weight: 800;
    letter-spacing: 1px;
}

div[data-testid="stForm"] {
    background: rgba(14,23,43,.65);
    border: 1px solid rgba(0,245,255,.18);
    border-radius: 18px;
    padding: 22px;
}

div[data-testid="stDownloadButton"] button {
    border-radius: 10px;
    min-height: 45px;
}

#MainMenu,
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

model = None
df = None
model_error = None

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"

if DATA_PATH.exists():
    try:
        df = load_data(str(DATA_PATH))
    except Exception:
        df = None

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div style="text-align:center;padding:12px 4px 22px">
    <div style="font-size:46px">✈️</div>
    <div style="font-family:Orbitron,sans-serif;font-size:27px;
                font-weight:800;color:#00f5ff;letter-spacing:3px">
        AERONEX
    </div>
    <div style="font-size:13px;color:#9aacc8;letter-spacing:2px">
        FLIGHT INTELLIGENCE
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Command Center",
        "Weather Intelligence",
        "Safety Prediction",
        "Dataset Explorer",
        "Model Information"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

if model is not None:
    st.sidebar.success("MODEL CONNECTED")
else:
    st.sidebar.error("MODEL NOT LOADED")

if df is not None:
    st.sidebar.success("DATASET CONNECTED")
else:
    st.sidebar.info("DATASET NOT FOUND")

if model_error:
    with st.sidebar.expander("Model loading details"):
        st.code(model_error)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">◉ NEXT-GENERATION AVIATION ANALYTICS</div>
    <div class="hero-title">
        WEATHER-BASED<br>
        <span class="gradient-text">FLIGHT SAFETY PREDICTION</span>
    </div>
    <div class="hero-desc">
        An interactive machine-learning demonstration exploring
        weather variables, flight conditions, and binary classification.
    </div>
    <br>
    <span style="color:#00ffb3;font-weight:800;letter-spacing:2px">
        ● FLIGHT WEATHER ANALYTICS
    </span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    rows = len(df) if df is not None else 0
    columns = len(df.columns) if df is not None else 0

    safe_count = 0
    unsafe_count = 0

    if df is not None and "flight_weather_safety" in df.columns:
        safe_count = int(
            (df["flight_weather_safety"] == "Safe").sum()
        )
        unsafe_count = int(
            (df["flight_weather_safety"] == "Unsafe").sum()
        )

    unsafe_rate = (
        unsafe_count / rows * 100 if rows else 0
    )

    st.markdown(
        '<div class="section-label">◈ PROJECT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    cards = [
        ("DATA RECORDS", f"{rows:,}", "Dataset observations"),
        ("DATASET FIELDS", str(columns), "Available columns"),
        ("SAFE LABELS", f"{safe_count:,}", "Dataset category count"),
        ("UNSAFE LABEL RATE", f"{unsafe_rate:.1f}%", "Dataset distribution")
    ]

    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Weather Feature Averages")

        if df is not None:
            numeric_cols = [
                c for c in [
                    "temperature_c",
                    "wind_speed_kmh",
                    "wind_gust_kmh",
                    "visibility_km",
                    "precipitation_mm",
                    "humidity_pct",
                    "cloud_cover_pct"
                ]
                if c in df.columns
            ]

            if numeric_cols:
                means = (
                    df[numeric_cols]
                    .mean(numeric_only=True)
                    .rename_axis("Feature")
                    .reset_index(name="Mean")
                )

                fig = px.bar(
                    means,
                    x="Feature",
                    y="Mean",
                    color="Feature",
                    color_discrete_sequence=[
                        "#00f5ff", "#a855f7", "#00ffb3",
                        "#ff61d8", "#ffb84d"
                    ]
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#dce9ff",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add weather_flight_safety.csv beside app.py.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Weather Intelligence Radar")

        categories = [
            "Visibility", "Wind", "Humidity",
            "Pressure", "Cloud cover", "Precipitation"
        ]
        values = [82, 45, 68, 76, 55, 35]

        fig = go.Figure(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line=dict(color="#00f5ff", width=2),
                fillcolor="rgba(0,245,255,.15)"
            )
        )

        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor="rgba(140,160,200,.15)"
                ),
                angularaxis=dict(
                    gridcolor="rgba(140,160,200,.15)"
                )
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#dce9ff",
            showlegend=False,
            height=330
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# WEATHER INTELLIGENCE
# ============================================================

elif page == "Weather Intelligence":

    st.markdown(
        '<div class="section-label">◈ WEATHER INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    if df is None:
        st.error("Dataset file not found. Check the filename.")
    else:
        options = [
            c for c in [
                "temperature_c",
                "wind_speed_kmh",
                "wind_gust_kmh",
                "visibility_km",
                "precipitation_mm",
                "humidity_pct",
                "atmospheric_pressure_hpa",
                "cloud_cover_pct"
            ]
            if c in df.columns
        ]

        selected = st.selectbox("Select weather feature", options)
        values = pd.to_numeric(df[selected], errors="coerce").dropna()

        a, b, c = st.columns(3)

        with a:
            st.metric("MEAN", f"{values.mean():.2f}")

        with b:
            st.metric("MEDIAN", f"{values.median():.2f}")

        with c:
            st.metric("STANDARD DEVIATION", f"{values.std():.2f}")

        fig = px.histogram(
            values,
            nbins=40,
            title=f"Distribution of {selected}"
        )

        fig.update_traces(marker_color="#00d9f5")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce9ff"
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SAFE / UNSAFE PREDICTION
# ============================================================

elif page == "Safety Prediction":

    st.markdown(
        '<div class="section-label">◈ FLIGHT WEATHER CLASSIFICATION</div>',
        unsafe_allow_html=True
    )

    if model is None:
        st.error("The trained model could not be loaded.")
        st.write("Expected model file:")
        st.code(str(MODEL_PATH))

        if model_error:
            st.code(model_error)

    else:
        feature_columns = list(
            getattr(model, "feature_names_in_", [])
        )

        if not feature_columns:
            st.error(
                "The model does not expose its expected feature columns. "
                "Use the fitted pipeline saved by the supplied notebook."
            )

        else:
            airport_names = {
                "NAG": "Dr. Babasaheb Ambedkar International Airport",
                "DEL": "Indira Gandhi International Airport",
                "BOM": "Chhatrapati Shivaji Maharaj International Airport",
                "BLR": "Kempegowda International Airport",
                "MAA": "Chennai International Airport",
                "HYD": "Rajiv Gandhi International Airport",
                "CCU": "Netaji Subhas Chandra Bose International Airport",
                "COK": "Cochin International Airport",
                "PNQ": "Pune Airport",
                "AMD": "Sardar Vallabhbhai Patel International Airport"
            }

            with st.form("flight_weather_form"):

                col1, col2, col3 = st.columns(3)

                with col1:
                    flight_date = st.date_input(
                        "Flight date",
                        value=date.today()
                    )

                    departure_time = st.time_input(
                        "Scheduled departure",
                        value=time(14, 30)
                    )

                    origin_code = st.selectbox(
                        "Origin airport",
                        list(airport_names.keys()),
                        index=0
                    )

                    destination_options = [
                        code for code in airport_names
                        if code != origin_code
                    ]

                    destination_code = st.selectbox(
                        "Destination airport",
                        destination_options,
                        index=0
                    )

                    airline = st.selectbox(
                        "Airline",
                        ["IndiGo", "Air India", "Akasa Air", "SpiceJet", "Other"]
                    )

                with col2:
                    temperature = st.slider(
                        "Temperature (°C)", -8.0, 49.0, 28.0, 0.5
                    )

                    wind_speed = st.slider(
                        "Wind speed (km/h)", 0.0, 130.0, 20.0, 1.0
                    )

                    wind_gust = st.slider(
                        "Wind gust (km/h)", 0.0, 180.0, 30.0, 1.0
                    )

                    visibility = st.slider(
                        "Visibility (km)", 0.1, 30.0, 10.0, 0.5
                    )

                    wind_direction = st.slider(
                        "Wind direction (°)", 0, 359, 180
                    )

                with col3:
                    precipitation = st.slider(
                        "Precipitation (mm)", 0.0, 100.0, 0.0, 0.5
                    )

                    humidity = st.slider(
                        "Humidity (%)", 8.0, 100.0, 55.0, 1.0
                    )

                    pressure = st.slider(
                        "Atmospheric pressure (hPa)", 965.0, 1045.0, 1013.0, 0.5
                    )

                    cloud_cover = st.slider(
                        "Cloud cover (%)", 0.0, 100.0, 30.0, 1.0
                    )

                    weather_condition = st.selectbox(
                        "Weather condition",
                        ["Clear", "Cloudy", "Overcast", "Rain", "Fog", "Thunderstorm"]
                    )

                submitted = st.form_submit_button(
                    "⚡ PREDICT FLIGHT WEATHER STATUS",
                    use_container_width=True
                )

            if submitted:

                if origin_code == destination_code:
                    st.error("Choose different origin and destination airports.")

                else:
                    departure_dt = datetime.strptime(
                        departure_time.strftime("%H:%M"), "%H:%M"
                    )

                    raw_features = {
                        "origin_airport_name": airport_names[origin_code],
                        "origin_airport_code": origin_code,
                        "destination_airport_name": airport_names[destination_code],
                        "destination_airport_code": destination_code,
                        "airline": airline,
                        "temperature_c": temperature,
                        "wind_speed_kmh": wind_speed,
                        "wind_direction_deg": wind_direction,
                        "wind_gust_kmh": wind_gust,
                        "visibility_km": visibility,
                        "precipitation_mm": precipitation,
                        "humidity_pct": humidity,
                        "atmospheric_pressure_hpa": pressure,
                        "cloud_cover_pct": cloud_cover,
                        "weather_condition": weather_condition,
                        "flight_month": flight_date.month,
                        "flight_day_of_week": flight_date.weekday(),
                        "departure_hour": departure_dt.hour,
                        "departure_minute": departure_dt.minute
                    }

                    input_row = pd.DataFrame(
                        [{
                            column: raw_features.get(column, np.nan)
                            for column in feature_columns
                        }],
                        columns=feature_columns
                    )

                    try:
                        prediction = str(model.predict(input_row)[0])

                        # Only the requested result statement is displayed.
                        if prediction.strip().lower() == "safe":
                            color = "#72ffbf"
                            message = "SAFE TO TAKE THE FLIGHT"
                        else:
                            color = "#ff557a"
                            message = "UNSAFE TO TAKE THE FLIGHT"

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-value"
                                     style="color:{color}; margin:0; padding:20px 0;">
                                    {message}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # No model-label text, probability, disclaimer,
                        # or additional warning is displayed in this result area.

                    except Exception as exc:
                        st.error("Prediction failed.")
                        with st.expander("Technical error"):
                            st.exception(exc)

# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.markdown(
        '<div class="section-label">◈ DATASET EXPLORER</div>',
        unsafe_allow_html=True
    )

    if df is None:
        st.error("Dataset file not found.")
    else:
        query = st.text_input("Search dataset")

        view = df.copy()

        if query:
            mask = view.astype(str).apply(
                lambda column: column.str.contains(
                    query, case=False, na=False, regex=False
                )
            ).any(axis=1)

            view = view[mask]

        st.dataframe(
            view.head(1000),
            use_container_width=True,
            height=430
        )

        st.download_button(
            "⬇ DOWNLOAD DATASET",
            data=view.to_csv(index=False),
            file_name="filtered_flight_weather_data.csv",
            mime="text/csv"
        )

        with st.expander("Missing values and data types"):
            info = pd.DataFrame({
                "Data type": df.dtypes.astype(str),
                "Missing values": df.isna().sum()
            })
            st.dataframe(info, use_container_width=True)

        with st.expander("Descriptive statistics"):
            st.dataframe(
                df.describe(include="all").transpose(),
                use_container_width=True
            )

# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="section-label">◈ MODEL INFORMATION</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <h3>Machine Learning Workflow</h3>
            <p>
                1. Dataset inspection<br><br>
                2. Feature engineering<br><br>
                3. Stratified train/test split<br><br>
                4. Imputation and encoding<br><br>
                5. Cross-validation<br><br>
                6. Hyperparameter tuning<br><br>
                7. Holdout evaluation
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
            <h3>Evaluation Metrics</h3>
            <p>
                Accuracy: overall correct predictions.<br><br>
                Precision: correctness of positive predictions.<br><br>
                Recall: proportion of positive cases detected.<br><br>
                F1-score: harmonic mean of precision and recall.<br><br>
                Confusion matrix: correct and incorrect classifications.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:18px;color:#8294b3;
            font-size:13px;letter-spacing:1px">
    AERONEX · WEATHER-BASED FLIGHT SAFETY CLASSIFICATION
</div>
""", unsafe_allow_html=True)
