import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk

# Load models and encoders
accident_model = joblib.load('accident_classifier.pkl')
severity_model = joblib.load('severity_model.pkl')
road_encoder = joblib.load('road_condition_encoder.pkl')
severity_encoder = joblib.load('severity_encoder.pkl')

# Set page config
st.set_page_config(page_title="Accident Detection", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚗 Real-Time Accident Detection & Severity Estimator</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 16px;'>
Enter vehicle and road details below to predict:
<br>- Accident likelihood
<br>- Severity level (if any)
</div><br>
""", unsafe_allow_html=True)

# --- Centered Form ---
with st.form("prediction_form"):
    st.markdown("### 📥 Vehicle Sensor Input")

    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Speed (km/h)", 0, 160, 60)
        impact_force = st.slider("Impact Force (0-10)", 0.0, 10.0, 2.5, step=0.1)
    with col2:
        acceleration = st.slider("Acceleration (m/s²)", 0.0, 10.0, 3.0, step=0.1)
        road_condition = st.selectbox("Road Condition", road_encoder.classes_)

    submitted = st.form_submit_button("🔍 Predict Accident & Severity")

# --- Prediction Logic ---
if submitted:
    road_encoded = road_encoder.transform([road_condition])[0]
    input_vector = np.array([[speed, acceleration, impact_force, road_encoded]])

    is_accident = accident_model.predict(input_vector)[0]
    accident_prob = accident_model.predict_proba(input_vector)[0][1] * 100

    if is_accident:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.error("⚠️ Accident Detected!")
        st.info(f"Confidence: {accident_prob:.2f}%")
        severity_encoded = severity_model.predict(input_vector)[0]
        severity_label = severity_encoder.inverse_transform([severity_encoded])[0]
        st.subheader("🩺 Estimated Severity:")
        if severity_label == 'low':
            st.warning("Low")
        elif severity_label == 'medium':
            st.warning("Medium")
        else:
            st.error("High")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.success("✅ No Accident Detected.")
        st.info(f"Confidence: {100 - accident_prob:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # 🌍 Map of Simulated Accident Hot Zones (Displayed After Prediction)
    # -------------------------------
    st.markdown("## 🌍 Accident Hot Zones Map")

    accident_data = pd.DataFrame({
        'lat': np.random.uniform(28.60, 28.70, 30),
        'lon': np.random.uniform(77.20, 77.35, 30),
        'severity': np.random.choice(['low', 'medium', 'high'], 30)
    })

    accident_data['color'] = accident_data['severity'].map({
        'low': [255, 215, 0],
        'medium': [255, 140, 0],
        'high': [255, 0, 0]
    })

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=pdk.ViewState(
            latitude=28.65,
            longitude=77.28,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=accident_data,
                get_position='[lon, lat]',
                get_color='color',
                get_radius=150,
                pickable=True,
            )
        ],
    ))

    st.caption("🗺️ Map shows simulated data. Hook into real-time GPS for live updates.")
