import streamlit as st
import numpy as np
import joblib

# Load Model + Scalers

model = joblib.load("best_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Prediction Function

def predict_energy(input_data):
    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler_X.transform(input_array)

    # Predict
    pred_scaled = model.predict(input_scaled)

    # Inverse scale output
    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

    return prediction[0][0]

# Streamlit UI

st.set_page_config(page_title="Energy Prediction", layout="centered")

st.title("⚡ Smart Building Energy Prediction")
st.write("Predict energy consumption based on environmental conditions")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Weather Inputs
T2M = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
T2M_MIN = st.sidebar.slider("Min Temperature", 0, 50, 23)
T2M_MAX = st.sidebar.slider("Max Temperature", 0, 50, 27)

# Other Inputs
RH2M = st.sidebar.slider("Humidity (%)", 0, 100, 60)
WS2M = st.sidebar.slider("Wind Speed", 0, 20, 2)

# Feature Vector

input_data = [
    2, 5, 3, 0, RH2M,
    T2M, T2M_MIN, T2M_MAX, (T2M_MAX - T2M_MIN),
    200, 0, 0,
    0.5, 0.5, 0.5, 0.5,
    WS2M
]

# Prediction Button

if st.button("🔍 Predict Energy"):
    energy = predict_energy(input_data)

    st.success(f"⚡ Predicted Energy Consumption: {energy:.2f} kWh")

    # Optimization Status

    st.subheader("📊 Optimization Status")

    if energy < 100:
        st.success("✅ Optimized (Low Energy Consumption)")
    elif 100 <= energy < 200:
        st.warning("⚠️ Moderate Energy Usage")
    else:
        st.error("❌ High Energy Consumption")

    # Insights

    st.subheader("💡 Insights")

    if T2M > 35:
        st.warning("High temperature → Cooling load increases ❗")
    elif T2M < 20:
        st.info("Low temperature → Energy consumption may reduce")
    else:
        st.info("Moderate temperature → Balanced energy usage")
