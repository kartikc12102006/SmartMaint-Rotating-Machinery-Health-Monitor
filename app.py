import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.fft import fft

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Rotating Machinery Health Monitor", page_icon="⚙️")

st.title("⚙️ Predictive Maintenance of Rotating Machinery")
st.write("Upload vibration signal CSV to identify machine condition")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

def extract_features(signal):
    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.sqrt(np.mean(signal**2)),
        np.var(signal),
        np.max(fft_vals),
        np.mean(fft_vals),
        np.std(fft_vals)
    ]

labels = {
    0: "✅ Healthy",
    1: "⚠️ Unbalance Fault",
    2: "⚠️ Misalignment Fault",
    3: "⚠️ Bearing Outer Race Fault",
    4: "⚠️ Bearing Inner Race Fault",
    5: "⚠️ Gear Mesh Fault"
}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col = st.selectbox("Select Vibration Signal Column", df.columns[1:])
    
    signal = df[col].values
    features = extract_features(signal)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]

    st.subheader("🧠 Machine Health Prediction")
    st.success(labels[prediction])

    st.line_chart(signal)

st.caption("⚠️ Educational & predictive maintenance purpose only")
