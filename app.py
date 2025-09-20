import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.title("European Farm Output Predictor (€)")

# Load artifacts
MODEL_PATH = Path("models/farm_output_model.pkl")
META_PATH = Path("assets/model_meta.json")

if not MODEL_PATH.exists():
    st.error("Model file not found at models/farm_output_model.pkl")
    st.stop()

if not META_PATH.exists():
    st.error("Metadata file not found at assets/model_meta.json")
    st.stop()

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
FEATURES = meta["features"]

st.caption("Enter values and click Predict. The model was trained on 2016 Eurostat country-level data.")

# Reasonable default ranges (conservative so the app runs even if you haven't opened the CSV)
area = st.number_input("Used Agricultural Area (ha)", min_value=0, value=50000, step=1000)
awu  = st.number_input("Total Labour (AWU)",       min_value=0, value=100000, step=1000)
mgr  = st.number_input("Managers with Full Training", min_value=0, value=20000, step=500)

if st.button("Predict Output (€)"):
    # Build a row in the exact feature order
    x_new = pd.DataFrame([[area, awu, mgr]], columns=FEATURES)
    pred = model.predict(x_new)[0]
    st.success(f"Predicted Standard Output: €{pred:,.0f}")
