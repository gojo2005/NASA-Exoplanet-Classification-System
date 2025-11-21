import streamlit as st
import pandas as pd
import pickle
import joblib

# ---------- Load model and preprocessing ----------
with open(r"C:\Users\LENOVO\Desktop\Nasaproject\model.pkl", "rb") as f:
    model = pickle.load(f)

medians = joblib.load(r"C:\Users\LENOVO\Desktop\Nasaproject\medians.pkl")
le_dict = joblib.load(r"C:\Users\LENOVO\Desktop\Nasaproject\label_encoders.pkl")

st.title("Exoplanet Prediction 🚀")
st.write("Input the KOI features to predict if it is a confirmed exoplanet.")

# ---------- Feature names ----------
feature_names = model.get_booster().feature_names  # all features used in training

# ---------- Friendly display names ----------
# Fill in all your column names here. Example below for common ones:
feature_display_names = {
    "kepid": "KepID",
    "kepoi_name": "KOI Name",
    "kepler_name": "Kepler Name",
    "koi_disposition": "Exoplanet Archive Disposition",
    "koi_pdisposition": "Disposition Using Kepler Data",
    "koi_score": "Disposition Score",
    "koi_fpflag_nt": "Not Transit-Like False Positive Flag",
    "koi_fpflag_ss": "Stellar Eclipse False Positive Flag",
    "koi_fpflag_co": "Centroid Offset False Positive Flag",
    "koi_fpflag_ec": "Ephemeris Contamination False Positive Flag",
    "koi_period": "Orbital Period [days]",
    "koi_period_err1": "Orbital Period Upper Unc. [days]",
    "koi_period_err2": "Orbital Period Lower Unc. [days]",
    "koi_time0bk": "Transit Epoch [BKJD]",
    "koi_time0bk_err1": "Transit Epoch Upper Unc. [BKJD]",
    "koi_time0bk_err2": "Transit Epoch Lower Unc. [BKJD]",
    "koi_impact": "Impact Parameter",
    "koi_impact_err1": "Impact Parameter Upper Unc.",
    "koi_impact_err2": "Impact Parameter Lower Unc.",
    "koi_duration": "Transit Duration [hrs]",
    "koi_duration_err1": "Transit Duration Upper Unc. [hrs]",
    "koi_duration_err2": "Transit Duration Lower Unc. [hrs]",
    "koi_depth": "Transit Depth [ppm]",
    "koi_depth_err1": "Transit Depth Upper Unc. [ppm]",
    "koi_depth_err2": "Transit Depth Lower Unc. [ppm]",
    "koi_prad": "Planetary Radius [Earth radii]",
    "koi_prad_err1": "Planetary Radius Upper Unc. [Earth radii]",
    "koi_prad_err2": "Planetary Radius Lower Unc. [Earth radii]",
    "koi_teq": "Equilibrium Temperature [K]",
    "koi_teq_err1": "Equilibrium Temperature Upper Unc. [K]",
    "koi_teq_err2": "Equilibrium Temperature Lower Unc. [K]",
    "koi_insol": "Insolation Flux [Earth flux]",
    "koi_insol_err1": "Insolation Flux Upper Unc. [Earth flux]",
    "koi_insol_err2": "Insolation Flux Lower Unc. [Earth flux]",
    "koi_model_snr": "Transit Signal-to-Noise",
    "koi_tce_plnt_num": "TCE Planet Number",
    "koi_tce_delivname": "TCE Delivery",
    "koi_steff": "Stellar Effective Temperature [K]",
    "koi_steff_err1": "Stellar Effective Temperature Upper Unc. [K]",
    "koi_steff_err2": "Stellar Effective Temperature Lower Unc. [K]",
    "koi_slogg": "Stellar Surface Gravity [log10(cm/s^2)]",
    "koi_slogg_err1": "Stellar Surface Gravity Upper Unc. [log10(cm/s^2)]",
    "koi_slogg_err2": "Stellar Surface Gravity Lower Unc. [log10(cm/s^2)]",
    "koi_srad": "Stellar Radius [Solar radii]",
    "koi_srad_err1": "Stellar Radius Upper Unc. [Solar radii]",
    "koi_srad_err2": "Stellar Radius Lower Unc. [Solar radii]",
    "ra": "RA [decimal degrees]",
    "dec": "Dec [decimal degrees]",
    "koi_kepmag": "Kepler-band [mag]"
}

# ---------- Collect user input ----------
user_input = {}
for feature in feature_names:
    display_name = feature_display_names.get(feature, feature)  # fallback to column name
    if feature in le_dict:
        user_input[feature] = st.text_input(f"{display_name} (categorical)")
    else:
        user_input[feature] = st.number_input(f"{display_name}", value=medians.get(feature, 0.0))

input_df = pd.DataFrame([user_input])

# ---------- Preprocessing ----------
# Label encode categorical inputs
for col in le_dict:
    if col in input_df.columns:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

# Fill numeric missing values
for col in medians:
    if col in input_df.columns:
        input_df[col] = input_df[col].fillna(medians[col])

# ---------- Prediction ----------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"This KOI is predicted to be an EXOPLANET! 🌟 (Confidence: {proba:.2f})")
    else:
        st.warning(f"This KOI is predicted to be NOT an EXOPLANET. (Confidence: {1-proba:.2f})")
