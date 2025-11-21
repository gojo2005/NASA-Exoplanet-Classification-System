import streamlit as st
import pandas as pd
import joblib
import pickle

# Load model and preprocessing info
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

medians = joblib.load("medians.pkl")
le_dict = joblib.load("label_encoders.pkl")

st.title("Exoplanet Prediction 🚀")
st.write("Input the KOI features to predict if it is a confirmed exoplanet.")

# Get feature names
feature_names = model.get_booster().feature_names

# Collect user input
user_input = {}
for feature in feature_names:
    if feature in le_dict:  # categorical
        # show text input
        user_input[feature] = st.text_input(f"{feature} (categorical)")
    else:  # numeric
        user_input[feature] = st.number_input(f"{feature}", value=medians.get(feature, 0.0))

# Button to predict
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # Apply preprocessing
    for col in le_dict:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    for col in medians:
        input_df[col] = input_df[col].fillna(medians[col])

    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"This KOI is predicted to be a PLANET! 🌟 (Confidence: {proba:.2f})")
    else:
        st.warning(f"This KOI is predicted to be NOT a planet. (Confidence: {1-proba:.2f})")
