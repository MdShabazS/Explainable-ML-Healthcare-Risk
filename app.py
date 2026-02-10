import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Load and train model (simple)
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data/raw/heart.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X

model, scaler, feature_df = train_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Explainable Healthcare Risk Prediction")
st.write("Predict heart disease risk with transparent ML explanations.")

st.sidebar.header("Patient Details")

user_input = []
for col in feature_df.columns:
    value = st.sidebar.number_input(f"{col}", value=float(feature_df[col].mean()))
    user_input.append(value)

user_input = np.array(user_input).reshape(1, -1)
user_scaled = scaler.transform(user_input)

# -----------------------------
# Prediction
# -----------------------------
risk_prob = model.predict_proba(user_scaled)[0][1]

st.subheader("🩺 Prediction Result")
st.write(f"**Heart Disease Risk Probability:** `{risk_prob:.2f}`")

if risk_prob > 0.5:
    st.error("⚠️ High Risk Detected")
else:
    st.success("✅ Low Risk Detected")

# -----------------------------
# Explainability
# -----------------------------
st.subheader("🔍 Model Explanation (Why this prediction?)")

explainer = shap.Explainer(
    model,
    pd.DataFrame(
        scaler.transform(feature_df),
        columns=feature_df.columns
    )
)

user_scaled_df = pd.DataFrame(
    user_scaled,
    columns=feature_df.columns
)

shap_values = explainer(user_scaled_df)


fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)


explainer = shap.Explainer(model, scaler.transform(feature_df))
shap_values = explainer(user_scaled)

shap_df = pd.DataFrame({
    "Feature": feature_df.columns,
    "Impact": shap_values.values[0]
}).sort_values(by="Impact", ascending=False)

st.bar_chart(shap_df.set_index("Feature"))
