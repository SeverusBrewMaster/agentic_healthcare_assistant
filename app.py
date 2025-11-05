import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import plotly.express as px
import re
import os

# ===============================
# 1️⃣ LOAD MODEL + DATASETS
# ===============================

@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load("model/disease_model.joblib")
        label_encoder = joblib.load("model/label_encoder.joblib")
        with open("model/symptom_list.json", "r") as f:
            symptom_list = json.load(f)
    except Exception as e:
        st.error(f"❌ Model files not found or invalid: {e}")
        st.stop()

    # Optional precaution dataset
    precaution_df = None
    try:
        precaution_df = pd.read_csv("data/symptom_precaution.csv")
        precaution_df.columns = [c.strip().lower() for c in precaution_df.columns]
    except Exception:
        pass

    return model, label_encoder, symptom_list, precaution_df


model, le, all_symptoms, precaution_df = load_model_and_data()

# ===============================
# 2️⃣ SYNONYM MAP (same as before)
# ===============================

SYNONYM_MAP = {
    "tired": "fatigue", "exhausted": "fatigue", "weak": "fatigue",
    "feverish": "fever", "high temperature": "fever", "pain": "body_pain",
    "coughing": "cough", "short of breath": "breathlessness",
    "stomach ache": "abdominal_pain", "vomiting": "nausea",
    "sore throat": "throat_irritation", "itchy": "itching", "rashes": "skin_rash",
    "head ache": "headache", "dizzy": "dizziness", "joint ache": "joint_pain",
    "back ache": "back_pain", "sad": "depression", "anxious": "anxiety",
}

# ===============================
# 3️⃣ SYMPTOM EXTRACTION (safe)
# ===============================

MULTIWORD_TOKEN_MATCH_RATIO = 0.75

def extract_symptoms(user_input):
    text = re.sub(r'[^a-zA-Z\s]', ' ', user_input.lower())
    text = re.sub(r'\s+', ' ', text).strip()

    for phrase, target in SYNONYM_MAP.items():
        pattern = r'\b' + re.escape(phrase) + r'\b'
        text = re.sub(pattern, target, text)

    found = []
    for sym in all_symptoms:
        sym_clean = sym.replace("_", " ").lower()
        tokens = sym_clean.split()
        if len(tokens) == 1:
            if re.search(rf"\b{sym_clean}\b", text):
                found.append(sym)
        else:
            match_tokens = sum(1 for t in tokens if t in text)
            if match_tokens / len(tokens) >= MULTIWORD_TOKEN_MATCH_RATIO:
                found.append(sym)

    return sorted(list(set(found)))


# ===============================
# 4️⃣ DISEASE PREDICTION
# ===============================

def predict_with_model(symptoms):
    """Predict disease using trained ML model."""
    if not symptoms:
        return None, None

    x_input = np.zeros(len(all_symptoms))
    for s in symptoms:
        if s in all_symptoms:
            idx = all_symptoms.index(s)
            x_input[idx] = 1

    probs = model.predict_proba([x_input])[0]
    diseases = le.inverse_transform(np.arange(len(probs)))

    df = pd.DataFrame({"Disease": diseases, "Probability": probs})
    df = df.sort_values(by="Probability", ascending=False)
    top_disease = df.iloc[0]["Disease"]

    return top_disease, df.head(5)


# ===============================
# 5️⃣ EXPLAINABLE AI (SHAP)
# ===============================

def explain_prediction(symptoms):
    """
    Compute simple feature importance by toggling each symptom
    and measuring change in predicted probability.
    This is a model-agnostic, human-readable explanation.
    """
    import numpy as np

    if not symptoms:
        return []

    # Create baseline input
    x_input = np.zeros(len(all_symptoms))
    for s in symptoms:
        if s in all_symptoms:
            x_input[all_symptoms.index(s)] = 1
    x_input = np.array([x_input])

    # Get predicted probabilities
    probs = model.predict_proba(x_input)[0]
    class_idx = np.argmax(probs)
    base_prob = probs[class_idx]

    importance_scores = []

    # Toggle each symptom off to measure its effect
    for s in symptoms:
        x_temp = x_input.copy()
        x_temp[0, all_symptoms.index(s)] = 0
        new_prob = model.predict_proba(x_temp)[0][class_idx]
        diff = base_prob - new_prob
        importance_scores.append((s, round(diff, 4)))

    # Sort by absolute influence
    importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return importance_scores


# ===============================
# 7️⃣ STREAMLIT UI
# ===============================

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🧠")
st.title("🧠 Agentic Healthcare Assistant")

st.markdown("""
This intelligent system analyzes your symptoms using an AI model trained on a medical dataset.  
It provides possible disease predictions, precautions, and explainable insights using SHAP values.  
⚠️ *This is not a diagnosis. Consult a qualified doctor for medical advice.*
""")

user_input = st.text_area("🩺 Describe your symptoms (e.g., skin rash, fatigue, headache):", height=120)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please describe your symptoms first.")
        st.stop()

    symptoms = extract_symptoms(user_input)
    if not symptoms:
        st.error("No recognizable symptoms found. Try listing them with commas (e.g., fever, cough).")
        st.stop()

    st.markdown(f"### 🩹 Detected Symptoms: `{', '.join(symptoms)}`")

    top_disease, preds_df = predict_with_model(symptoms)
    if preds_df is None:
        st.warning("No disease prediction available.")
        st.stop()

    # Show top 5 predictions
    st.markdown("### 🧠 Predicted Disease Probabilities:")
    fig = px.bar(preds_df, x="Probability", y="Disease", orientation="h",
                 color="Probability", range_x=[0, 1], color_continuous_scale="Blues_r")
    st.plotly_chart(fig, use_container_width=True)

    # Top prediction details
    st.success(f"**Most Likely Disease:** {top_disease}")



    # Explainable AI visualization
    st.markdown("### 🔍 Explainable AI — Symptom Influence Analysis:")
    imp = explain_prediction(symptoms)
    if imp:
        imp_df = pd.DataFrame(imp, columns=["Symptom", "Influence"])
        fig2 = px.bar(
            imp_df,
            x="Influence",
            y="Symptom",
            orientation="h",
            color="Influence",
            color_continuous_scale="RdBu",
            title="How Each Symptom Affects the Predicted Disease",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No explainable analysis available for the given symptoms.")
    

st.markdown("---")
st.caption("© 2025 Agentic Healthcare Assistant — Educational Purpose Only.")
