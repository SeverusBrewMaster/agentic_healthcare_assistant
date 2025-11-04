import streamlit as st
import pandas as pd
import numpy as np

# ===========================
# LOAD DATA SAFELY
# ===========================
@st.cache_data
def load_data():
    try:
        df_pairs = pd.read_csv("data/dataset.csv")
        df_desc = pd.read_csv("data/symptom_Description.csv")
    except FileNotFoundError:
        st.error("❌ Dataset files not found. Make sure they are inside the 'data' folder.")
        st.stop()

    # Normalize column names
    df_pairs.columns = [c.strip().lower() for c in df_pairs.columns]
    df_desc.columns = [c.strip().lower() for c in df_desc.columns]

    # --- Create symptom → diseases map ---
    symptom_disease_map = {}
    for _, row in df_pairs.iterrows():
        disease = row["disease"]
        for col in df_pairs.columns:
            if "symptom" in col and pd.notna(row[col]):
                symptom = str(row[col]).strip().lower()
                symptom_disease_map.setdefault(symptom, set()).add(disease)

    # --- Create disease → symptom list (for explanation) ---
    disease_symptoms_map = {}
    for _, row in df_desc.iterrows():
        disease = row["disease"]
        symptoms = [
            str(s).strip().lower()
            for col, s in row.items()
            if "symptom" in col and pd.notna(s)
        ]
        disease_symptoms_map[disease] = symptoms

    return symptom_disease_map, disease_symptoms_map


# ✅ Load once at start
symptom_disease_map, disease_symptoms_map = load_data()


# ====================================
# 🧩 Comprehensive Symptom Synonym Map
# ====================================
SYNONYM_MAP = {
    # General symptoms
    "tired": "fatigue",
    "exhausted": "fatigue",
    "weak": "fatigue",
    "tiredness": "fatigue",
    "sleepy": "fatigue",
    "lethargic": "fatigue",
    "feverish": "fever",
    "high temperature": "fever",
    "running temperature": "fever",
    "pain": "body_pain",
    "body ache": "body_pain",
    "body pain": "body_pain",

    # Respiratory
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "breathless": "breathlessness",
    "short of breath": "breathlessness",
    "difficulty breathing": "breathlessness",
    "hard to breathe": "breathlessness",
    "chest tightness": "chest_pain",
    "pain in chest": "chest_pain",
    "pressure in chest": "chest_pain",

    # Gastrointestinal
    "stomach ache": "abdominal_pain",
    "stomach pain": "abdominal_pain",
    "abdominal ache": "abdominal_pain",
    "vomiting": "nausea",
    "throwing up": "nausea",
    "nauseous": "nausea",
    "upset stomach": "nausea",
    "loose motion": "diarrhoea",
    "runny stool": "diarrhoea",
    "constipated": "constipation",

    # ENT (ear, nose, throat)
    "sore throat": "throat_irritation",
    "throat pain": "throat_irritation",
    "blocked nose": "congestion",
    "stuffy nose": "congestion",
    "runny nose": "nasal_discharge",
    "nose running": "nasal_discharge",
    "sneezing": "sneezing",

    # Skin-related
    "itchy": "itching",
    "itch": "itching",
    "rashes": "skin_rash",
    "rash": "skin_rash",
    "spots": "skin_rash",
    "skin irritation": "skin_rash",
    "red patches": "skin_rash",

    # Neurological
    "head ache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "vertigo": "dizziness",

    # Musculoskeletal
    "joint ache": "joint_pain",
    "joint stiffness": "joint_pain",
    "knee pain": "joint_pain",
    "back ache": "back_pain",
    "back pain": "back_pain",
    "shoulder pain": "joint_pain",
    "muscle ache": "muscle_pain",
    "body ache": "body_pain",

    # Psychological
    "sad": "depression",
    "depressed": "depression",
    "anxious": "anxiety",
    "nervous": "anxiety",
    "panic": "anxiety",
    "stressed": "stress",
    "insomnia": "lack_of_sleep",
    "cannot sleep": "lack_of_sleep",
    "no sleep": "lack_of_sleep",
}

# ===========================
# CORE FUNCTIONS
# ===========================
import re

def extract_symptoms(user_input):
    # Clean the input
    text = re.sub(r'[^a-zA-Z\s]', ' ', user_input.lower())
    text = re.sub(r'\s+', ' ', text).strip()

    # Apply synonym normalization
    for phrase, target in SYNONYM_MAP.items():
        if phrase in text:
            text = text.replace(phrase, target)

    found = set()

    # Match dataset symptoms (underscores handled)
    for raw_sym in symptom_disease_map.keys():
        sym_clean = raw_sym.replace("_", " ").lower().strip()
        if sym_clean in text:
            found.add(raw_sym)

    return list(found)



def predict_disease(symptoms):
    predictions = {}
    for s in symptoms:
        for d in symptom_disease_map.get(s, []):
            predictions[d] = predictions.get(d, 0) + 1

    # Weight by number of matches vs total known symptoms for each disease
    weighted = {}
    for disease, count in predictions.items():
        total = len(disease_symptoms_map.get(disease, [])) or 1
        weighted[disease] = count / total

    total_score = sum(weighted.values()) or 1
    probs = {k: round(v / total_score, 2) for k, v in weighted.items()}
    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))



def explain_disease(disease):
    if disease in disease_symptoms_map:
        syms = ", ".join(disease_symptoms_map[disease][:5])
        return f"Common symptoms include: {syms}"
    return "No detailed description available."


def generate_explanation(predictions, symptoms):
    if not predictions:
        return "No matching diseases found. Please describe more symptoms."

    exp = f"Based on your symptoms ({', '.join(symptoms)}), the system suggests:\n\n"
    for disease, prob in predictions.items():
        extra = explain_disease(disease)
        exp += f"• **{disease}** — {prob*100}% match\n    ↳ {extra}\n"
    exp += "\n⚠️ *This is not a diagnosis. Please consult a healthcare professional.*"
    return exp


# ===========================
# STREAMLIT UI
# ===========================
st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🧠")

st.title("🧠 Agentic Healthcare Assistant")
st.markdown(
    "Welcome to your **Agentic Healthcare Assistant**. "
    "Describe your symptoms below, and the assistant will suggest possible conditions based on dataset knowledge."
)

user_input = st.text_area("🩺 Enter your symptoms (e.g., fever, cough, fatigue):", height=120)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some symptoms.")
    else:
        symptoms = extract_symptoms(user_input)
        if not symptoms:
            st.error("⚠️ No recognizable symptoms found. Try listing them with commas (e.g., fever, headache).")
        else:
            preds = predict_disease(symptoms)

            if not preds:
                st.warning("No likely diseases found. Please describe more symptoms.")
            else:
                st.markdown(f"### 🧩 Based on your symptoms: `{', '.join(symptoms)}`")

                for disease, prob in preds.items():
                    with st.container():
                        st.markdown(f"**🧠 {disease}**")
                        st.progress(prob)
                        desc = explain_disease(disease)
                        st.caption(desc)

                st.markdown("---")
                st.info(
                    "⚠️ **Disclaimer:** This is not a medical diagnosis. Please consult a qualified healthcare professional."
                )
