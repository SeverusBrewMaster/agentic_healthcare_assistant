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
# STREAMLIT PAGE CONFIG (MUST BE FIRST)
# ===============================
st.set_page_config(
    page_title="Agentic Healthcare Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===============================
# CUSTOM CSS FOR LIGHT MODE
# ===============================
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom card styling */
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-left: 4px solid #4a90e2;
    }
    
    /* Header styling */
    .custom-header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .custom-subheader {
        color: #5a6c7d;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        text-align: center;
        border-top: 3px solid #4a90e2;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4a90e2;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border: 2px solid #e1e8ed;
        border-radius: 8px;
        font-size: 1rem;
        padding: 12px;
    }
    
    .stTextArea textarea:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Remove default streamlit branding space */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


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
    """
    if not symptoms:
        return []

    x_input = np.zeros(len(all_symptoms))
    for s in symptoms:
        if s in all_symptoms:
            x_input[all_symptoms.index(s)] = 1
    x_input = np.array([x_input])

    probs = model.predict_proba(x_input)[0]
    class_idx = np.argmax(probs)
    base_prob = probs[class_idx]

    importance_scores = []

    for s in symptoms:
        x_temp = x_input.copy()
        x_temp[0, all_symptoms.index(s)] = 0
        new_prob = model.predict_proba(x_temp)[0][class_idx]
        diff = base_prob - new_prob
        importance_scores.append((s, round(diff, 4)))

    importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return importance_scores

# ===============================
# 6️⃣ PRECautions (Optional)
# ===============================

def get_precautions(disease):
    """Fetch precaution list for the given disease (case-insensitive, trimmed)."""
    if precaution_df is None or precaution_df.empty:
        return []

    # Normalize columns and disease names once
    precaution_df.columns = [c.strip().lower() for c in precaution_df.columns]
    precaution_df["disease"] = precaution_df["disease"].astype(str).str.strip().str.lower()

    target = disease.strip().lower()

    # Check match
    matches = precaution_df[precaution_df["disease"] == target]

    if matches.empty:
        # Try fuzzy partial match (e.g., "diabetes" vs "diabetes mellitus")
        matches = precaution_df[precaution_df["disease"].str.contains(target, case=False, na=False)]

    if not matches.empty:
        row = matches.iloc[0]
        # Return all non-empty precaution columns
        return [
            str(v).strip().capitalize()
            for k, v in row.items()
            if "precaution" in k and pd.notna(v) and str(v).strip()
        ]

    return []


# ===============================
# 6️⃣ SIDEBAR
# ===============================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/brain.png", width=80)
    st.markdown("### 🩺 About This Tool")
    st.info("""
    This AI-powered assistant analyzes your symptoms and predicts potential diseases using machine learning.
    
    **Features:**
    - 🧠 ML-based prediction
    - 📊 Explainable AI
    - 💊 Precaution recommendations
    - 🔍 Interactive visualizations
    """)
    
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. Describe your symptoms
    2. Click 'Analyze Symptoms'
    3. Review predictions
    4. Check explanations
    """)
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a qualified healthcare professional.")


# ===============================
# 7️⃣ MAIN UI
# ===============================

# Header
st.markdown('<h1 class="custom-header">🧠 Agentic Healthcare Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="custom-subheader">AI-Powered Symptom Analysis & Disease Prediction System</p>', unsafe_allow_html=True)

st.markdown("---")

# Input Section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🩺 Enter Your Symptoms")
    user_input = st.text_area(
        label="Describe your symptoms below:",
        placeholder="Example: I have a fever, cough, and body pain...",
        height=150,
        label_visibility="collapsed"
    )
    
    analyze_button = st.button("🔍 Analyze Symptoms")

st.markdown("---")

# Analysis Section
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please describe your symptoms first.")
        st.stop()

    with st.spinner("🔄 Analyzing your symptoms..."):
        symptoms = extract_symptoms(user_input)
        
        if not symptoms:
            st.error("❌ No recognizable symptoms found. Try listing them clearly (e.g., fever, cough, headache).")
            st.stop()

        # Detected Symptoms Card
        st.markdown("### 🩹 Detected Symptoms")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #4a90e2; margin-bottom: 15px;">Identified Symptoms:</h4>
                <p style="font-size: 1.1rem; color: #2c3e50;">
                    <strong>{', '.join(symptoms)}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Symptoms</div>
                <div class="metric-value">{len(symptoms)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Prediction
        top_disease, preds_df = predict_with_model(symptoms)
        
        if preds_df is None:
            st.warning("⚠️ No disease prediction available.")
            st.stop()

        # Results in Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🔍 AI Explanation", "💊 Precautions"])
        
        # Replace the two plotly chart sections in tab1 and tab2 with this:

        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="custom-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-left: none;">
                    <h3 style="margin: 0; color: white;">Most Likely Disease</h3>
                    <h2 style="margin-top: 15px; font-size: 2rem; color: white;">{top_disease}</h2>
                    <p style="margin-top: 10px; opacity: 0.9;">Confidence: {preds_df.iloc[0]['Probability']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # High contrast color scale for light background
                fig = px.bar(
                    preds_df,
                    x="Probability",
                    y="Disease",
                    orientation="h",
                    color="Probability",
                    range_x=[0, 1],
                    color_continuous_scale=[[0, '#e74c3c'], [0.5, '#f39c12'], [1, '#27ae60']],  # Red to Orange to Green
                    title="Top 5 Disease Predictions",
                    labels={"Probability": "Confidence Score", "Disease": ""}
                )
                fig.update_layout(
                    showlegend=False,
                    height=350,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=13, color='#2c3e50', family="Arial"),
                    xaxis=dict(
                        gridcolor='#e1e8ed',
                        linecolor='#34495e',
                        linewidth=2
                    ),
                    yaxis=dict(
                        gridcolor='#e1e8ed',
                        linecolor='#34495e',
                        linewidth=2
                    ),
                    title_font=dict(size=16, color='#2c3e50', family="Arial Bold")
                )
                fig.update_traces(
                    marker=dict(
                        line=dict(color='#34495e', width=1)
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            imp = explain_prediction(symptoms)
            if imp:
                st.markdown("#### 🧠 How Each Symptom Influences the Prediction")
                imp_df = pd.DataFrame(imp, columns=["Symptom", "Influence"])
                
                # High contrast diverging color scale
                fig2 = px.bar(
                    imp_df,
                    x="Influence",
                    y="Symptom",
                    orientation="h",
                    color="Influence",
                    color_continuous_scale=[
                        [0, '#c0392b'],      # Dark red for negative
                        [0.5, '#95a5a6'],    # Gray for neutral
                        [1, '#16a085']       # Teal for positive
                    ],
                    title="Symptom Impact Analysis"
                )
                fig2.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=13, color='#2c3e50', family="Arial"),
                    xaxis=dict(
                        gridcolor='#e1e8ed',
                        linecolor='#34495e',
                        linewidth=2,
                        zeroline=True,
                        zerolinecolor='#7f8c8d',
                        zerolinewidth=2
                    ),
                    yaxis=dict(
                        gridcolor='#e1e8ed',
                        linecolor='#34495e',
                        linewidth=2
                    ),
                    title_font=dict(size=16, color='#2c3e50', family="Arial Bold")
                )
                fig2.update_traces(
                    marker=dict(
                        line=dict(color='#34495e', width=1)
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                st.info("💡 **Interpretation:** Higher values indicate symptoms that have a stronger influence on the predicted disease.")
            else:
                st.info("ℹ️ No explainable analysis available for the given symptoms.")
        
            # ===== TAB 3 : PRECAUTIONS =====
            with tab3:
                st.markdown(f"#### 💊 Recommended Precautions for **{top_disease}**")
            
                # Get precautions using the improved lookup
                precautions_list = get_precautions(top_disease)
            
                if precautions_list:
                    # Split into two columns
                    left_col, right_col = st.columns([3, 2])
            
                    with left_col:
                        st.markdown("##### 📋 Action Steps")
                        for idx, precaution in enumerate(precautions_list, 1):
                            # Choose icon based on content
                            icon = "🩺"
                            text = precaution.lower()
                            if any(w in text for w in ["consult", "doctor", "hospital", "care"]):
                                icon = "👨‍⚕️"
                            elif any(w in text for w in ["avoid", "stop", "limit", "reduce"]):
                                icon = "🚫"
                            elif any(w in text for w in ["drink", "water", "fluid", "hydrate"]):
                                icon = "💧"
                            elif any(w in text for w in ["exercise", "walk", "activity", "physical"]):
                                icon = "🏃"
                            elif any(w in text for w in ["rest", "sleep", "relax"]):
                                icon = "😴"
                            elif any(w in text for w in ["food", "eat", "diet", "meal"]):
                                icon = "🍎"
                            elif any(w in text for w in ["medicine", "medication", "drug", "antibiotic"]):
                                icon = "💊"
                            elif any(w in text for w in ["wash", "clean", "hygiene"]):
                                icon = "🧼"
            
                            st.markdown(f"""
                            <div class="custom-card" style="background: linear-gradient(90deg, #f8f9fa 0%, white 100%); border-left-color: #27ae60; margin-bottom: 15px;">
                                <div style="display: flex; align-items: center;">
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                color: white; 
                                                width: 35px; 
                                                height: 35px; 
                                                border-radius: 50%; 
                                                display: flex; 
                                                align-items: center; 
                                                justify-content: center; 
                                                font-weight: bold; 
                                                margin-right: 15px; 
                                                flex-shrink: 0;">
                                        {idx}
                                    </div>
                                    <div style="flex-grow: 1;">
                                        <h4 style="color: #2c3e50; margin: 0; font-size: 1.1rem;">
                                            {icon} {precaution.capitalize()}
                                        </h4>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
                    with right_col:
                        st.markdown("##### 📊 Precaution Overview")
            
                        # Categorize precaution types
                        medical_care = sum(1 for p in precautions_list if any(w in p.lower() for w in ["consult", "doctor", "hospital", "medication", "antibiotic"]))
                        lifestyle = sum(1 for p in precautions_list if any(w in p.lower() for w in ["exercise", "rest", "sleep", "avoid", "diet", "food"]))
                        hygiene = sum(1 for p in precautions_list if any(w in p.lower() for w in ["wash", "clean", "hygiene", "water"]))
            
                        precaution_types = []
                        type_counts = []
                        colors_list = []
            
                        if medical_care > 0:
                            precaution_types.append("Medical Care")
                            type_counts.append(medical_care)
                            colors_list.append("#e74c3c")
                        if lifestyle > 0:
                            precaution_types.append("Lifestyle")
                            type_counts.append(lifestyle)
                            colors_list.append("#3498db")
                        if hygiene > 0:
                            precaution_types.append("Hygiene")
                            type_counts.append(hygiene)
                            colors_list.append("#2ecc71")
            
                        other = len(precautions_list) - (medical_care + lifestyle + hygiene)
                        if other > 0:
                            precaution_types.append("Other")
                            type_counts.append(other)
                            colors_list.append("#95a5a6")
            
                        if precaution_types:
                            fig_precaution = px.pie(
                                names=precaution_types,
                                values=type_counts,
                                title="Precaution Categories",
                                color_discrete_sequence=colors_list
                            )
                            fig_precaution.update_layout(
                                height=300,
                                paper_bgcolor='white',
                                font=dict(size=12, color='#2c3e50'),
                                showlegend=True,
                                legend=dict(
                                    orientation="v",
                                    yanchor="middle",
                                    y=0.5,
                                    xanchor="left",
                                    x=1.05
                                )
                            )
                            fig_precaution.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                marker=dict(line=dict(color='#34495e', width=2))
                            )
                            st.plotly_chart(fig_precaution, use_container_width=True)
            
                        st.markdown(f"""
                        <div class="metric-card" style="margin-top: 20px;">
                            <div class="metric-label">Total Actions</div>
                            <div class="metric-value" style="font-size: 2.5rem;">{len(precautions_list)}</div>
                            <p style="color: #7f8c8d; font-size: 0.85rem; margin-top: 10px;">Follow these steps for better recovery</p>
                        </div>
                        """, unsafe_allow_html=True)
            
                else:
                    st.info(f"ℹ️ No specific precautions available for **{top_disease}** in the dataset.")



st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p>© 2025 Agentic Healthcare Assistant | Educational Purpose Only</p>
    <p style="font-size: 0.85rem;">Built with ❤️ using Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
