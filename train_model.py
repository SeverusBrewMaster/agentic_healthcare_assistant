"""
train_model.py — Enhanced version
---------------------------------
Trains an optimized ML model (RandomForest + NaiveBayes comparison)
to predict diseases from symptoms using the Kaggle Disease-Symptom dataset.

Includes:
✅ Cross-validation
✅ Anti-overfitting regularization
✅ Rare-symptom filtering
✅ Accuracy, F1-score, CV metrics
✅ Model saving to /model/
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import joblib

# ==========================
# CONFIG
# ==========================
DATA_PATH = "data/dataset.csv"
OUT_DIR = "model"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# ==========================
# 1️⃣ LOAD AND CLEAN DATA
# ==========================
print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]

symptom_cols = [c for c in df.columns if "symptom" in c]
if not symptom_cols:
    raise ValueError("No symptom columns found in dataset.csv")

def clean_symptom(s):
    return str(s).strip().lower().replace(" ", "_")

# Build unique symptom list
all_symptoms = set()
for _, row in df.iterrows():
    for c in symptom_cols:
        val = row.get(c)
        if pd.notna(val) and str(val).strip():
            all_symptoms.add(clean_symptom(val))

all_symptoms = sorted(list(all_symptoms))
print(f"✅ Found {len(all_symptoms)} unique symptoms.")

# ==========================
# 2️⃣ BUILD FEATURE MATRIX
# ==========================
print("🔹 Building feature matrix...")
X = []
for _, row in df.iterrows():
    row_syms = set()
    for c in symptom_cols:
        val = row.get(c)
        if pd.notna(val) and str(val).strip():
            row_syms.add(clean_symptom(val))
    features = [1 if sym in row_syms else 0 for sym in all_symptoms]
    X.append(features)

X = np.array(X)

# Encode disease labels
label_col = next((c for c in df.columns if "disease" in c), "disease")
y_raw = df[label_col].astype(str).str.strip()
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"✅ Encoded {len(le.classes_)} diseases.")

# ==========================
# 3️⃣ REMOVE RARE SYMPTOMS
# ==========================
symptom_counts = np.sum(X, axis=0)
rare_mask = symptom_counts >= 2  # keep only symptoms appearing in >=2 diseases
X = X[:, rare_mask]
filtered_symptoms = [s for s, keep in zip(all_symptoms, rare_mask) if keep]
print(f"✅ Filtered rare symptoms: {len(filtered_symptoms)} retained.")

# ==========================
# 4️⃣ SPLIT AND CROSS-VALIDATE
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ==========================
# 5️⃣ TRAIN RANDOM FOREST (REGULARIZED)
# ==========================
print("\n🌲 Training RandomForestClassifier (regularized)...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"Cross-validation accuracy: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\n🔹 RandomForest Results:")
print(f"Train accuracy: {rf.score(X_train, y_train):.3f}")
print(f"Test accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 (macro)    : {f1_score(y_test, y_pred, average='macro'):.3f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==========================
# 6️⃣ TRAIN NAIVE BAYES (BASELINE)
# ==========================
print("\n🧠 Training Naive Bayes (BernoulliNB)...")
nb = BernoulliNB()
nb_cv_scores = cross_val_score(nb, X, y, cv=cv, scoring="accuracy")
print(f"Cross-validation accuracy: {nb_cv_scores.mean():.3f} ± {nb_cv_scores.std():.3f}")

nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("\n🔹 Naive Bayes Results:")
print(f"Train accuracy: {nb.score(X_train, y_train):.3f}")
print(f"Test accuracy : {accuracy_score(y_test, y_pred_nb):.3f}")
print(f"F1 (macro)    : {f1_score(y_test, y_pred_nb, average='macro'):.3f}")

# ==========================
# 7️⃣ CHOOSE BEST MODEL
# ==========================
rf_acc = accuracy_score(y_test, y_pred)
nb_acc = accuracy_score(y_test, y_pred_nb)

best_model, model_name = (rf, "RandomForest") if rf_acc >= nb_acc else (nb, "NaiveBayes")
print(f"\n🏆 Selected Best Model: {model_name} (based on test accuracy)")

# ==========================
# 8️⃣ SAVE ARTIFACTS
# ==========================
joblib.dump(best_model, os.path.join(OUT_DIR, "disease_model.joblib"))
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))

with open(os.path.join(OUT_DIR, "symptom_list.json"), "w") as f:
    json.dump(filtered_symptoms, f, indent=2)

print(f"\n✅ Model, encoder, and symptom list saved to '{OUT_DIR}/'")

# ==========================
# 9️⃣ SUMMARY
# ==========================
print("\n📊 SUMMARY:")
print(f"Model Used : {model_name}")
print(f"CV Accuracy (RF): {rf_cv_scores.mean():.3f}")
print(f"CV Accuracy (NB): {nb_cv_scores.mean():.3f}")
print(f"Final Test Accuracy: {max(rf_acc, nb_acc):.3f}")
print("Artifacts saved to: /model/")
