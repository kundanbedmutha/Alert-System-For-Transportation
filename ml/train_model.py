# =============================================================
# 🚗 Smart Transport System - ML Model Training (Dual Models)
# =============================================================

# 1️⃣ IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# 2️⃣ LOAD DATASET
file_path = "Synthetic_Transportation_Dataset_Expanded_v2.csv"
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# 3️⃣ BASIC CLEANING
df.fillna(0, inplace=True)

# Convert categorical columns to numeric
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Ensure target columns exist
if "Accident_Occurred" not in df.columns:
    raise ValueError("❌ 'Accident_Occurred' column not found in dataset.")
if "Accident_Severity" not in df.columns:
    raise ValueError("❌ 'Accident_Severity' column not found in dataset.")

# 4️⃣ DEFINE FEATURE SETS
# --- Model 1: Accident Occurrence ---
X1 = df.drop(columns=["Accident_Occurred", "Accident_Severity"])
y1 = df["Accident_Occurred"].astype(int)

# --- Model 2: Accident Severity ---
# Train only on rows where an accident occurred
df_severity = df[df["Accident_Occurred"] == 1].copy()
X2 = df_severity.drop(columns=["Accident_Occurred", "Accident_Severity"])
y2 = df_severity["Accident_Severity"].astype(int)

print("\n📊 Accident Occurrence target distribution:\n", y1.value_counts())
print("\n📊 Accident Severity target distribution:\n", y2.value_counts())

# 5️⃣ TRAIN/TEST SPLITS
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# 6️⃣ TRAIN RANDOM FOREST MODELS
occurrence_model = RandomForestClassifier(n_estimators=200, random_state=42)
severity_model = RandomForestClassifier(n_estimators=200, random_state=42)

print("\n🧠 Training Accident Occurrence Model...")
occurrence_model.fit(X1_train, y1_train)
print("✅ Accident Occurrence Model trained!")

print("\n🧠 Training Accident Severity Model...")
severity_model.fit(X2_train, y2_train)
print("✅ Accident Severity Model trained!")

# 7️⃣ EVALUATION
print("\n📈 Accident Occurrence Model Evaluation")
y1_pred = occurrence_model.predict(X1_test)
print(classification_report(y1_test, y1_pred))
cm1 = confusion_matrix(y1_test, y1_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Accident Occurrence")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n📈 Accident Severity Model Evaluation")
y2_pred = severity_model.predict(X2_test)
print(classification_report(y2_test, y2_pred))
cm2 = confusion_matrix(y2_test, y2_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix – Accident Severity")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8️⃣ SAVE TRAINED MODELS
os.makedirs("models", exist_ok=True)
joblib.dump(occurrence_model, "models/accident_model.joblib")
joblib.dump(occurrence_model, "models/alert_model.joblib")
joblib.dump(severity_model, "models/severity_model.joblib")
print("\n✅ Models saved successfully to /models folder!")

# 9️⃣ FEATURE IMPORTANCE VISUALIZATION
feat_imp = pd.Series(occurrence_model.feature_importances_, index=X1.columns)
top_features = feat_imp.nlargest(10)
plt.figure(figsize=(8,5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features – Accident Occurrence Model")
plt.xlabel("Feature Importance Score")
plt.show()

print("🏁 Training Complete!")
