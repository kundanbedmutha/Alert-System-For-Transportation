import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Load model ----------
model, features = joblib.load("app/models/alert_model.joblib")
print("✅ Model loaded successfully.")

# ---------- Load dataset ----------
data_path = "data/Synthetic_Transportation_Dataset_Expanded_v2.csv"
df = pd.read_csv(data_path)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print(f"✅ Dataset loaded with {len(df)} records.")

# ---------- Predict anomalies ----------
df["prediction"] = model.predict(df[features])
print("\n🔹 Prediction summary:")
print(df["prediction"].value_counts())

# ---------- Optional: Evaluate if label exists ----------
label_cols = [c for c in df.columns if c in ["label", "status", "alert", "target"]]
if label_cols:
    label_col = label_cols[0]
    print(f"\n🧠 Found label column: '{label_col}' — evaluating performance...")

    # Prepare y_true and y_pred
    y_true = df[label_col].astype(str).str.lower().replace({
        "normal": 1, "ok": 1, "safe": 1,
        "alert": -1, "anomaly": -1
    })
    y_pred = df["prediction"]

    # Classification metrics
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 Classification Report:")
    print(report)
    print("🧩 Confusion Matrix:")
    print(cm)

    # Save results to file
    with open("evaluation_report.txt", "w") as f:
        f.write("=== MODEL EVALUATION REPORT ===\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Label column: {label_col}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print("\n📁 Evaluation report saved to: evaluation_report.txt")

else:
    print("\n⚠️ No label column found — skipping evaluation metrics.")
    print("You can still inspect predictions using the 'prediction' column.")

# ---------- Optional: Save predictions ----------
df.to_csv("predicted_output.csv", index=False)
print("📁 Predictions saved to: predicted_output.csv")
