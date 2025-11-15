import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib, os

df = pd.read_csv("data/Enhanced_Transportation_Dataset.csv")
df = df.fillna(0)
X = df[["speed", "temperature", "humidity", "rain", "visibility"]]
y = df["accident"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Accident model trained. Accuracy:", model.score(X_test, y_test))
joblib.dump(model, "app/models/accident_model.joblib")
