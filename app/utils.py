import os
import joblib

def load_model():
    alert_model = joblib.load('models/alert_model.joblib')
    accident_model = joblib.load('models/accident_model.joblib')
    feature_list = []  # later you can fill this with real feature names
    return alert_model, accident_model, feature_list

