# ml/preprocess.py
import pandas as pd
import numpy as np

def load_and_select_features(csv_path, label_col=None):
    df = pd.read_csv(csv_path)
    # lowercase cols
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # drop obviously ID-like non-numeric columns if necessary
    # pick numeric columns as features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) == 0:
        # try converting some columns
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except:
                pass
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    X = df[numeric_cols].fillna(0)
    y = None
    if label_col and label_col in df.columns:
        y = df[label_col]
    return X, y, df
