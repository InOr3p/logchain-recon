import os
import joblib
import pandas as pd

# Load LightGBM model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "AI_models", "LightGBM_fd_all_cat.joblib")
clf = joblib.load(MODEL_PATH)

def classify_logs(logs):
    """Classify logs as benign or attack using LightGBM."""
    df = pd.DataFrame([log.dict() for log in logs])
    # Assuming the model expects a specific feature set
    features = df[["feature1", "feature2", "feature3"]]  # adapt to your data
    preds = clf.predict(features)
    labels = ["attack" if p == 1 else "benign" for p in preds]
    for i, log in enumerate(logs):
        log.label = labels[i]
    return logs
