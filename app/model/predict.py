import pickle
from pathlib import Path

import pandas as pd

# import ml model (path is relative to this file, so it works from any working directory)
MODEL_PATH = Path(__file__).resolve().parent / 'model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
    
MODEL_VERSION = '1.0.0' # usually we get this from MLFlow
class_labels = model.classes_.tolist()

def predict_output(user_input : dict):
    df = pd.DataFrame([user_input])
    
    predicted_class = str(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0]
    confidence = float(max(probabilities))
    
    class_probs = {label: round(float(p), 4) for label, p in zip(class_labels, probabilities)}
    
    return {'predicted_category' : predicted_class,
            'confidence' : round(confidence, 4),
            'class_probabilities' : class_probs}