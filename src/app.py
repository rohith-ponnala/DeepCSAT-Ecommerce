from fastapi import FastAPI
from pydantic import BaseModel, RootModel
from typing import Any, Dict, List
import joblib, pandas as pd, os

app = FastAPI(title="DeepCSAT Customer Satisfaction API")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/csat_pipeline.pkl")
pipe = None


@app.on_event("startup")
def load_model():
    global pipe
    if os.path.exists(MODEL_PATH):
        pipe = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")


# ✅ New Pydantic v2 style for root model
class Record(RootModel[List[Dict[str, Any]]]):
    pass


@app.post("/predict")
def predict(records: Record):
    global pipe
    if pipe is None:
        return {"error": "Model not loaded. Please place csat_pipeline.pkl in the models/ folder."}

    # Convert input data to DataFrame
    df = pd.DataFrame(records.root)

    # Make predictions
    preds = pipe.predict(df).tolist()

    try:
        proba = pipe.predict_proba(df)[:, 1].tolist()
    except Exception:
        proba = [None] * len(preds)

    return {"pred_label": preds, "pred_prob_satisfied": proba}
