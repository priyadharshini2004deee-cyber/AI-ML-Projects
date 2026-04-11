import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "bank_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

# Best threshold from notebook for saved XGBoost model
BEST_THRESHOLD = 0.36

app = FastAPI(
    title="Bank Term Deposit Prediction API",
    description="FastAPI backend for bank term deposit subscription prediction",
    version="1.0.0"
)

# -----------------------------
# Load model artifacts once
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    model = None
    preprocessor = None
    LOAD_ERROR = str(e)
else:
    LOAD_ERROR = None


class CustomerInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


def build_features(data: CustomerInput) -> pd.DataFrame:
    df = pd.DataFrame([data.model_dump()])

    # Must exactly match notebook feature engineering
    df["balance_age"] = df["balance"] * df["age"]
    df["campaign_balance"] = df["campaign"] * df["balance"]

    # duration not used in training
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    return df


@app.get("/")
def home():
    if LOAD_ERROR:
        return {"status": "error", "message": LOAD_ERROR}
    return {"status": "ok", "message": "Bank API running successfully"}


@app.get("/health")
def health():
    return {
        "api_status": "running",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "error": LOAD_ERROR,
        "threshold": BEST_THRESHOLD
    }


@app.post("/predict")
def predict(data: CustomerInput):
    if LOAD_ERROR or model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail=f"Artifacts not loaded: {LOAD_ERROR}")

    try:
        input_df = build_features(data)
        transformed = preprocessor.transform(input_df)

        prob = float(model.predict_proba(transformed)[0][1])
        pred = "Subscribe" if prob >= BEST_THRESHOLD else "Will Not Subscribe"

        if prob >= 0.60:
            segment = "High Potential"
        elif prob >= 0.30:
            segment = "Medium Potential"
        else:
            segment = "Low Potential"

        return {
            "probability": round(prob, 4),
            "probability_percent": round(prob * 100, 2),
            "prediction": pred,
            "lead_segment": segment,
            "threshold_used": BEST_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
