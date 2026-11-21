import warnings
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sklearn.exceptions import InconsistentVersionWarning

from src import schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"

# Load only the trusted local artifact exported by our promotion gate.
with warnings.catch_warnings():
    warnings.simplefilter("error", InconsistentVersionWarning)
    model = joblib.load(MODEL_PATH)

if list(model.feature_names_in_) != schema.FEATURES:
    raise RuntimeError("Champion features do not match the serving schema")

app = FastAPI(title="AI381 Churn Prediction API")


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0)
    monthly_fee: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract: str = Field(min_length=1)
    internet_service: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
    customer_data = customer.model_dump()
    X = pd.DataFrame([customer_data]).rename(columns={"monthly_fee": "monthly_charges"})
    X = schema.validate(X, need_target=False)[schema.FEATURES]

    probability = float(model.predict_proba(X)[0, 1])

    return {
        "churn_probability": round(probability, 3)
    }
