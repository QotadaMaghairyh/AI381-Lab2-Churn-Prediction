
import numpy as np
from fastapi.testclient import TestClient

import src.app.main as app_module
from src import schema
from src.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_probability():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_fee": 75.0,
            "support_calls": 2,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
        },
    )

    assert response.status_code == 200
    probability = response.json()["churn_probability"]
    assert 0 <= probability <= 1


def test_predict_rejects_negative_tenure():
    response = client.post(
        "/predict",
        json={
            "tenure_months": -1,
            "monthly_fee": 75.0,
            "support_calls": 2,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
        },
    )

    assert response.status_code == 422


def test_predict_rejects_missing_field():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_fee": 75.0,
            "support_calls": 2,
            "contract": "Month-to-month",
        },
    )

    assert response.status_code == 422


def test_predict_is_deterministic():
    customer = {
        "tenure_months": 12,
        "monthly_fee": 75.0,
        "support_calls": 2,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
    }

    first_response = client.post("/predict", json=customer)
    second_response = client.post("/predict", json=customer)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert (
        first_response.json()["churn_probability"]
        == second_response.json()["churn_probability"]
    )



def test_model_loads():
    from src.app.main import MODEL_PATH, PROJECT_ROOT, model

    assert model is not None
    assert hasattr(model, "predict_proba")
    assert MODEL_PATH == PROJECT_ROOT / "artifacts" / "model.joblib"
    assert list(model.feature_names_in_) == schema.FEATURES


def test_predict_maps_monthly_fee_to_champion_schema(monkeypatch):
    class RecordingModel:
        def predict_proba(self, frame):
            assert list(frame.columns) == schema.FEATURES
            assert "monthly_fee" not in frame.columns
            assert frame.loc[0, "monthly_charges"] == 75.0
            assert frame.loc[0, "tenure_months"] == 12
            assert frame.loc[0, "support_calls"] == 2
            assert frame.loc[0, "contract"] == "Month-to-month"
            assert frame.loc[0, "internet_service"] == "Fiber optic"
            return np.array([[0.25, 0.75]])

    monkeypatch.setattr(app_module, "model", RecordingModel())
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_fee": 75.0,
            "support_calls": 2,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"churn_probability": 0.75}
