
from fastapi.testclient import TestClient

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
    from src.app.main import model

    assert model is not None
    assert hasattr(model, "predict_proba")
