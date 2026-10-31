# AI381 Lab 2 – Churn Prediction API

[![CI](https://github.com/QotadaMaghairyh/AI381-Lab2-Churn-Prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/QotadaMaghairyh/AI381-Lab2-Churn-Prediction/actions/workflows/ci.yml)

A machine learning API that predicts the probability of customer churn using a trained Scikit-learn model.

## Features

- FastAPI service with `/health` and `/predict` endpoints.
- Input validation using Pydantic.
- Automated tests using Pytest.
- Code linting using Ruff.
- Multi-stage Docker image running as a non-root user.
- GitHub Actions CI workflow with Docker image scanning using Trivy.
- Docker images published to GitHub Container Registry (GHCR).

## Run with Docker

Start the API with one command:

```bash
docker run --rm --name ai381-churn-api -p 8000:8000 ghcr.io/qotadamaghairyh/ai381-lab2-churn-prediction:cc668be8e6eca9cec40b0aff38b9f9370d8f83aa
```

The API will be available at:

- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Predict Customer Churn

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_fee": 70.0,
    "support_calls": 2,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic"
  }'
```

The API returns a churn probability between 0 and 1.

## Run Tests

Install the project dependencies and development tools, then run:

```bash
python -m pytest -v
ruff check src tests
```

## Project Structure

```text
.
├── src/
│   ├── app/
│   │   └── main.py
│   └── train/
│       └── train.py
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
├── model.joblib
└── churn_v2.csv
```

## Continuous Integration

GitHub Actions automatically runs the test and lint checks for pull requests targeting `main`.

When changes are pushed to `main`, the workflow:

1. Runs tests and linting.
2. Builds the Docker image.
3. Scans the image for vulnerabilities using Trivy.
4. Publishes the image to GitHub Container Registry (GHCR), tagged with the commit SHA.

The `main` branch is protected and requires the `test` status check to pass before merging.

## Technologies

- Python
- FastAPI
- Scikit-learn
- Pydantic
- Pytest
- Ruff
- Docker
- GitHub Actions
- Trivy
- GitHub Container Registry
