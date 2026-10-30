
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "churn_v2.csv"
MODEL_PATH = PROJECT_ROOT / "model.joblib"

# Features used by the model
NUMERIC = ["tenure_months", "monthly_fee", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def main():
    # 1. Load the dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # 2. Separate features and target
    X = df[FEATURES]
    y = df[TARGET]

    # 3. Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Build the preprocessing and classification pipeline
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            CATEGORICAL,
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Evaluate on the test set
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities > 0.5

    print(f"F1 : {f1_score(y_test, predictions):.3f}")
    print(f"AUC: {roc_auc_score(y_test, probabilities):.3f}")

    # 7. Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
