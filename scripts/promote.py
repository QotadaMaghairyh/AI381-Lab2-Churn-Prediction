"""Lab 3: the promotion gate. Only a better model becomes the champion.

    python scripts/promote.py                      # compares this batch's best candidate with the champion
    python scripts/promote.py --batch abc1234      # a specific batch (default: the BATCH or GIT_SHA environment variable)

How it decides (thresholds are documented in MODEL_SELECTION.md):
  1. Pick the best FINISHED candidate of the exact batch by F1 with valid metrics, a model URI,
     and latency within the p95 budget.
  2. Compare it with the current champion, whose numbers live in artifacts/champion.json (a CI runner forgets
     everything between runs, so the champion's numbers are kept in Git). In CI the workflow passes the champion
     from the base branch with --champion-file, so a pull request is judged against what is on main today.
  3. Better by at least MIN_GAIN F1 points -> promote: register the model in MLflow, set the alias "champion",
     export it to artifacts/ (this is what the Dockerfile packages) and write the new artifacts/champion.json.
     Not better -> exit code 1. In CI that is a red run: the gate blocked the model.
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

REGISTERED_NAME = "churn-model"
ARTIFACTS = Path("artifacts")
CHAMPION_FILE = ARTIFACTS / "champion.json"
LATENCY_BUDGET_MS = 25.0  # Provisional budget documented in MODEL_SELECTION.md.
MIN_GAIN = 0.005  # Absolute F1 gain; provisional, not statistically validated.

def eligible_run(run, batch: str) -> bool:
    """Require a finished run from this exact batch with usable metrics and metadata."""
    if run.info.status != "FINISHED" or run.data.tags.get("batch") != batch:
        return False
    if not isinstance(run.info.run_id, str) or not run.info.run_id.strip():
        return False
    metrics = run.data.metrics
    for key in ("f1", "auc", "p95_latency_ms"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
    if not (0 <= metrics["f1"] <= 1 and 0 <= metrics["auc"] <= 1):
        return False
    if metrics["p95_latency_ms"] < 0:
        return False
    for key in ("model", "data_sha256"):
        value = run.data.params.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    uri = run.data.tags.get("model_uri")
    # Native log_model() URIs in MLflow 3, and run artifact URIs in older runs.
    if not isinstance(uri, str) or not re.fullmatch(
        r"models:/m-[A-Za-z0-9_-]+|runs:/[0-9a-fA-F]{32}/[^\s?#]+", uri
    ):
        return False
    return True


def best_candidate(batch: str) -> dict:
    if not isinstance(batch, str) or not batch.strip():
        sys.exit("BLOCKED: a nonempty batch is required.")
    # Quote the filter without changing the batch value; reject unrepresentable values.
    quote = "'" if "'" not in batch else '"'
    if quote in batch:
        sys.exit("BLOCKED: batch cannot contain both single and double quotes.")
    runs = mlflow.search_runs(
        experiment_names=["churn"],
        filter_string=f"tags.batch = {quote}{batch}{quote} and attributes.status = 'FINISHED'",
        output_format="list",
    )
    runs = [r for r in runs if eligible_run(r, batch)]
    fast = [r for r in runs if r.data.metrics["p95_latency_ms"] <= LATENCY_BUDGET_MS]
    if not runs:
        sys.exit(
            f"BLOCKED: no FINISHED runs with valid metrics and a model URI "
            f"found for batch '{batch}'. Check the batch tag and tracking server."
        )
    if not fast:
        sys.exit(f"BLOCKED: no candidate is within the {LATENCY_BUDGET_MS} ms latency budget.")
    run = max(fast, key=lambda r: r.data.metrics["f1"])
    return {
        "run_id": run.info.run_id,
        "model": run.data.params["model"],
        "f1": run.data.metrics["f1"],
        "auc": run.data.metrics["auc"],
        "p95_latency_ms": run.data.metrics["p95_latency_ms"],
        "data_sha256": run.data.params["data_sha256"],
        "model_uri": run.data.tags["model_uri"],
    }


def current_champion(path: Path, *, required: bool = False) -> dict | None:
    try:
        champion = json.loads(path.read_text())
    except FileNotFoundError:
        if not required:
            return None
        sys.exit(f"BLOCKED: explicitly specified champion file does not exist: {path}")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        sys.exit(f"BLOCKED: cannot read champion file {path}: {exc}")

    if not isinstance(champion, dict):
        sys.exit(f"BLOCKED: champion file must contain an object: {path}")
    if not isinstance(champion.get("model"), str) or not champion["model"].strip():
        sys.exit(f"BLOCKED: champion file has no valid model name: {path}")
    for key in ("f1", "p95_latency_ms"):
        value = champion.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            sys.exit(f"BLOCKED: champion file has invalid {key}: {path}")
    if not 0 <= champion["f1"] <= 1 or champion["p95_latency_ms"] < 0:
        sys.exit(f"BLOCKED: champion file has out-of-range metrics: {path}")
    return champion


def promote(candidate: dict) -> None:
    client = MlflowClient()
    version = mlflow.register_model(candidate["model_uri"], REGISTERED_NAME)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")
    meta = {
        "model_version": os.getenv("GIT_SHA", "local")[:7],  # the Dockerfile overwrites this with the image's commit
        "mlflow_run_id": candidate["run_id"],
        "mlflow_model_version": int(version.version),
        "model": candidate["model"],
        "data_sha256": candidate["data_sha256"],
        "auc": round(candidate["auc"], 3),
        "f1": candidate["f1"],
        "p95_latency_ms": round(candidate["p95_latency_ms"], 2),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(meta, indent=2))
    CHAMPION_FILE.write_text(json.dumps(meta, indent=2))
    print(f"PROMOTED: {candidate['model']} is the new champion (MLflow version {version.version}).")


def has_minimum_f1_gain(candidate_f1: float, champion_f1: float) -> bool:
    """Compare unrounded F1 values, including the exact promotion boundary.

    >>> has_minimum_f1_gain(0.4998, 0.4948)
    True
    >>> has_minimum_f1_gain(0.4997, 0.4948)
    False
    >>> has_minimum_f1_gain(0.4999, 0.4948)
    True
    """
    return candidate_f1 >= champion_f1 + MIN_GAIN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=os.getenv("BATCH", os.getenv("GIT_SHA", "local")[:7]))
    ap.add_argument(
        "--champion-file", type=Path,
        help="the champion to beat (explicit paths must exist; default: artifacts/champion.json)",
    )
    a = ap.parse_args()

    champion = current_champion(
        a.champion_file if a.champion_file is not None else CHAMPION_FILE,
        required=a.champion_file is not None,
    )
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    candidate = best_candidate(a.batch)
    print(f"best candidate: {candidate['model']}  f1 {candidate['f1']:.3f}  p95 {candidate['p95_latency_ms']:.1f} ms")

    if champion is None:
        print("No champion yet: the first model inside the latency budget is promoted.")
        promote(candidate)
        return 0
    print(f"champion:       {champion['model']}  f1 {champion['f1']:.3f}  p95 {champion['p95_latency_ms']:.1f} ms")
    if has_minimum_f1_gain(candidate["f1"], champion["f1"]):
        promote(candidate)
        return 0
    print(f"BLOCKED: F1 {candidate['f1']:.3f} is not at least {MIN_GAIN} better than the champion's {champion['f1']:.3f}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
