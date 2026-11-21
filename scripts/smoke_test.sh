#!/usr/bin/env bash
# Based on the official Lab 3 Starter; run only after Compose deployment is approved.
#   bash scripts/smoke_test.sh [http://127.0.0.1:8001]
# Requires bash, curl, python3 and Docker Compose. Creates no containers.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8001}"
BASE_URL="${BASE_URL%/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ghcr.io/qotadamaghairyh/ai381-lab2-churn-prediction:d00ba9d154b4efb22409891e42cb178ff1930358@sha256:07e9ebbdc028f006fb4f363b7bca1f5979d4fde93e757fcd68802f809b1dfd3c"
HEALTH_FILE="$(mktemp "${TMPDIR:-/tmp}/ai381-smoke-health.XXXXXX")"
trap 'rm -f -- "$HEALTH_FILE"' EXIT

echo "Waiting for $BASE_URL/health (30 attempts)..."
for i in $(seq 1 30); do
  if curl -fsS --connect-timeout 2 --max-time 3 "$BASE_URL/health" -o "$HEALTH_FILE" 2>/dev/null; then
    break
  fi
  if [ "$i" -eq 30 ]; then echo "FAIL: /health did not answer"; exit 1; fi
  sleep 2
done
echo "health: $(cat "$HEALTH_FILE")"

# Preserve the Lab 2 external schema; use canonical training values for categories.
REQUEST='{"tenure_months":5,"monthly_fee":89.9,"contract":"month-to-month","internet_service":"fiber","support_calls":3}'
PRED="$(curl -fsS --connect-timeout 2 --max-time 10 -X POST "$BASE_URL/predict" \
  -H 'Content-Type: application/json' -d "$REQUEST")"
echo "predict: $PRED"
python3 -B - "$PRED" "$(cat "$HEALTH_FILE")" <<'PY'
import json
import math
import sys

pred, health = json.loads(sys.argv[1]), json.loads(sys.argv[2])
p = pred["churn_probability"]
assert not isinstance(p, bool) and isinstance(p, (int, float)), "prediction is not numeric"
assert math.isfinite(p) and 0.0 <= p <= 1.0, f"probability out of range: {p}"
assert health.get("status") == "ok", "health is not ok"
print("PASS: /health and /predict")
PY

# Verify the existing Compose container; never start, replace, or remove one here.
CONTAINER="$(docker compose -f "$HERE/docker-compose.deploy.yml" ps -q api)"
[[ -n "$CONTAINER" && "$CONTAINER" != *$'\n'* ]] || { echo "FAIL: expected one running Compose API container"; exit 1; }
EXPECTED_IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
ACTUAL_IMAGE_ID="$(docker inspect "$CONTAINER" --format '{{.Image}}')"
[[ "$ACTUAL_IMAGE_ID" == "$EXPECTED_IMAGE_ID" ]] || { echo "FAIL: container image does not match the pinned image"; exit 1; }

docker compose -f "$HERE/docker-compose.deploy.yml" exec -T api python -B - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("/app/artifacts")
metadata = json.loads((root / "metadata.json").read_text())
champion = json.loads((root / "champion.json").read_text())
git_sha = "d00ba9d154b4efb22409891e42cb178ff1930358"
assert metadata == champion, "provenance files disagree"
assert metadata["mlflow_run_id"] == "86cf70eed0be4c06908008fde42432b3", "unexpected MLflow run"
assert metadata["model"] == "logreg" and metadata["mlflow_model_version"] == 1, "unexpected model/version"
assert metadata["model_version"] == git_sha[:7], "Git SHA prefix does not match the pinned image tag"
model_sha = hashlib.sha256((root / "model.joblib").read_bytes()).hexdigest()
assert model_sha == (root / "model.sha256").read_text().split()[0], "model checksum mismatch"
# Obtained from the champion artifact of workflow run 44896074255, attempt 1.
assert model_sha == "0a3eef066827368867168ca831893c183646058f22e3433a0ebe91a09f1dcfee", "model differs from the approved workflow artifact"
print("PASS: champion run ID =", metadata["mlflow_run_id"])
print("PASS: Git SHA from pinned image tag =", git_sha)
print("PASS: model SHA-256 =", model_sha)
PY

echo "PASS: local deployment smoke test and champion provenance"
