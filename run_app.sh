#!/usr/bin/env bash
# Check/install dependencies and launch the local Streamlit explorer for ShanghaiTech & UBnormal.
# Usage:
#   ./run_app.sh [--check-only] [--port PORT]

set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${ROOT_DIR}/experiments"
APP_FILE="${EXP_DIR}/src/edgelab/app/streamlit_app.py"
PORT="${PORT:-8501}"
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

echo "==> [1/3] Checking Python environment and dependencies in ${EXP_DIR}..."
cd "${EXP_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is required but was not found on PATH." >&2
  exit 1
fi

# Check if streamlit and core packages are importable in the existing venv; install if missing
if ! .venv/bin/python -c "import streamlit, ultralytics, cv2, onnxruntime, edgelab" >/dev/null 2>&1; then
  echo "==> Installing/syncing 'streamlit' into experiments/.venv..."
  uv pip install --python .venv/bin/python "streamlit>=1.40" -e .
fi

echo "==> [2/3] Verifying dataset paths & model catalog..."
.venv/bin/python -c "
from edgelab.app.catalog import build_model_catalog, list_dataset_videos, DATA_ROOT
models = build_model_catalog()
sht = list_dataset_videos('ShanghaiTech')
ub = list_dataset_videos('UBnormal')
avail_weights = sum(1 for m in models if m.weights_path and m.weights_path.exists())
print(f'    Data root       : {DATA_ROOT}')
print(f'    Catalog models  : {len(models)} total ({avail_weights} local weight files + precomputed caches)')
print(f'    ShanghaiTech    : {len(sht)} test clips ready')
print(f'    UBnormal        : {len(ub)} test clips ready')
"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "==> Check complete (--check-only)."
  exit 0
fi

echo "==> [3/3] Starting Streamlit server on http://localhost:${PORT} ..."
exec .venv/bin/streamlit run "${APP_FILE}" \
  --server.port "${PORT}" \
  --server.headless false \
  "$@"
