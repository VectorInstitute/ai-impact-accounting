#!/usr/bin/env bash
# Local DIA dashboard: ingest the demo model into state, then launch the UI.
#
# Requires:  pip install "ai-impact-accounting[dashboard]"
# Usage:     ./scripts/run_local.sh [model-id]
#
# Override demo defaults with your own Hugging Face namespace:
#   export DIA_DATASET=your-username/dia-state
#   export DIA_BASES=distilbert-base-uncased
#   ./scripts/run_local.sh your-username/my-bert-sentiment
set -euo pipefail
cd "$(dirname "$0")"

export DIA_DATASET="${DIA_DATASET:-DIA-MVP/dia-state-lab-2026}"
export DIA_BASES="${DIA_BASES:-distilbert-base-uncased}"
export WEBHOOK_SECRET="${WEBHOOK_SECRET:-local-dev}"
export DIA_INGEST_MODEL="${1:-DIA-MVP/my-bert-sentiment}"

if [ -z "${HF_TOKEN:-}" ] && ! python -c "from huggingface_hub import get_token; \
import sys; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
  echo "Not logged in. Run: hf auth login   (write token)"
  exit 1
fi

echo ">> Ingesting ${DIA_INGEST_MODEL} into ${DIA_DATASET} ..."
python -c "import os; from ai_impact_accounting import Store, ingest_model; \
s=Store(os.environ['DIA_DATASET']); print(ingest_model(os.environ['DIA_INGEST_MODEL'], s))"

echo ">> Launching dashboard (Ctrl-C to stop) ..."
# view_local.py launches the FastAPI web UI (no webhook server) for reliable
# local viewing. The deployed Space uses ai_impact_accounting.dashboard.app.
python view_local.py
