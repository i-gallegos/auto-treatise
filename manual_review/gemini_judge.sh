#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
: "${GOOGLE_CLOUD_LOCATION:?set GOOGLE_CLOUD_LOCATION}"
: "${GCS_BATCH_BUCKET:?set GCS_BATCH_BUCKET}"
exec python -m manual_review.model_label \
  --model gemini-2.5-pro \
  --overlay gpt-5.6-sol_tools \
  "$@"
