#!/usr/bin/env bash
# Run a kit script inside WSL with the Idol environment. The Hugging Face token comes from $HF_TOKEN or from
# .env in this folder (copy .env.example to .env; .env is git-ignored and never committed).
#   bash run_wsl.sh bench.py --only 01-robot-toy
#   bash run_wsl.sh idol_test.py
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
. "$HOME/.venvs/idol/bin/activate"
ENV_FILE="$(pwd)/.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
  HF_TOKEN=$(grep -E '^[[:space:]]*HF_TOKEN[[:space:]]*=' "$ENV_FILE" | grep -oE 'hf_[A-Za-z0-9]+' | head -1 || true)
  export HF_TOKEN
fi
export PYTHONUNBUFFERED=1
exec python -X faulthandler "$@"
