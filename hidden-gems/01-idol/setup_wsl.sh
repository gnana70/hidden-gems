#!/usr/bin/env bash
# One-time setup of the Idol test kit inside WSL2 (Ubuntu). Run-Idol-Tests.ps1 calls this for you.
#   cd hidden-gems/01-idol && bash setup_wsl.sh   (from the repository root)
set -euo pipefail
cd "$(dirname "$0")"
VENV="$HOME/.venvs/idol"     # on the Linux disk: installing PyTorch onto /mnt/d would be very slow

echo "== GPU inside WSL"
if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  echo "nvidia-smi doesn't work inside WSL. Update the NVIDIA driver on Windows, run 'wsl --shutdown' and retry."
  exit 1
fi

if ! command -v gcc >/dev/null 2>&1; then
  echo "== Installing a C compiler (torch.compile's Triton backend needs gcc). Ubuntu asks for your password."
  sudo apt-get update -y && sudo apt-get install -y build-essential git
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "== Installing uv (fast Python package installer)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d idol-sdk ]; then
  echo "== Cloning the Idol SDK"
  git clone --depth 1 https://github.com/speridlabs/idol-sdk.git idol-sdk
fi
git -C idol-sdk rev-parse HEAD > idol-sdk-commit.txt || true

# Slow or flaky connections: long timeouts, fewer parallel downloads, and retries (the CUDA libraries are big files).
export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=3 UV_HTTP_RETRIES=8
retry() { for i in 1 2 3 4; do "$@" && return 0; echo "   (attempt $i failed, retrying in 20 s)"; sleep 20; done; return 1; }

echo "== Creating $VENV (Python 3.12)"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
rm -f "$VENV/.idol-setup-ok"

# PyTorch with CUDA from PyPI. We tested with torch 2.15.0 (CUDA 13.2 build); override with e.g. TORCH_SPEC="torch==2.15.0".
TORCH_SPEC="${TORCH_SPEC:-torch>=2.5}"
echo "== Installing PyTorch with CUDA ($TORCH_SPEC; about 3 GB the first time, takes a while)"
retry uv pip install "$TORCH_SPEC" torchvision || { echo "PyTorch install failed (network). Run the same command again: finished downloads are cached."; exit 1; }
echo "== Installing the Idol SDK and the test tools"
retry uv pip install -e ./idol-sdk || exit 1
retry uv pip install nvidia-ml-py numba plyfile || exit 1

python - <<'PYEOF'
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'
import idol, huggingface_hub, numba, pynvml; print('idol SDK import ok')
PYEOF
touch "$VENV/.idol-setup-ok"
echo "Setup done."
