#!/usr/bin/env bash
# Everything for the video, in order: setup (first time only), downloads, the full test, the torch.compile comparison.
# Run-Idol-Tests.ps1 runs this inside WSL.
set -uo pipefail
cd "$(dirname "$0")"
[ -f "$HOME/.venvs/idol/.idol-setup-ok" ] || bash setup_wsl.sh || exit 1
export PATH="$HOME/.local/bin:$PATH"
# accelerate: loads the encoders with less CPU memory (optional, small)
"$HOME/.venvs/idol/bin/python" -c "import accelerate" 2>/dev/null || VIRTUAL_ENV="$HOME/.venvs/idol" uv pip install accelerate || true

bash run_wsl.sh download_weights.py || exit 1

STAMP=$(date +%Y%m%d-%H%M%S)
OUT="results/$STAMP"
mkdir -p "$OUT" "$OUT-compiled"
{ echo "== memory before the run"; free -m; nproc; } > "$OUT/system.txt" 2>&1
echo "== 1/2  Full test WITHOUT torch.compile (the safe path): 9 images + settings sweep + seed check  -> $OUT"
bash run_wsl.sh bench.py --no-optimize --out "$OUT" 2>&1 | tee "$OUT/console.log"

echo "== 2/2  torch.compile comparison on two images (one compile thread, to keep memory low)  -> $OUT-compiled"
TORCHINDUCTOR_COMPILE_THREADS=1 bash run_wsl.sh bench.py --no-sweep --only 01-robot-toy 02-red-panda --tag compiled --out "$OUT-compiled" 2>&1 | tee "$OUT-compiled/console.log"
echo "exit code of the compiled run: ${PIPESTATUS[0]}" | tee -a "$OUT-compiled/console.log"

nvidia-smi > "$OUT/nvidia-smi-after.txt" 2>&1 || true
echo
echo "All done. Results are in $OUT (and $OUT-compiled)."
