#!/usr/bin/env bash
# SplatGen trainer - one-time setup on Linux / macOS.
# On Linux with an NVIDIA GPU set TORCH_INDEX to a CUDA build, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./install.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
if [ -n "${TORCH_INDEX:-}" ]; then
  python -m pip install torch --index-url "$TORCH_INDEX"
fi
python -m pip install -e .
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  python -m pip install gsplat || echo "gsplat could not be installed; the PyTorch renderer will be used."
fi
python -m splatgen info
echo "Done. Start SplatGen with ./splatgen.sh"
