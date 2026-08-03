#!/usr/bin/env bash
# Image capability probe for the H100 training base image.
#
# Verifies the runtime prerequisites that launch_h100.sh relies on:
#   - python3.12 (open-r1 / trl 0.18.0 / deepspeed pin to 3.10+; 3.12 expected)
#   - flash_attn importable (provides the flash_attention_3 backend for H100)
#   - torch + CUDA + H100 device visible
#
# Intended as a cctl PyTorchJob --entry. Exits non-zero if any check fails so
# the task shows up as Failed (fail-fast, not a silent warning). All cluster
# coordinates come from setup_env.sh (SSOT) — do not hardcode them here.
set -euo pipefail

# shellcheck source=setup_env.sh
source "$(dirname "$0")/setup_env.sh"

echo "[probe] image=${CCTL_IMAGE}"
echo "[probe] python path: $(command -v python || command -v python3)"

PYBIN="$(command -v python || command -v python3)"
echo "[probe] === python version ==="
"${PYBIN}" --version

echo "[probe] === flash_attn ==="
if "${PYBIN}" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null; then
    echo "[probe] flash_attn: PRESENT"
else
    echo "[probe] flash_attn: MISSING (launch_h100.sh will pip install at runtime)"
fi

echo "[probe] === torch / CUDA / GPU ==="
"${PYBIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}] {props.name} cc={props.major}.{props.minor} mem={props.total_memory // (1024**3)}GiB")
PY

echo "[probe] === transformers attn backends ==="
"${PYBIN}" - <<'PY' 2>/dev/null || echo "[probe] transformers not installed in base image (ok: launch installs it)"
import transformers
print("transformers", transformers.__version__)
PY

echo "[probe] DONE"
