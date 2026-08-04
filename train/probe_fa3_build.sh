#!/usr/bin/env bash
# Probe: confirm flash-attention source tree + nvcc + torch ABI before
# attempting to compile the FA3 Hopper kernel.
set +e
unset PIP_CONSTRAINT

SRC=/user/dengxianglong/projects/flash-attention

echo "=== source git ref ==="
git -C "$SRC" rev-parse --abbrev-ref HEAD 2>&1
git -C "$SRC" describe --tags 2>&1

echo "=== hopper dir ==="
ls -la "$SRC/hopper/" 2>&1 | head -20

echo "=== submodule status (CUTLASS etc.) ==="
git -C "$SRC" submodule status 2>&1 | head -10

echo "=== nvcc ==="
which nvcc 2>&1
nvcc --version 2>&1 | tail -3

echo "=== torch + cuda ==="
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('abi', torch._C._GLIBCXX_USE_CXX11_ABI); import sys; print('py', sys.version)" 2>&1

echo "=== ninja ==="
which ninja 2>&1; ninja --version 2>&1

echo "=== hopper setup.py exists? ==="
ls -la "$SRC/hopper/setup.py" 2>&1

echo "=== DONE ==="
