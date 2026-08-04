#!/usr/bin/env bash
# Compile the FlashAttention-3 Hopper kernel (flash_attn_3_cuda*.so) from the
# pre-downloaded flash-attention source on GPFS, then collect the .so to GPFS.
# Run via cctl --code-type git mount, 1 GPU + 64 CPU node.
set -euo pipefail
unset PIP_CONSTRAINT

SRC=/user/dengxianglong/projects/flash-attention
OUT=/user/dengxianglong/wheels

echo "[build] source: ${SRC} (ref $(git -C "${SRC}" describe --tags 2>/dev/null || echo unknown))"
mkdir -p "${OUT}"

# Force a from-source build: without this, flash-attn's setup.py first tries to
# download a prebuilt wheel from GitHub, which times out on the cluster. Also
# cap the parallel compile jobs to the node's core count.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=64

# Offline build: hopper/setup.py otherwise downloads nvcc 12.6 + ptxas 12.8 from
# developer.download.nvidia.com (image CUDA 12.9 != pinned 12.8). The whitelist
# proxy rejects that host, so skip the download and compile with the image's
# nvcc 12.9 (CUTLASS 3.9.2 in-tree supports it). See setup.py is_offline_build().
export FLASH_ATTENTION_OFFLINE_BUILD=TRUE

# Compile the FA3 Hopper kernel. setup.py install builds + places the .so in
# site-packages. --no-build-isolation so it uses the image's torch/nvcc.
# Keep the FULL log (no tail) so a failure shows the complete traceback.
cd "${SRC}/hopper"
echo "[build] running setup.py install (this takes several minutes)..."
python setup.py install --no-build-isolation 2>&1

# Collect the compiled .so to GPFS.
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "[build] site-packages: ${SITE}"
echo "[build] flash_attn_3 artifacts in site-packages:"
ls -la "${SITE}"/flash_attn_3* 2>&1 || true

# Copy the .so (and any flash_attn_3 package dir) to GPFS for reuse.
cp -f "${SITE}"/flash_attn_3_cuda*.so "${OUT}/" 2>/dev/null || echo "[build] no flash_attn_3_cuda*.so found"
# Also copy the hopper python package (flash_attn_interface) if it landed somewhere.
ls -la "${OUT}"/flash_attn_3* 2>&1 || true

# Verify import.
echo "[build] verifying import..."
python -c "import flash_attn_3; print('flash_attn_3 OK', flash_attn_3.__file__)" 2>&1 || echo "[build] import flash_attn_3 FAILED"

echo "[build] DONE"
