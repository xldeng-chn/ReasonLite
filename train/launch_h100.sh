#!/usr/bin/env bash
# PyTorchJob entry for ReasonLite stage1/stage2 training on a single 8xH100 node.
#
# Usage (inside the container, as the cctl --entry):
#   bash launch_h100.sh smoke stage1   # 3-step smoke test
#   bash launch_h100.sh full  stage2   # full training run
#
# This script is the runtime environment bootstrap: it installs deps, clones
# open-r1, wires the ReasonLite recipes into the open-r1 tree, then dispatches
# to stage{1,2}.sh. All cluster coordinates live in setup_env.sh (SSOT).
set -euo pipefail

MODE="${1:?usage: launch_h100.sh <smoke|full> <stage1|stage2>}"
STAGE="${2:?usage: launch_h100.sh <smoke|full> <stage1|stage2>}"

# Resolve the repo root from the actual mount when present: cctl mounts the
# cloned repo at /local/apps/ReasonLite, but for local runs fall back to the
# file's own parent so the script works outside the cluster too.
if [ -d /local/apps/ReasonLite/train ]; then
    export REASONLITE_REPO_ROOT=/local/apps/ReasonLite
fi

# shellcheck source=setup_env.sh
source "$(dirname "$0")/setup_env.sh"

echo "[launch] mode=${MODE} stage=${STAGE}"
echo "[launch] workspace=${REASONLITE_WORKSPACE_ROOT} dataset=${DATASET_PATH}"
# Pre-create all cache dirs on GPFS (sourced from setup_env.sh). HF_DATASETS_CACHE
# etc. must exist before datasets tries to write split generation output there.
mkdir -p "${OUTPUT_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TMPDIR}"

# Training nodes have no public internet. pip reaches PyPI via the Tsinghua
# mirror through the whitelist proxy (PyPI-only; github is NOT whitelisted,
# so open-r1 is cloned from its Codeup mirror instead — see setup_env.sh).
# The proxy is scoped to the pip installs ONLY: Codeup is on the intranet and
# must NOT be routed through the proxy (it breaks git clone), so we unset the
# proxy vars before cloning open-r1 below.
pip_install() {
    http_proxy="${PIP_PROXY}" https_proxy="${PIP_PROXY}" \
        pip install --no-cache-dir \
        -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" "$@"
}

# --- 1. Install Python deps (venv built at runtime on the base image) ---
echo "[launch] installing training requirements"
pip_install -r "${REASONLITE_REPO_ROOT}/train/requirements_train.txt"

# datasets 4.0.0 declares pyarrow>=21.0.0 + dill constraints that conflict
# with the base image's pinned pyarrow==19.0.1 / dill==0.3.9 (required by
# cudf/dask). trl depends on datasets, so it triggers the same conflict at
# resolution time. Install both with --no-deps so they reuse the image's
# pyarrow/dill; their other runtime deps (transformers, accelerate, torch)
# are already satisfied by the requirements install above or the base image.
# open-r1 SFT reads local jsonl and does not exercise the pyarrow>=21 API
# surface. Ceiling: replace with a base image that ships datasets/pyarrow>=21.
pip_install --no-deps "datasets==4.0.0" "trl==0.18.0"

# flash-attn: the base image ships 2.7.3, which lacks the flash_attn_3 module
# (FA3 for H100, introduced in flash-attn 2.8). transformers 4.56's
# flash_attention_3 backend imports flash_attn_3, so we override with a
# matching prebuilt 2.8.3 wheel from GPFS (cu12 / torch2.7 / cxx11abiTRUE /
# cp312 — verified against the image's torch 2.7.0a0+nv25.04, abi=True).
FA_WHL="${REASONLITE_WORKSPACE_ROOT}/wheels/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
if [ ! -f "${FA_WHL}" ]; then
    echo "[launch] FATAL: flash-attn wheel not found at ${FA_WHL}" >&2
    exit 1
fi
echo "[launch] installing flash-attn 2.8.3 (FA3) from ${FA_WHL}"
# The image's flash-attn 2.7.3 was NOT installed via pip (pip show does not
# see it), so `pip uninstall` cannot remove it. pip's resolver nonetheless
# scans site-packages, finds the 2.7.3 dist-info, and treats it as a hard
# constraint that conflicts with the 2.8.3 wheel. Delete the leftover files
# directly so the resolver sees a clean site-packages.
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE_PKGS}/flash_attn" \
       "${SITE_PKGS}"/flash_attn-*.dist-info \
       "${SITE_PKGS}"/flash_attn_2_cuda*.so*
pip install --no-deps "${FA_WHL}"

# --- 2. open-r1 (preinstalled on shared GPFS; no online clone) ---
# Training nodes cannot reach codeup.aliyun.com:22, so open-r1 is placed on
# GPFS ahead of time (OPENR1_ROOT). Verify it exists, then editable-install.
# Install from the repo root (setup.py/pyproject.toml live there, not in src/).
if [ ! -f "${OPENR1_ROOT}/setup.py" ] && [ ! -f "${OPENR1_ROOT}/pyproject.toml" ]; then
    echo "[launch] FATAL: open-r1 setup.py/pyproject.toml not found at ${OPENR1_ROOT}" >&2
    echo "[launch]        pre-clone open-r1 onto GPFS at OPENR1_ROOT" >&2
    exit 1
fi
pip install --no-deps -e "${OPENR1_ROOT}"

# Patch open-r1 configs.py for transformers 4.56 compat (ParallelismConfig
# type-resolution). Must run AFTER editable install so the source is in place.
bash "${REASONLITE_REPO_ROOT}/train/patch_openr1.sh"

# --- 3. Wire ReasonLite accelerate config into the open-r1 tree ---
# stage{1,2}.sh reference ${OPENR1_ROOT}/recipes/accelerate_configs/zero1.yaml,
# which open-r1 does not ship. Copy ours in so the path resolves.
mkdir -p "${OPENR1_ROOT}/recipes/accelerate_configs"
cp "${REASONLITE_REPO_ROOT}/recipes/accelerate_configs/zero1.yaml" \
   "${OPENR1_ROOT}/recipes/accelerate_configs/zero1.yaml"

# --- 4. Dispatch ---
# smoke: 3 optimizer steps, no checkpoint saving — verifies memory + path wiring.
# full:  run to completion (num_train_epochs from the yaml).
# NPROC: number of accelerate processes (one per GPU). Default 8 (matches the
# 8xH100 target); 1-GPU smoke sets NPROC=1 to avoid 8 processes on 1 GPU.
case "${MODE}" in
    smoke)
        export REASONLITE_EXTRA_ARGS="--max_steps 3 --save_strategy no"
        export NPROC="${NPROC:-1}"
        ;;
    full)
        export REASONLITE_EXTRA_ARGS=""
        export NPROC="${NPROC:-8}"
        ;;
    *)
        echo "[launch] unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac

# HF model/tokenizer downloads (get_tokenizer/get_model inside sft.py) must go
# through the whitelist proxy — hf-mirror.com is not directly reachable. Exclude
# intranet hosts (Codeup, k8s services, GPFS) so they bypass the proxy.
export http_proxy="${PIP_PROXY}"
export https_proxy="${PIP_PROXY}"
export no_proxy="codeup.aliyun.com,.cybertron.svc.cluster.local,.svc.cluster.local,127.0.0.1,localhost"

bash "${REASONLITE_REPO_ROOT}/train/${STAGE}.sh" ${REASONLITE_EXTRA_ARGS}
