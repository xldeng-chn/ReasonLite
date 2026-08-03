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
mkdir -p "${OUTPUT_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"

# Training nodes have no public internet. pip reaches PyPI via the Tsinghua
# mirror through the whitelist proxy (PyPI-only; github is NOT whitelisted,
# so open-r1 is cloned from its Codeup mirror instead — see setup_env.sh).
export http_proxy="${PIP_PROXY}"
export https_proxy="${PIP_PROXY}"

# --- 1. Install Python deps (venv built at runtime on the base image) ---
echo "[launch] installing training requirements"
pip install --no-cache-dir \
    -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
    -r "${REASONLITE_REPO_ROOT}/train/requirements_train.txt"

# datasets 4.0.0 declares pyarrow>=21.0.0 + dill constraints that conflict
# with the base image's pinned pyarrow==19.0.1 / dill==0.3.9 (required by
# cudf/dask). trl depends on datasets, so it triggers the same conflict at
# resolution time. Install both with --no-deps so they reuse the image's
# pyarrow/dill; their other runtime deps (transformers, accelerate, torch)
# are already satisfied by the requirements install above or the base image.
# open-r1 SFT reads local jsonl and does not exercise the pyarrow>=21 API
# surface. Ceiling: replace with a base image that ships datasets/pyarrow>=21.
pip install --no-cache-dir --no-deps -i "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    "datasets==4.0.0" "trl==0.18.0"

# flash-attn provides the flash_attention_3 backend for H100; the base image
# may already ship it. Install only if importable check fails, since building
# from source is slow and the base image wheels are preferred.
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "[launch] flash_attn missing; installing"
    pip install --no-build-isolation -i "${PIP_INDEX_URL}" \
        --trusted-host "${PIP_TRUSTED_HOST}" flash-attn
fi

# --- 2. Clone open-r1 (Codeup mirror; no proxy needed, intranet-reachable) ---
if [ ! -d "${OPENR1_ROOT}/.git" ]; then
    echo "[launch] cloning open-r1 -> ${OPENR1_ROOT}"
    git clone --depth 1 "${OPENR1_REPO}" "${OPENR1_ROOT}"
fi
pip install --no-deps -e "${OPENR1_ROOT}/src"

# --- 3. Wire ReasonLite accelerate config into the open-r1 tree ---
# stage{1,2}.sh reference ${OPENR1_ROOT}/recipes/accelerate_configs/zero1.yaml,
# which open-r1 does not ship. Copy ours in so the path resolves.
mkdir -p "${OPENR1_ROOT}/recipes/accelerate_configs"
cp "${REASONLITE_REPO_ROOT}/recipes/accelerate_configs/zero1.yaml" \
   "${OPENR1_ROOT}/recipes/accelerate_configs/zero1.yaml"

# --- 4. Dispatch ---
# smoke: 3 optimizer steps, no checkpoint saving — verifies memory + path wiring.
# full:  run to completion (num_train_epochs from the yaml).
case "${MODE}" in
    smoke)
        export REASONLITE_EXTRA_ARGS="--max_steps 3 --save_strategy no"
        ;;
    full)
        export REASONLITE_EXTRA_ARGS=""
        ;;
    *)
        echo "[launch] unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac

bash "${REASONLITE_REPO_ROOT}/train/${STAGE}.sh" ${REASONLITE_EXTRA_ARGS}
