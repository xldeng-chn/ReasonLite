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

# shellcheck source=setup_env.sh
source "$(dirname "$0")/setup_env.sh"

echo "[launch] mode=${MODE} stage=${STAGE}"
echo "[launch] workspace=${REASONLITE_WORKSPACE_ROOT} dataset=${DATASET_PATH}"
mkdir -p "${OUTPUT_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"

# --- 1. Install Python deps (venv built at runtime on the base image) ---
echo "[launch] installing training requirements"
pip install --no-cache-dir -r "${REASONLITE_REPO_ROOT}/train/requirements_train.txt"

# flash-attn provides the flash_attention_3 backend for H100; the base image
# may already ship it. Install only if importable check fails, since building
# from source is slow and the base image wheels are preferred.
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "[launch] flash_attn missing; installing"
    pip install --no-build-isolation flash-attn
fi

# --- 2. Clone open-r1 (provides src/open_r1/sft.py) ---
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
