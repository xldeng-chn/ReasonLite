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

# The base image exports PIP_CONSTRAINT=/etc/pip/constraint.txt, which pins
# flash_attn==2.7.3 (and torch/pyarrow/dill). pip's resolver honors it as a
# hard user constraint, blocking our 2.8.3 wheel (and any version override).
# We manage versions via requirements_train.txt (SSOT), so drop the image's
# constraint entirely.
unset PIP_CONSTRAINT

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
# etc. must exist before datasets tries to write split generation output there,
# and triton writes its autotune cache without creating the directory first.
mkdir -p "${OUTPUT_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TMPDIR}" \
         "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}"

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

# flash-attn FA3: transformers 4.56's flash_attention_3 backend imports the
# flash_attn_3 module, which is NOT in any PyPI wheel — its Hopper kernel must
# be compiled from source (nvcc 12.6/ptxas 12.8, blocked by the air-gapped
# proxy). We pre-compiled it once on a devspace and staged the resulting egg on
# GPFS. Copy the egg into the training pod's site-packages so import works.
# Only needed when the recipe actually asks for flash_attention_3: this branch
# runs flash_attention_2, which the base image already ships (verified
# flash_attn 2.7.3), so hard-failing on an FA3 artifact the run never imports
# would be a false blocker.
# Ceiling: bake FA3 into a custom base image to skip this copy.
RECIPE_ATTN="$(sed -nE 's/^attn_implementation:[[:space:]]*([A-Za-z0-9_]+).*/\1/p' \
    "${REASONLITE_REPO_ROOT}/train/config_${STAGE}.yaml" | head -1)"
echo "[launch] recipe attn_implementation=${RECIPE_ATTN:-<unset>}"
if [ "${RECIPE_ATTN}" = "flash_attention_3" ]; then
    FA3_EGG="${REASONLITE_WORKSPACE_ROOT}/wheels/flash_attn_3-3.0.0b1-py3.12-linux-x86_64.egg"
    if [ ! -d "${FA3_EGG}" ]; then
        echo "[launch] FATAL: FA3 egg not found at ${FA3_EGG}" >&2
        exit 1
    fi
    SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
    echo "[launch] installing FA3 egg into ${SITE_PKGS}"
    rm -rf "${SITE_PKGS}/flash_attn_3-3.0.0b1-py3.12-linux-x86_64.egg"
    cp -r "${FA3_EGG}" "${SITE_PKGS}/"
    # Register the egg on sys.path via easy-install.pth (egg is not zip-safe).
    echo "./flash_attn_3-3.0.0b1-py3.12-linux-x86_64.egg" >> "${SITE_PKGS}/easy-install.pth"
    python -c "import flash_attn_3; print('[launch] FA3 import OK:', flash_attn_3.__file__)"
else
    echo "[launch] skipping FA3 egg install (recipe uses ${RECIPE_ATTN:-<unset>})"
fi

# Fail fast if the recipe asks for FA2 but the image cannot bind it: a silent
# fallback to sdpa produces a perfectly normal-looking loss curve on a baseline
# that is no longer the recipe's baseline.
if [ "${RECIPE_ATTN}" = "flash_attention_2" ]; then
    python - <<'PYFA2'
import sys
from transformers.utils import is_flash_attn_2_available
ok = is_flash_attn_2_available()
print("[launch] is_flash_attn_2_available():", ok)
if not ok:
    sys.exit("[launch] FATAL: recipe asks for flash_attention_2 but the image cannot bind it")
PYFA2
fi

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
# Caller-supplied REASONLITE_EXTRA_ARGS is appended, not discarded: parity runs
# pass per-run overrides (--output_dir with a run-unique path, --max_steps,
# --save_steps) through the cctl entry, and each run MUST write to its own
# directory so no run can overwrite another's checkpoints.
CALLER_EXTRA_ARGS="${REASONLITE_EXTRA_ARGS:-}"
case "${MODE}" in
    smoke)
        export REASONLITE_EXTRA_ARGS="--max_steps 3 --save_strategy no ${CALLER_EXTRA_ARGS}"
        export NPROC="${NPROC:-1}"
        ;;
    full)
        export REASONLITE_EXTRA_ARGS="${CALLER_EXTRA_ARGS}"
        export NPROC="${NPROC:-8}"
        ;;
    *)
        echo "[launch] unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
echo "[launch] extra args: ${REASONLITE_EXTRA_ARGS}"

# HF model/tokenizer downloads (get_tokenizer/get_model inside sft.py) must go
# through the whitelist proxy — hf-mirror.com is not directly reachable. Exclude
# intranet hosts (Codeup, k8s services, GPFS) so they bypass the proxy.
export http_proxy="${PIP_PROXY}"
export https_proxy="${PIP_PROXY}"
export no_proxy="codeup.aliyun.com,.cybertron.svc.cluster.local,.svc.cluster.local,127.0.0.1,localhost"

bash "${REASONLITE_REPO_ROOT}/train/${STAGE}.sh" ${REASONLITE_EXTRA_ARGS}
