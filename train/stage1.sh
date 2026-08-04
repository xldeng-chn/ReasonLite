#!/usr/bin/env bash
# Stage 1 SFT launch — single-node multi-GPU (default 8xH100).
# Open-r1 SFT via accelerate + DeepSpeed ZeRO-1.
# Global batch = per_device(32) x grad_accum(1) x NPROC GPUs = 32*NPROC.
# Override NPROC for 1-GPU smoke runs (e.g. NPROC=1).
set -euo pipefail

# Coordinates and paths come from the SSOT file; do not hardcode here.
# shellcheck source=setup_env.sh
source "$(dirname "$0")/setup_env.sh"

export WANDB_DISABLED=True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Number of accelerate processes (one per GPU). Default 8; 1-GPU smoke sets NPROC=1.
NPROC="${NPROC:-8}"

# TrlParser loads the YAML via --config (sets dataclass defaults; CLI overrides win).
# The config path is absolute so it resolves regardless of CWD.
cd "${OPENR1_ROOT}"

accelerate launch \
    --num_processes "${NPROC}" \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip 127.0.0.1 \
    --main_process_port 8848 \
    --config_file "${OPENR1_ROOT}/recipes/accelerate_configs/zero1.yaml" \
    "${REASONLITE_REPO_ROOT}/train/sft_reasonlite.py" \
    --config "${REASONLITE_REPO_ROOT}/train/config_stage1.yaml" \
    ${REASONLITE_EXTRA_ARGS:-}

