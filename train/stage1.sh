#!/usr/bin/env bash
# Stage 1 SFT launch — multi-node multi-GPU (2 nodes x 8 H100 = 16 GPUs).
# Open-r1 SFT via accelerate + DeepSpeed ZeRO-1.
# Global batch = per_device(16) x grad_accum(1) x (NNODES*NPROC) = 16*16 = 256.
# Rendezvous comes from the PyTorchJob operator env (WORLD_SIZE=node count,
# RANK=node index, MASTER_ADDR/MASTER_PORT); all default to single-node so
# 1-GPU/8-GPU single-node runs (e.g. smoke) still work unchanged.
set -euo pipefail

# Coordinates and paths come from the SSOT file; do not hardcode here.
# shellcheck source=setup_env.sh
source "$(dirname "$0")/setup_env.sh"

export WANDB_DISABLED=True
export NCCL_DEBUG=INFO
# InfiniBand fabric is present (mlx5_0 on the paratera_train H100 nodes); use it
# for cross-node NCCL allreduce instead of falling back to TCP sockets.
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0

# Per-node GPU count (accelerate processes per machine). Default 8; 1-GPU smoke
# sets NPROC=1. Total procs = NNODES * NPROC.
NPROC="${NPROC:-8}"
# Multi-node rendezvous from the operator-injected env (single-node defaults).
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_IP="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-8848}"
NUM_PROCESSES=$((NNODES * NPROC))

# TrlParser loads the YAML via --config (sets dataclass defaults; CLI overrides win).
# The config path is absolute so it resolves regardless of CWD.
cd "${OPENR1_ROOT}"

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NNODES}" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_IP}" \
    --main_process_port "${MASTER_PORT}" \
    --config_file "${OPENR1_ROOT}/recipes/accelerate_configs/zero1.yaml" \
    "${REASONLITE_REPO_ROOT}/train/sft_reasonlite.py" \
    --config "${REASONLITE_REPO_ROOT}/train/config_stage1.yaml" \
    ${REASONLITE_EXTRA_ARGS:-}

