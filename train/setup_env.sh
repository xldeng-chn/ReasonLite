#!/usr/bin/env bash
# SSOT for ReasonLite H100 training on the paratera_train cluster.
# Source this file from launch_h100.sh / stage{1,2}.sh — never re-declare
# these values elsewhere. Edit coordinates HERE only.

# --- cctl submission coordinates (paratera_train / H100) ---
export CCTL_CLUSTER="${CCTL_CLUSTER:-paratera_train}"
export CCTL_RESOURCE_POOL="${CCTL_RESOURCE_POOL:-aiforai}"
export CCTL_PROJECT="${CCTL_PROJECT:-neimeng-devbox}"
export CCTL_BILLING="${CCTL_BILLING:-N00007}"
export CCTL_PRIORITY="${CCTL_PRIORITY:-NORMAL}"
export CCTL_GPU_MODEL="${CCTL_GPU_MODEL:-h100}"
export CCTL_GPU_COUNT="${CCTL_GPU_COUNT:-8}"
export CCTL_CPU="${CCTL_CPU:-64}"
export CCTL_MEMORY="${CCTL_MEMORY:-512}"
# Base image: NV PyTorch with CUDA/torch preinstalled; venv built at runtime.
export CCTL_IMAGE="${CCTL_IMAGE:-infra/nvidia-pytorch:latest}"

# --- shared GPFS paths (writable account root) ---
export REASONLITE_WORKSPACE_ROOT="${REASONLITE_WORKSPACE_ROOT:-/user/dengxianglong}"
export DATASET_PATH="${DATASET_PATH:-/user/dengxianglong/datasets/ReasonLite-Dataset}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/user/dengxianglong/outputs}"
export TORCHINDUCTOR_CACHE_DIR="${REASONLITE_WORKSPACE_ROOT}/.cache/torchinductor"

# --- open-r1 source (cloned at runtime into the container) ---
# Codeup mirror of open-r1: training nodes have no public internet, so the
# GitHub origin is unreachable. Codeup is on the Aliyun intranet.
export OPENR1_REPO="${OPENR1_REPO:-git@codeup.aliyun.com:modelbest/xldeng-chn/open-r1.git}"
export OPENR1_ROOT="${OPENR1_ROOT:-/workspace/open-r1}"

# --- pip mirror + egress proxy (training nodes have no public internet) ---
# pip reaches PyPI via the Tsinghua mirror, tunneled through the whitelist
# proxy (which only whitelists PyPI/mirrors, NOT github).
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.tuna.tsinghua.edu.cn}"
export PIP_PROXY="${PIP_PROXY:-http://whitelist-proxy.cybertron.svc.cluster.local:7891}"

# --- ReasonLite repo (mounted into the container via cctl --code-type git) ---
# Codeup is reachable from the training nodes over the Aliyun intranet; the
# branch must match what cctl submits with --git-ref. cctl mounts the cloned
# repo at /local/apps/ReasonLite (probed on the paratera_train node).
export REASONLITE_GIT_REPO="${REASONLITE_GIT_REPO:-git@codeup.aliyun.com:modelbest/xldeng-chn/ReasonLite.git}"
export REASONLITE_GIT_REF="${REASONLITE_GIT_REF:-worktree-train-on-h100}"
export REASONLITE_REPO_ROOT="${REASONLITE_REPO_ROOT:-/local/apps/ReasonLite}"
