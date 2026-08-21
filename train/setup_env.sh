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

# --- CUDA allocator ---
# expandable_segments relieves fragmentation (reserved-but-unallocated blocks
# that OOM despite free capacity). Recommended by the torch OOM message itself.
# Declared here so every accelerate rank inherits it via the sourced env.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- shared GPFS paths (writable account root) ---
# REASONLITE_USER is the single source for the GPFS account name; every
# /user/<name> path below derives from it so a new account only overrides
# this one var.
export REASONLITE_USER="${REASONLITE_USER:-dengxianglong}"
export REASONLITE_WORKSPACE_ROOT="${REASONLITE_WORKSPACE_ROOT:-/user/${REASONLITE_USER}}"
export DATASET_PATH="${DATASET_PATH:-${REASONLITE_WORKSPACE_ROOT}/datasets/ReasonLite-Dataset}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REASONLITE_WORKSPACE_ROOT}/outputs}"
export TORCHINDUCTOR_CACHE_DIR="${REASONLITE_WORKSPACE_ROOT}/.cache/torchinductor"

# --- caches redirected to GPFS (container /root overlay has limited space) ---
# datasets writes generated splits to HF_DATASETS_CACHE; the default /root/.cache
# fills up and crashes during split generation. All HF/torch/tmp caches go to
# the shared GPFS account root instead.
export XDG_CACHE_HOME="${REASONLITE_WORKSPACE_ROOT}/.cache"
export HF_HOME="${REASONLITE_WORKSPACE_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${REASONLITE_WORKSPACE_ROOT}/.cache/huggingface/datasets"
export HF_HUB_CACHE="${REASONLITE_WORKSPACE_ROOT}/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="${REASONLITE_WORKSPACE_ROOT}/.cache/huggingface/hub"
export TMPDIR="${REASONLITE_WORKSPACE_ROOT}/.cache/tmp"
# triton, torch and pip default to $HOME (the container overlay) and were the
# one group still landing there: orig_1_2k's log carries exactly one container
# path, `df: /root/.triton/autotune`, while the packed side exports the full set
# via devspace_env.sh and carries none. Beyond the disk-space concern above,
# this is an environment asymmetry between the two sides of a parity
# experiment, and triton's autotune sits on the compute path via liger-kernel.
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"

# --- HuggingFace endpoint (nodes have no public internet to huggingface.co) ---
# hf-mirror.com is the domestic mirror; huggingface_hub/transformers read
# HF_ENDPOINT and route all from_pretrained / hf_hub_download calls through it.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- open-r1 source (cloned at runtime from DevCloud to /local/app) ---
# open-r1 is cloned at runtime by launch_h100.sh from its DevCloud mirror
# into a per-pod path under /local/app. Training nodes CANNOT reach DevCloud
# port 22 (SSH clone times out — verified by smoke 747104), so the clone uses
# the HTTPS endpoint on port 443 instead. DevCloud HTTPS auth is username +
# access token, injected into the URL by launch_h100.sh when OPENR1_GIT_TOKEN
# is set. DevCloud is on the intranet and must NOT be routed through the
# whitelist proxy (PyPI-only; it rejects DevCloud), so launch_h100.sh unsets
# the proxy vars before cloning. OPENR1_GIT_REPO is the SSOT for the clone URL
# (HTTPS, no embedded creds); OPENR1_GIT_USER / OPENR1_GIT_TOKEN are the
# optional secrets (set via cctl --env, NEVER committed to the repo).
export OPENR1_ROOT="${OPENR1_ROOT:-/local/app/open-r1}"
export OPENR1_GIT_REPO="${OPENR1_GIT_REPO:-https://codehub.devcloud.cn-north-4.huaweicloud.com/66cb35255b8140c08f7af25e4a10542d/xldeng-chn/open-r1.git}"
export OPENR1_GIT_USER="${OPENR1_GIT_USER:-}"
export OPENR1_GIT_TOKEN="${OPENR1_GIT_TOKEN:-}"

# --- pip mirror + egress proxy (training nodes have no public internet) ---
# pip reaches PyPI via the Tsinghua mirror, tunneled through the whitelist
# proxy (which only whitelists PyPI/mirrors, NOT github).
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.tuna.tsinghua.edu.cn}"
export PIP_PROXY="${PIP_PROXY:-http://whitelist-proxy.cybertron.svc.cluster.local:7891}"

# --- ReasonLite repo (mounted into the container via cctl --code-type git) ---
# DevCloud is reachable from the training nodes over the intranet; the branch
# must match what cctl submits with --git-ref. cctl mounts the cloned repo at
# /local/apps/ReasonLite (probed on the paratera_train node).
export REASONLITE_GIT_REPO="${REASONLITE_GIT_REPO:-git@codehub.devcloud.cn-north-4.huaweicloud.com:66cb35255b8140c08f7af25e4a10542d/xldeng-chn/ReasonLite.git}"
export REASONLITE_GIT_REF="${REASONLITE_GIT_REF:-parity-baseline}"
export REASONLITE_REPO_ROOT="${REASONLITE_REPO_ROOT:-/local/apps/ReasonLite}"
