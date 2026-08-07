#!/usr/bin/env bash
# Prove the tokenizer-identity fingerprint fix on the real stack.
#
# The 18 unit tests run against a duck-typed fake: they never touch
# Qwen2TokenizerFast, tokenizers' Rust backend, or the datasets pickler. This
# script closes that gap on-cluster, and it is written to FAIL LOUDLY rather
# than to look reassuring.
#
# Three claims, each independently falsifiable:
#   1. STABLE   -- the identity is identical across two separate processes.
#                  (A repr-derived 0x address would differ here.)
#   2. HITS     -- pass 2 loads the cache instead of recomputing it.
#                  (This is the claim the whole change exists to make true.)
#   3. IDENTICAL-- the cached arrow bytes equal the freshly-computed ones.
#                  (This is the SAFETY claim: labelling changed, data did not.)
#
# Claim 3 is the one that matters most. Claims 1 and 2 only show the cache is
# reusable; claim 3 shows reusing it is not a lie.
set -euo pipefail

REPO="${REASONLITE_REPO_ROOT:-/local/apps/ReasonLite}"
source "${REPO}/train/setup_env.sh"

MODEL="${MODEL_PATH:-/user/dengxianglong/models/Qwen3-0.6B}"
PROBE_CACHE="${REASONLITE_WORKSPACE_ROOT}/.cache/fingerprint_probe"
rm -rf "${PROBE_CACHE}"
mkdir -p "${PROBE_CACHE}"

echo "=== [1/3] identity stability across processes ==="
read_identity() {
    python3 - "$MODEL" <<'PY'
import sys
sys.path.insert(0, __import__("os").environ.get("REASONLITE_REPO_ROOT", "/local/apps/ReasonLite") + "/train")
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "sft_reasonlite",
    os.path.join(os.environ.get("REASONLITE_REPO_ROOT", "/local/apps/ReasonLite"), "train", "sft_reasonlite.py"),
)
sft = importlib.util.module_from_spec(spec); spec.loader.exec_module(sft)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
print(sft.tokenizer_identity(tok))
PY
}
ID_A="$(read_identity)"
ID_B="$(read_identity)"

if [ "${ID_A}" != "${ID_B}" ]; then
    echo "FAIL: identity differs across processes" >&2
    diff <(echo "${ID_A}") <(echo "${ID_B}") >&2 || true
    exit 1
fi
case "${ID_A}" in
    *0x*) echo "FAIL: identity contains a memory address: ${ID_A}" >&2; exit 1 ;;
esac
echo "PASS: stable across processes"
echo "  ${ID_A}"

echo
echo "=== [2/3] + [3/3] cache hit and byte-identical content ==="
# A small synthetic split keeps this to seconds. The mechanism under test is the
# fingerprint, which does not care how many rows there are -- a 4.33M-row run
# would test the same code path and cost 12 minutes to say the same thing.
python3 - "$MODEL" "$PROBE_CACHE" <<'PY'
import importlib.util, os, sys, glob, hashlib

repo = os.environ.get("REASONLITE_REPO_ROOT", "/local/apps/ReasonLite")
spec = importlib.util.spec_from_file_location(
    "sft_reasonlite", os.path.join(repo, "train", "sft_reasonlite.py"))
sft = importlib.util.module_from_spec(spec); spec.loader.exec_module(sft)

model_path, cache_dir = sys.argv[1], sys.argv[2]

from datasets import Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(model_path)
sft.register_stable_tokenizer_hash(tok)

ds = Dataset.from_dict({"text": [f"sample number {i}" for i in range(256)]})

def tokenize(batch, tokenizer):
    # tokenizer lands in fn_kwargs -- exactly how SFTTrainer passes it, which is
    # the arrangement that was falling back to a random fingerprint.
    return tokenizer(batch["text"], add_special_tokens=False)

def run(tag):
    out = ds.map(
        tokenize, batched=True, fn_kwargs={"tokenizer": tok},
        cache_file_name=os.path.join(cache_dir, f"{tag}.arrow"),
        load_from_cache_file=True, desc=f"tokenize:{tag}",
    )
    return out

# Two independent map calls with the SAME cache_file_name: if the fingerprint is
# stable the second is a cache hit. Capture the fingerprints to compare directly.
first = run("probe")
fp_first = first._fingerprint

second = run("probe")
fp_second = second._fingerprint

print(f"fingerprint pass1: {fp_first}")
print(f"fingerprint pass2: {fp_second}")

if fp_first != fp_second:
    print("FAIL: fingerprint changed between identical map calls -> still random", file=sys.stderr)
    sys.exit(1)
print("PASS: fingerprint reproducible across map calls")

# Byte-level equality: recompute with caching bypassed and compare the columns.
fresh = ds.map(
    tokenize, batched=True, fn_kwargs={"tokenizer": tok},
    cache_file_name=os.path.join(cache_dir, "fresh.arrow"),
    load_from_cache_file=False, desc="tokenize:fresh",
)

def digest(d):
    h = hashlib.sha256()
    for row in d:
        h.update(repr(sorted(row.items())).encode())
    return h.hexdigest()

d_cached, d_fresh = digest(second), digest(fresh)
print(f"sha256 cached: {d_cached}")
print(f"sha256 fresh : {d_fresh}")
if d_cached != d_fresh:
    print("FAIL: cached content differs from recomputed content", file=sys.stderr)
    sys.exit(1)
print("PASS: cached bytes identical to recomputed bytes")
PY

echo
echo "=== ALL CHECKS PASSED ==="
