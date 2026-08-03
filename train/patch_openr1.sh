#!/usr/bin/env bash
# Patch open-r1's configs.py so its SFTConfig dataclass is resolvable under
# transformers 4.56 + trl 0.18.0.
#
# Root cause: transformers 4.56 added `parallelism_config: Optional["ParallelismConfig"]`
# to TrainingArguments. open-r1's SFTConfig subclasses it and uses
# `from __future__ import annotations` (PEP 563), so TrlParser resolves the
# annotation at runtime — but open-r1 never imports ParallelismConfig, causing
# NameError -> "Type resolution failed for SFTConfig".
#
# Fix: add a direct import of ParallelismConfig from accelerate. This is NOT
# try/except — if accelerate lacks the symbol the import fails fast, which is
# the correct signal (we depend on accelerate exposing it).
#
# Idempotent: skips patching if the import is already present.
set -euo pipefail

# shellcheck source=setup_env.sh
source "$(dirname "$0")/setup_env.sh"

CFG="${OPENR1_ROOT}/src/open_r1/configs.py"
if [ ! -f "${CFG}" ]; then
    echo "[patch] FATAL: ${CFG} not found" >&2
    exit 1
fi

MARKER="from accelerate.parallelism_config import ParallelismConfig  # patched: transformers 4.56 compat"

if grep -qF "patched: transformers 4.56 compat" "${CFG}"; then
    echo "[patch] already patched, skipping"
    exit 0
fi

# Insert the import immediately after the `import trl` line.
python3 - "${CFG}" "${MARKER}" <<'PY'
import sys
cfg, marker = sys.argv[1], sys.argv[2]
with open(cfg) as f:
    lines = f.readlines()
out = []
inserted = False
for line in lines:
    out.append(line)
    if not inserted and line.strip() == "import trl":
        out.append(marker + "\n")
        inserted = True
if not inserted:
    print("[patch] FATAL: `import trl` anchor not found in configs.py", file=sys.stderr)
    sys.exit(1)
with open(cfg, "w") as f:
    f.writelines(out)
print("[patch] inserted ParallelismConfig import into configs.py")
PY
