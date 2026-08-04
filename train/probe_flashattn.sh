#!/usr/bin/env bash
# Probe: find the source of the `flash-attn==2.7.3` constraint that pip's
# resolver reports. Checks pip env vars, pip config files, then tries a clean
# install after rm-ing the dist-info. Run via cctl --code-type git mount.
set +e
echo "=== PIP env vars ==="
env | grep -iE "^PIP_|CONSTRAINT" | sort

echo "=== pip config files ==="
for f in /etc/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf /etc/xdg/pip/pip.conf; do
  if [ -f "$f" ]; then echo "--- $f ---"; cat "$f"; fi
done

echo "=== pip config debug ==="
pip config debug 2>&1 | head -30

echo "=== pip install -v dry-run (constraint trace) ==="
pip install --no-deps --dry-run /user/dengxianglong/wheels/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl 2>&1 | tail -20

echo "=== DONE ==="
