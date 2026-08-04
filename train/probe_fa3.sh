#!/usr/bin/env bash
# Probe: does flash-attn 2.8.3 wheel contain the flash_attn_3 module?
# Run via cctl --code-type git mount. Read-only inspection + a test install.
set +e
unset PIP_CONSTRAINT

WHL=/user/dengxianglong/wheels/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

echo "=== 1. wheel flash_attn entries ==="
python -c "import zipfile; z=zipfile.ZipFile('$WHL'); [print(n) for n in z.namelist() if 'flash_attn' in n.lower()]"

echo "=== 2. wheel flash_attn_3 entries (FA3) ==="
python -c "import zipfile; z=zipfile.ZipFile('$WHL'); hits=[n for n in z.namelist() if 'flash_attn_3' in n.lower()]; print('\n'.join(hits) if hits else 'NONE - flash_attn_3 NOT in wheel')"

echo "=== 3. install 2.8.3 + import test ==="
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
rm -rf "$SITE"/flash_attn "$SITE"/flash_attn-*.dist-info "$SITE"/flash_attn_2_cuda*.so* "$SITE"/flash_attn_3*
pip install --no-deps "$WHL" 2>&1 | tail -5
echo "--- pip show ---"
pip show flash-attn 2>&1 | grep -iE "version|location"
echo "--- import flash_attn ---"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__, flash_attn.__file__)" 2>&1
echo "--- import flash_attn_3 ---"
python -c "import flash_attn_3; print('flash_attn_3 OK', flash_attn_3.__file__)" 2>&1
echo "--- flash_attn package dir listing ---"
python -c "import flash_attn, os; d=os.path.dirname(flash_attn.__file__); print(d); print(sorted(os.listdir(d)))" 2>&1

echo "=== 4. transformers FA3 import path ==="
grep -n "flash_attn_3\|flash_attn_interface" /usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py 2>/dev/null | head

echo "=== DONE ==="
