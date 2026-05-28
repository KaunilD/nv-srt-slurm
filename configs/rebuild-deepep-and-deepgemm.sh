#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# srt-slurm setup script: rebuild DeepEP, then install DeepGEMM if the container
# is missing m_grouped_bf16_gemm_nt_masked (required by some Qwen3.5 wide-EP recipes).
set -euxo pipefail

# Keep the original DeepEP rebuild step from the current recipe.
if [ -f /configs/rebuild-deepep.sh ]; then
  bash /configs/rebuild-deepep.sh
fi

# Skip the rebuild if the container already has the required DeepGEMM API.
if python3 - <<'PY'
import deep_gemm
assert hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_masked")
print("Existing deep_gemm is OK:", deep_gemm.__file__)
PY
then
  exit 0
fi

DEEPGEMM_REF="${DEEPGEMM_REF:-714dd1a4a980f7937a74343d19a8eba4fe321480}"

rm -rf /tmp/DeepGEMM
git clone https://github.com/sgl-project/DeepGEMM.git /tmp/DeepGEMM
cd /tmp/DeepGEMM
git checkout "${DEEPGEMM_REF}"
git submodule update --init --recursive

python3 -m pip uninstall -y sgl-deep-gemm deep-gemm deep_gemm || true

bash build_sgl_deep_gemm.sh
python3 -m pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl

python3 - <<'PY'
import deep_gemm
print("deep_gemm:", deep_gemm.__file__)
print("has bf16 masked:", hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_masked"))
assert hasattr(deep_gemm, "m_grouped_bf16_gemm_nt_masked")
PY
