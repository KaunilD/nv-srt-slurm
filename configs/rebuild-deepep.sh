#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# srt-slurm setup script for Qwen3.5 + sglang nightly cu13 + dynamo source build on gb300.
# Idempotent — safe to re-run on the same container layer.
set -eux

# ----------------------------------------------------------------------------
# 0. Disable PEP 668 globally for this ephemeral container.
#    srtctl's hardcoded pip install commands don't pass --break-system-packages,
#    and the container is ephemeral, so removing the marker is safe and avoids
#    patching upstream srtctl.
# ----------------------------------------------------------------------------
for d in /usr/lib/python3.* /usr/local/lib/python3.*; do
    [ -e "$d/EXTERNALLY-MANAGED" ] && rm -f "$d/EXTERNALLY-MANAGED"
done

# ----------------------------------------------------------------------------
# 1. System deps (single apt round-trip).
# ----------------------------------------------------------------------------
echo "=== System deps ==="
NEEDED_PKGS=()
[ ! -e /usr/include/infiniband/mlx5dv.h ]   && NEEDED_PKGS+=(libibverbs-dev rdma-core)
command -v protoc       >/dev/null 2>&1 || NEEDED_PKGS+=(protobuf-compiler)
command -v pkg-config   >/dev/null 2>&1 || NEEDED_PKGS+=(pkg-config)
command -v cmake        >/dev/null 2>&1 || NEEDED_PKGS+=(cmake)
[ ! -e /usr/include/openssl/ssl.h ]      && NEEDED_PKGS+=(libssl-dev)
dpkg -s libclang-dev   >/dev/null 2>&1 || NEEDED_PKGS+=(clang libclang-dev)

if [ "${#NEEDED_PKGS[@]}" -gt 0 ]; then
    echo "Installing apt packages: ${NEEDED_PKGS[*]}"
    apt-get update -qq
    apt-get install -y --no-install-recommends "${NEEDED_PKGS[@]}"
fi

# ----------------------------------------------------------------------------
# 2. Maturin (Python build of dynamo Rust bindings).
#    Some sglang images ship the python package without /usr/local/bin/maturin,
#    so force-reinstall to regenerate the script. Fall back to a filesystem
#    search if the binary lands somewhere unusual.
# ----------------------------------------------------------------------------
echo "=== Maturin ==="
pip install --no-cache-dir --force-reinstall --no-deps 'maturin>=1.5'

if ! command -v maturin >/dev/null 2>&1; then
    MATURIN_BIN=$(find /usr/local /usr /root /home -maxdepth 5 -type f -name maturin -executable 2>/dev/null | head -1)
    if [ -n "${MATURIN_BIN:-}" ]; then
        echo "Found maturin at $MATURIN_BIN — symlinking to /usr/local/bin"
        ln -sf "$MATURIN_BIN" /usr/local/bin/maturin
    fi
fi

if ! command -v maturin >/dev/null 2>&1; then
    echo "ERROR: maturin still not found after force-reinstall + filesystem search" >&2
    pip show -f maturin | head -50 >&2
    echo "PATH=$PATH" >&2
    exit 1
fi
echo "maturin: $(command -v maturin) ($(maturin --version))"

# ----------------------------------------------------------------------------
# 3. Rust toolchain (cargo, rustc) — needed for ai-dynamo-runtime native build.
#    Install to /usr/local/cargo and symlink into /usr/local/bin so subsequent
#    bash invocations see the binaries (PATH exports from this subshell don't
#    propagate to the parent shell that runs `maturin build`).
# ----------------------------------------------------------------------------
echo "=== Rust toolchain ==="
if ! command -v cargo >/dev/null 2>&1; then
    export RUSTUP_HOME=/usr/local/rustup
    export CARGO_HOME=/usr/local/cargo
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
    for bin in cargo rustc rustup rustdoc; do
        [ -e "$CARGO_HOME/bin/$bin" ] && ln -sf "$CARGO_HOME/bin/$bin" "/usr/local/bin/$bin"
    done
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo install reported success but binary not on PATH" >&2
    ls -la /usr/local/cargo/bin/ 2>/dev/null || true
    echo "PATH=$PATH" >&2
    exit 1
fi
echo "cargo: $(command -v cargo) ($(cargo --version))"
echo "rustc: $(command -v rustc) ($(rustc --version))"

# ----------------------------------------------------------------------------
# 4. Locate a *real* NVSHMEM install (one that actually has the header).
#    The nvidia-nvshmem-cu13 PyPI wheel is the canonical location in modern
#    sglang containers. We must avoid the tilelang/TVM stub directory which
#    has the path layout but no headers.
# ----------------------------------------------------------------------------
echo "=== Locate NVSHMEM ==="
NVSHMEM_DIR=""
for cand in \
    /usr/local/lib/python3.*/dist-packages/nvidia/nvshmem \
    /opt/nvshmem \
    /usr/local/nvshmem \
    /usr/local/cuda/nvshmem; do
    if [ -f "$cand/include/nvshmem.h" ]; then
        NVSHMEM_DIR="$cand"
        break
    fi
done

# Filesystem fallback, excluding tilelang/TVM stubs.
if [ -z "$NVSHMEM_DIR" ]; then
    HEADER=$(find /usr/local /opt /usr -type f -name "nvshmem.h" 2>/dev/null \
        | grep -v -E "tilelang|3rdparty/tvm" \
        | head -1)
    [ -n "$HEADER" ] && NVSHMEM_DIR="$(dirname "$(dirname "$HEADER")")"
fi

if [ -z "${NVSHMEM_DIR:-}" ] || [ ! -f "$NVSHMEM_DIR/include/nvshmem.h" ]; then
    echo "ERROR: NVSHMEM with usable headers not found." >&2
    echo "Searched candidates:" >&2
    find /usr/local /opt /usr -type d -name nvshmem 2>/dev/null | head -10 >&2
    exit 1
fi
echo "NVSHMEM_DIR=$NVSHMEM_DIR"

# Container ships .so.3 but not the .so symlink that the build expects.
NVSHMEM_LIB="$NVSHMEM_DIR/lib"
if [ ! -f "$NVSHMEM_LIB/libnvshmem_host.so" ] && [ -f "$NVSHMEM_LIB/libnvshmem_host.so.3" ]; then
    echo "Creating missing nvshmem symlink..."
    ln -sf libnvshmem_host.so.3 "$NVSHMEM_LIB/libnvshmem_host.so"
fi

# ----------------------------------------------------------------------------
# 5. Rebuild DeepEP with kNumMaxTopK=16 patch (Qwen3.5 uses topk=10;
#    upstream default is 8 which silently truncates).
# ----------------------------------------------------------------------------
echo "=== Rebuilding DeepEP with kNumMaxTopK=16 ==="

DEEPEP_SRC="/sgl-workspace/DeepEP"
if [ ! -d "$DEEPEP_SRC" ]; then
    echo "ERROR: DeepEP source not found at $DEEPEP_SRC" >&2
    exit 1
fi
cd "$DEEPEP_SRC"

# Patch: source has both kNumMaxTopK and kNumMaxTopk as separate variables.
sed -i 's/kNumMaxTopK[[:space:]]*=[[:space:]]*[0-9][0-9]*/kNumMaxTopK = 16/g' csrc/kernels/internode_ll.cu
sed -i 's/kNumMaxTopk[[:space:]]*=[[:space:]]*[0-9][0-9]*/kNumMaxTopk = 16/g' csrc/kernels/internode_ll.cu
grep -q "kNumMaxTop. = 16" csrc/kernels/internode_ll.cu \
    && echo "Patch verified: kNumMaxTopK/k=16" \
    || { echo "ERROR: kNumMaxTopK patch failed to apply!" >&2; exit 1; }

TORCH_CUDA_ARCH_LIST="10.0" \
NVSHMEM_DIR="$NVSHMEM_DIR" \
pip install -e . --no-build-isolation 2>&1

echo "=== DeepEP rebuild complete ==="
python3 -c "import deep_ep; print('deep_ep imported successfully')"
