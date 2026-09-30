#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OPENMVS_DIR="${OPENMVS_DIR:-third_party/openMVS}"
OPENMVS_VERSION="${OPENMVS_VERSION:-v2.3.0}"
VCG_DIR="${VCG_DIR:-third_party/vcglib}"
# IMPORTANT: OpenMVS itself contains tracked build/Utils.cmake. Never put the
# CMake build tree inside the source tree or it can overwrite/delete that file.
BUILD_DIR="${OPENMVS_BUILD_DIR:-third_party/openMVS-build}"
INSTALL_DIR="${OPENMVS_INSTALL_DIR:-$PWD/.local/openmvs}"
# Mesh.cpp can exhaust memory when multiple compiler processes run in a small
# Codespace. Callers on larger machines can still override this explicitly.
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"

mkdir -p third_party .local/bin

# Make the native build self-healing in Debian/Ubuntu Codespaces. OpenMVS's
# Common library explicitly requires a CMake nanoflann package, while the root
# build requires Eigen >= 3.4, OpenCV and Boost.
if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    libeigen3-dev libopencv-dev libnanoflann-dev libboost-all-dev libcgal-dev \
    libgl1 libegl1 libglib2.0-0 libgomp1 libomp-dev ffmpeg
fi

if [[ ! -d "$VCG_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/cdcseacave/VCG.git "$VCG_DIR"
fi

if [[ ! -d "$OPENMVS_DIR/.git" ]]; then
  rm -rf "$OPENMVS_DIR"
  git clone --branch "$OPENMVS_VERSION" --depth 1 --recurse-submodules \
    https://github.com/cdcseacave/openMVS.git "$OPENMVS_DIR"
else
  # Pin the native backend: the development branch changes its dependency
  # contract frequently and can break an otherwise reproducible build.
  git -C "$OPENMVS_DIR" fetch --depth 1 origin "refs/tags/$OPENMVS_VERSION:refs/tags/$OPENMVS_VERSION"
  git -C "$OPENMVS_DIR" checkout --detach "$OPENMVS_VERSION"
  git -C "$OPENMVS_DIR" submodule update --init --recursive
fi

# If an older run deleted the tracked source-side build/Utils.cmake, restore it.
if [[ ! -f "$OPENMVS_DIR/build/Utils.cmake" ]]; then
  git -C "$OPENMVS_DIR" restore --source=HEAD -- build/Utils.cmake build/Modules || true
fi
if [[ ! -f "$OPENMVS_DIR/build/Utils.cmake" ]]; then
  echo "OpenMVS source tree is incomplete: build/Utils.cmake is missing." >&2
  echo "Delete $OPENMVS_DIR and rerun this script." >&2
  exit 1
fi

rm -rf "$BUILD_DIR"
cmake -S "$OPENMVS_DIR" -B "$BUILD_DIR" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DVCG_ROOT="$(realpath "$VCG_DIR")" \
  -DOpenMVS_USE_CUDA=OFF \
  -DOpenMVS_BUILD_VIEWER=OFF \
  -DOpenMVS_BUILD_TOOLS=ON \
  -DOpenMVS_USE_PYTHON=OFF \
  -DOpenMVS_USE_BREAKPAD=OFF \
  -DOpenMVS_USE_SSE=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE

cmake --build "$BUILD_DIR" --parallel "$JOBS" --target \
  InterfaceCOLMAP DensifyPointCloud ReconstructMesh TextureMesh

for binary in InterfaceCOLMAP DensifyPointCloud ReconstructMesh TextureMesh; do
  candidate="$(find "$BUILD_DIR" "$INSTALL_DIR" -type f -name "$binary" -perm -111 -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" ]]; then
    ln -sf "$(realpath "$candidate")" ".local/bin/$binary"
    printf 'Linked %-22s %s\n' "$binary" "$candidate"
  else
    printf 'Missing required OpenMVS binary: %s\n' "$binary" >&2
    exit 1
  fi
done

printf '\nOpenMVS binaries:\n'
for binary in InterfaceCOLMAP DensifyPointCloud ReconstructMesh TextureMesh; do
  "$PWD/.local/bin/$binary" --help >/dev/null
  printf '  OK  %s\n' "$binary"
done
