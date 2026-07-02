#!/bin/bash
# One-time Fields2Cover install inside Ubuntu Docker container.
# Installs into /workspace/.f2c_install (persisted via volume mount).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
INSTALL_PREFIX="${F2C_INSTALL_PREFIX:-$WORKSPACE/.f2c_install}"
F2C_SRC="${F2C_SRC:-$WORKSPACE/third_party/Fields2Cover}"

echo "=== Fields2Cover install ==="
echo "  workspace:      $WORKSPACE"
echo "  install prefix:   $INSTALL_PREFIX"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends software-properties-common ca-certificates
add-apt-repository ppa:ubuntugis/ppa -y
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake git \
  libeigen3-dev libgdal-dev libpython3-dev python3 python3-pip python3-dev \
  swig libgeos-dev libtinyxml2-dev nlohmann-json3-dev libtbb-dev

python3 -m pip install --upgrade pip
python3 -m pip install numpy

mkdir -p "$(dirname "$F2C_SRC")"
if [ ! -d "$F2C_SRC/.git" ]; then
  git clone --depth 1 https://github.com/Fields2Cover/Fields2Cover.git "$F2C_SRC"
fi

mkdir -p "$F2C_SRC/build"
cd "$F2C_SRC/build"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_TUTORIALS=OFF \
  -DBUILD_DOC=OFF \
  -DUSE_ORTOOLS_RELEASE=ON \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  ..
make -j"$(nproc)"
make install

PY_SITE="$(python3 - <<'PY'
import sysconfig
print(sysconfig.get_path("purelib"))
PY
)"
# When CMAKE_INSTALL_PREFIX is custom, modules land under prefix/lib/pythonX.Y/site-packages
if [ -d "$INSTALL_PREFIX/lib" ]; then
  PY_SITE="$(find "$INSTALL_PREFIX/lib" -type d -name site-packages | head -1)"
fi

ENV_FILE="$WORKSPACE/scripts/docker/f2c_env.sh"
cat > "$ENV_FILE" <<EOF
# Source inside Docker: source /workspace/scripts/docker/f2c_env.sh
export F2C_INSTALL_PREFIX="$INSTALL_PREFIX"
export PYTHONPATH="$PY_SITE:\${PYTHONPATH:-}"
EOF

echo ""
echo "=== Install complete ==="
echo "Run: source $ENV_FILE"
echo "Test: python3 -c \"import fields2cover as f2c; print('Fields2Cover OK')\""
