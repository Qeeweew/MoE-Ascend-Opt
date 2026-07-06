#!/bin/bash
set -e

SOC_VERSION="${1:-Ascend910_9382}"

# Source CANN environment
_CANN_TOOLKIT_INSTALL_PATH=/usr/local/Ascend/cann-9.0.0
if [ ! -d "$_CANN_TOOLKIT_INSTALL_PATH" ]; then
    _CANN_TOOLKIT_INSTALL_PATH=$(cat /etc/Ascend/ascend_cann_install.info 2>/dev/null | grep "Toolkit_InstallPath" | awk -F'=' '{print $2}')
fi
source ${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh

echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"

# Find ASCConfig.cmake
ASC_CONFIG_CMAKE=$(find "$ASCEND_HOME_PATH" -name "ASCConfig.cmake" -type f 2>/dev/null | head -n1)
if [ -n "$ASC_CONFIG_CMAKE" ]; then
    ASC_CMAKE_DIR=$(dirname "$ASC_CONFIG_CMAKE")
    echo "Found ASCConfig.cmake at: $ASC_CONFIG_CMAKE"
    export CMAKE_PREFIX_PATH="$ASC_CMAKE_DIR:$CMAKE_PREFIX_PATH"
    export ASC_DIR="$ASC_CMAKE_DIR"
fi

ASCEND_INCLUDE_DIR=${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include
CURRENT_DIR=$(pwd)
OUTPUT_DIR="${CURRENT_DIR}/moe_ascend_npu/lib"
mkdir -p "$OUTPUT_DIR"

BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cmake \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
    -DASCEND_HOME_PATH=$ASCEND_HOME_PATH \
    -DASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR \
    -DCMAKE_PREFIX_PATH="$ASC_CMAKE_DIR" \
    -DASC_DIR="$ASC_CMAKE_DIR" \
    -DSOC_VERSION=$SOC_VERSION \
    -B "$BUILD_DIR" \
    -S .

cmake --build "$BUILD_DIR" -j 16

echo -e "\e[1;32mBuild complete. Library in $OUTPUT_DIR\e[0m"

# Install the .pth auto-import hook into site-packages so the SGLang monkey
# patches are applied at interpreter start-up. Skip if python is unavailable.
if command -v python >/dev/null 2>&1; then
    python -m moe_ascend_npu._install_pth || \
        echo -e "\e[1;33mWarning: failed to install moe_ascend_npu.pth (run 'python -m moe_ascend_npu._install_pth' manually)\e[0m"
fi
