#!/bin/sh
# Compile the GLSL shader to SPIR-V for the floatpack Vulkan backend
# Requires glslc from the Vulkan SDK

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GLSLC="${GLSLC:-glslc}"

echo "Compiling floatpack_fp4.comp -> floatpack_fp4.spv ..."
"$GLSLC" "$SCRIPT_DIR/floatpack_fp4.comp" -o "$SCRIPT_DIR/floatpack_fp4.spv"
echo "Done: floatpack_fp4.spv ($(wc -c < "$SCRIPT_DIR/floatpack_fp4.spv") bytes)"
