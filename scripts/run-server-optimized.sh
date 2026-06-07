#!/usr/bin/env bash
# =============================================================================
# llama.cpp optimized server runner
# Uses every available performance optimization flag for maximum throughput.
#
# Usage:
#   ./scripts/run-server-optimized.sh -m /path/to/model.gguf [options...]
#
# Hardware: Intel i5-8365U (4C/8T, AVX2), Intel UHD 620 iGPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build_fork/bin/llama-server"

# --- Source shared env ---
if [ -f "$SCRIPT_DIR/optimize.env" ]; then
    set -a; source "$SCRIPT_DIR/optimize.env"; set +a
fi

# --- Detect physical cores ---
PHYSICAL_CORES=$(lscpu | awk '/^Core\(s\) per socket:/ {print $4}')
SOCKETS=$(lscpu | awk '/^Socket\(s\):/ {print $2}')
TOTAL_PHYSICAL=$((PHYSICAL_CORES * SOCKETS))
[ -z "$TOTAL_PHYSICAL" ] && TOTAL_PHYSICAL=4
PHYSICAL_RANGE="0-$((TOTAL_PHYSICAL - 1))"

# --- Help ---
if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 -m /path/to/model.gguf [options...]"
    echo "  Default: http://127.0.0.1:8080"
    echo ""
    echo "Available models:"
    find /home/alivetilleve/llama_models/ -name '*.gguf' 2>/dev/null | head -10
    echo ""
    echo "Examples:"
    echo "  $0 -m ~/llama_models/qwen/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    echo "  $0 -m ~/llama_models/qwen/qwen2.5-coder-7.6B-Q4_K_M.gguf -- --port 8081 --host 0.0.0.0"
    exit 1
fi

# --- Extract model path and filter out -m+arg ---
MODEL_PATH=""
USER_ARGS=()
skip=0
for arg in "$@"; do
    if [ $skip -eq 1 ]; then MODEL_PATH="$arg"; skip=0; continue; fi
    if [ "$arg" = "-m" ] || [ "$arg" = "--model" ]; then skip=1; continue; fi
    USER_ARGS+=("$arg")
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No model specified. Use -m /path/to/model.gguf"
    exit 1
fi

# --- Build optimized flags ---
OPT_FLAGS=(
    # CPU-only: 2x faster than Vulkan on this hardware
    -ngl 0
    # Threading: physical cores only, no SMT
    -t "$TOTAL_PHYSICAL"
    --cpu-range "$PHYSICAL_RANGE"
    --cpu-strict 1
    --prio 2
    --poll 0
    # Memory: lock in RAM
    --mlock
    # KV Cache: flash attention, Q4 cache
    -fa on
    -ctk q4_0
    -ctv q4_0
    # Misc
    --warmup
    -c 4096
    # Server-specific: continuous batching, prompt cache
    -np 1
    --cont-batching
    --cache-prompt
    --cache-reuse 256
    --threads-http 2
    --no-webui
    --host 127.0.0.1
    --port 8080
)

# --- Print config ---
echo ""
echo "=== llama.cpp OPTIMIZED SERVER ==="
echo " Model:      $MODEL_PATH"
echo " Threads:    $TOTAL_PHYSICAL (phys cores $PHYSICAL_RANGE)"
echo " Priority:   high  |  Poll: none"
echo " FlashAttn:  on    |  KVcache: q4_0"
echo " Mlock:      yes   |  Warmup:  yes"
echo " Batching:   cont  |  PromptCache: yes (reuse=256)"
echo " HTTP:       2 thr |  WebUI: no"
echo " Listening:  http://127.0.0.1:8080"
echo ""

# --- Execute ---
exec "$BINARY" "${OPT_FLAGS[@]}" -m "$MODEL_PATH" "${USER_ARGS[@]}"
