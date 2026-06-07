#!/usr/bin/env bash
# =============================================================================
# llama.cpp optimized CLI runner
# Uses every available performance optimization flag for maximum inference speed.
#
# Usage:
#   ./scripts/run-optimized.sh -m /path/to/model.gguf [options...]
#
# Hardware: Intel i5-8365U (4C/8T, AVX2), Intel UHD 620 iGPU
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build_fork/bin/llama-cli"

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
    echo ""
    echo "Available models:"
    find /home/alivetilleve/llama_models/ -name '*.gguf' 2>/dev/null | head -10
    echo ""
    echo "Examples:"
    echo "  $0 -m ~/llama_models/qwen/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -p \"Hello\" -n 256"
    echo "  $0 -m ~/llama_models/qwen/qwen2.5-coder-7.6B-Q4_K_M.gguf -p \"Hello\" -n 128 --temp 0.7"
    exit 1
fi

# --- Extract model path and filter out -m/--model + its argument ---
MODEL_PATH=""
USER_ARGS=()
skip=0
for arg in "$@"; do
    if [ $skip -eq 1 ]; then
        MODEL_PATH="$arg"
        skip=0
        continue
    fi
    if [ "$arg" = "-m" ] || [ "$arg" = "--model" ]; then
        skip=1
        continue
    fi
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
    # Memory: lock in RAM, use mmap (default, fastest)
    --mlock
    # KV Cache: flash attention, Q4 cache reduces bandwidth
    -fa on
    -ctk q4_0
    -ctv q4_0
    # Misc
    --warmup
    --show-timings
    -c 4096
)

# Default temp=0 unless user specified --temp
case " $* " in
    *\ --temp\ *|*\ -temp\ *|*\ --temperature\ *) ;;
    *) OPT_FLAGS+=(--temp 0) ;;
esac

# --- Print config ---
echo ""
echo "=== llama.cpp OPTIMIZED RUN ==="
echo " Model:     $MODEL_PATH"
echo " Threads:   $TOTAL_PHYSICAL (phys cores $PHYSICAL_RANGE)"
echo " Priority:  high  |  Poll: none"
echo " FlashAttn: on    |  KVcache: q4_0"
echo " Mlock:     yes   |  Warmup:  yes"
echo ""

# --- Execute ---
exec "$BINARY" "${OPT_FLAGS[@]}" -m "$MODEL_PATH" "${USER_ARGS[@]}"
