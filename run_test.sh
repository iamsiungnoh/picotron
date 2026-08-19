#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [-tp N] [-cp N] [-dp N] [-pp N] [-sp] [-cp-mode MODE]" >&2
    echo "Example: $0 -tp 2 -sp" >&2
}

TP_SIZE=1
CP_SIZE=1
DP_SIZE=1
PP_SIZE=1
CP_MODE=ring
SEQUENCE_PARALLEL=false
MODEL_NAME=JackFram/llama-160m

while [[ $# -gt 0 ]]; do
    case "$1" in
        -tp)
            [[ $# -ge 2 ]] || { echo "Missing value for -tp" >&2; exit 2; }
            TP_SIZE=$2
            shift 2
            ;;
        -cp)
            [[ $# -ge 2 ]] || { echo "Missing value for -cp" >&2; exit 2; }
            CP_SIZE=$2
            shift 2
            ;;
        -dp)
            [[ $# -ge 2 ]] || { echo "Missing value for -dp" >&2; exit 2; }
            DP_SIZE=$2
            shift 2
            ;;
        -pp)
            [[ $# -ge 2 ]] || { echo "Missing value for -pp" >&2; exit 2; }
            PP_SIZE=$2
            shift 2
            ;;
        -sp|--sequence-parallel)
            SEQUENCE_PARALLEL=true
            shift
            ;;
        -cp-mode|-cp_mode)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            CP_MODE=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

for size in "$TP_SIZE" "$CP_SIZE" "$DP_SIZE" "$PP_SIZE"; do
    if [[ ! "$size" =~ ^[1-9][0-9]*$ ]]; then
        echo "Parallelism degrees must be positive integers, got: $size" >&2
        exit 2
    fi
done

if [[ "$CP_MODE" != "ring" && "$CP_MODE" != "headwise" ]]; then
    echo "CP_MODE must be 'ring' or 'headwise', got: $CP_MODE" >&2
    exit 2
fi

if [[ "$SEQUENCE_PARALLEL" == true && "$TP_SIZE" -eq 1 ]]; then
    echo "Sequence parallelism requires -tp greater than 1" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_TAG=${MODEL_NAME##*/}
CONFIG_NAME="${MODEL_TAG}_tp${TP_SIZE}_cp${CP_SIZE}_dp${DP_SIZE}_pp${PP_SIZE}_${CP_MODE}"
if [[ "$SEQUENCE_PARALLEL" == true ]]; then
    CONFIG_NAME="${CONFIG_NAME}_sp"
fi
CONFIG_PATH="$SCRIPT_DIR/configs/$CONFIG_NAME/config.json"
WORLD_SIZE=$((TP_SIZE * CP_SIZE * DP_SIZE * PP_SIZE))

cd "$SCRIPT_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Creating $CONFIG_PATH"
    CREATE_CONFIG_ARGS=(
        --out_dir configs
        --exp_name "$CONFIG_NAME"
        --model_name "$MODEL_NAME"
        --tp "$TP_SIZE"
        --cp "$CP_SIZE"
        --cp_mode "$CP_MODE"
        --dp "$DP_SIZE"
        --pp "$PP_SIZE"
    )
    if [[ "$SEQUENCE_PARALLEL" == true ]]; then
        CREATE_CONFIG_ARGS+=(--sequence_parallel)
    fi
    python3 create_config.py "${CREATE_CONFIG_ARGS[@]}"
fi

echo "Running $CONFIG_NAME on $WORLD_SIZE processes"

export CUDA_DEVICE_MAX_CONNECTIONS=1

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$WORLD_SIZE" \
    train.py \
    --config "$CONFIG_PATH"
