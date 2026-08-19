#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [-tp N] [-cp N] [-dp N] [-pp N] [-sp] [-cp-mode MODE] [FEATURE FLAGS]" >&2
    echo "Feature flags: --vocab-padding-en --cp-seq-padding-en --fuse-qkv-en --cp-zigzag-en" >&2
    echo "Example: $0 -tp 2 -sp --vocab-padding-en --fuse-qkv-en" >&2
}

TP_SIZE=1
CP_SIZE=1
DP_SIZE=1
PP_SIZE=1
CP_MODE=ring
SEQUENCE_PARALLEL=false
VOCAB_PADDING=false
CP_SEQUENCE_PADDING=false
FUSE_QKV=false
CP_ZIGZAG=false
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
        --vocab-padding-en|--vocab_padding_en)
            VOCAB_PADDING=true
            shift
            ;;
        --cp-seq-padding-en|--cp_seq_padding_en)
            CP_SEQUENCE_PADDING=true
            shift
            ;;
        --fuse-qkv-en|--fuse_qkv_en)
            FUSE_QKV=true
            shift
            ;;
        --cp-zigzag-en|--cp_zigzag_en)
            CP_ZIGZAG=true
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
if [[ "$VOCAB_PADDING" == true ]]; then
    CONFIG_NAME="${CONFIG_NAME}_vocabpad"
fi
if [[ "$CP_SEQUENCE_PADDING" == true ]]; then
    CONFIG_NAME="${CONFIG_NAME}_cpseqpad"
fi
if [[ "$FUSE_QKV" == true ]]; then
    CONFIG_NAME="${CONFIG_NAME}_fuseqkv"
fi
if [[ "$CP_ZIGZAG" == true ]]; then
    CONFIG_NAME="${CONFIG_NAME}_zigzag"
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
    if [[ "$VOCAB_PADDING" == true ]]; then
        CREATE_CONFIG_ARGS+=(--vocab_padding_en)
    fi
    if [[ "$CP_SEQUENCE_PADDING" == true ]]; then
        CREATE_CONFIG_ARGS+=(--cp_seq_padding_en)
    fi
    if [[ "$FUSE_QKV" == true ]]; then
        CREATE_CONFIG_ARGS+=(--fuse_qkv_en)
    fi
    if [[ "$CP_ZIGZAG" == true ]]; then
        CREATE_CONFIG_ARGS+=(--cp_zigzag_en)
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
