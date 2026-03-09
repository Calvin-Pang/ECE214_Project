#!/usr/bin/env bash
# Run all 6 variants in parallel, one per GPU.
# Usage: bash run_all.sh [--epochs N] [--out-dir DIR]
#
# Defaults: --epochs 100, --out-dir results

EPOCHS=100
OUT_DIR=results

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)  EPOCHS="$2";  shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

declare -A GPUS=(
    [pncc]=1
    [pncc_39]=2
    [pncc_42]=3
    [mfcc]=4
    [mfcc_39]=5
    [pncc_mfcc_78]=6
    [pncc_n40]=7
    [pncc_contrast]=1  # shares gpu 1 with pncc; both are lightweight
)

echo "Launching all variants (epochs=$EPOCHS, out-dir=$OUT_DIR) ..."

for VARIANT in "${!GPUS[@]}"; do
    GPU=${GPUS[$VARIANT]}
    LOG="$OUT_DIR/${VARIANT}.log"
    echo "  $VARIANT -> gpu $GPU  (log: $LOG)"
    uv run main.py \
        --variant "$VARIANT" \
        --gpu-id "$GPU" \
        --epochs "$EPOCHS" \
        --out-dir "$OUT_DIR" \
        > "$LOG" 2>&1 &
done

wait
echo "All done. Run 'uv run summary.py --out-dir $OUT_DIR' to see results."
