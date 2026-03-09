#!/usr/bin/env bash
# Run only the two new variants.
# Usage: bash run_new.sh [--epochs N] [--out-dir DIR]

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

echo "Launching new variants (epochs=$EPOCHS, out-dir=$OUT_DIR) ..."

uv run main.py --variant pncc_n40      --gpu-id 7 --epochs "$EPOCHS" --out-dir "$OUT_DIR" \
    > "$OUT_DIR/pncc_n40.log"      2>&1 &

uv run main.py --variant pncc_contrast --gpu-id 6 --epochs "$EPOCHS" --out-dir "$OUT_DIR" \
    > "$OUT_DIR/pncc_contrast.log" 2>&1 &

wait
echo "Done. Run 'uv run summary.py --out-dir $OUT_DIR' to see results."
