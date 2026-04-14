#!/bin/bash
set -e

CHECKPOINT="models/checkpoints/transformer_d128_l3_v1896_noaug_combined_best.pt"
VOCAB=1896
ONNX="models/sign_model.onnx"

echo ""
echo "=== ASL Real-Time Demo ==="
echo ""

# Threshold selection
echo "Select confidence threshold:"
echo "  1) 0.05  — show almost everything"
echo "  2) 0.10  — low filter (default)"
echo "  3) 0.20  — medium filter"
echo "  4) 0.40  — high filter (stricter)"
echo "  5) Custom"
echo ""
read -p "Choice [1-5]: " choice

case $choice in
    1) THRESHOLD=0.05 ;;
    2) THRESHOLD=0.10 ;;
    3) THRESHOLD=0.20 ;;
    4) THRESHOLD=0.40 ;;
    5)
        read -p "Enter threshold (0.0 - 1.0): " THRESHOLD
        ;;
    *)
        echo "Invalid choice, using default 0.10"
        THRESHOLD=0.10
        ;;
esac

echo ""
echo "Threshold: $THRESHOLD"
echo ""

# Export
echo "--- Exporting to ONNX ---"
python -m src.export --checkpoint "$CHECKPOINT" --vocab $VOCAB --output "$ONNX"

echo ""
echo "--- Starting demo (press Q to quit) ---"
echo ""
python -m src.demo --model "$ONNX" --vocab $VOCAB --threshold $THRESHOLD
