#!/bin/bash
# Download MBPP dataset to blue storage for offline use

# Blue storage configuration
BLUE_STORAGE="${BLUE_STORAGE:-/blue/qi855292.ucf/ji757406.ucf}"
DATASET_PATH="${BLUE_STORAGE}/datasets/mbpp/full"

echo "[MBPP Download] Starting download to blue storage..."
echo "[MBPP Download] Target: ${DATASET_PATH}"

# Run the Python download script
python3 scripts/download_mbpp_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "[MBPP Download] ✓ Download complete!"
    echo ""
    echo "To use the offline dataset in your experiments, add this to your script:"
    echo "  export MBPP_DATASET_PATH=\"${DATASET_PATH}\""
    echo ""
    echo "Or add it to your ~/.bashrc for permanent use:"
    echo "  echo 'export MBPP_DATASET_PATH=\"${DATASET_PATH}\"' >> ~/.bashrc"
else
    echo "[MBPP Download] ✗ Download failed!"
    exit 1
fi
