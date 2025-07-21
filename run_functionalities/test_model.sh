#!/bin/bash

IMAGE1_PATH="$1"
IMAGE2_PATH="$2"
MASK_PATH="$3"
MODEL_PATH="$4"

if [ -z "$IMAGE1_PATH" ] || [ -z "$IMAGE2_PATH" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <image1_path> <image2_path> <model_path>"
    exit 1
fi

python model_predict.py --model_path "$MODEL_PATH" \
                        --imageA "$IMAGE1_PATH" \
                        --imageB "$IMAGE2_PATH" \
                        --mask "$MASK_PATH" 

if [ $? -ne 0 ]; then
    echo "Error: Model prediction failed."
    exit 1
fi
echo "Model prediction completed successfully."
# Optionally, you can add a line to print the output or save it to a file
