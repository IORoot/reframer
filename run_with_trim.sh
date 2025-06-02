#!/bin/bash

# Size of the model to use. Options: "s", "m", "l", "x"
MODEL_SIZE="x"

# Number of frames to skip for saliency detection
SKIP_FRAMES=10

# Number of frames for temporal video smoothing
SMOOTHING_WINDOW=20

# Confidence threshold for saliency detection
# 0.0 to 1.0
CONF_THRESHOLD=0.4

# Use saliency detection
USE_SALIENCY="--use_saliency"

# Number of workers for processing
# This should be set to the number of CPU cores available
MAX_WORKERS=8

# Ratio of the target size to the original size
# 4:3 aspect ratio is 0.75
# 16:9 aspect ratio is 0.5625
# 1:1 aspect ratio is 1.0
TARGET_RATIO=0.5625


# Check if the input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

FILENAME=$1
ABSOLUTE_PATH=$(realpath "$FILENAME")
ABSOLUTE_DIR=$(dirname "$ABSOLUTE_PATH")
OUTPUT_FOLDER="${ABSOLUTE_DIR}/portrait"
PROCESSED_FILENAME="${FILENAME%.*}_processed.${FILENAME##*.}"
TRIMMED_FILENAME="trimmed_${FILENAME%.*}_trimmed.${FILENAME##*.}"

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

PROCESSED_BASENAME=$(basename "${TRIMMED_FILENAME}")
echo "Checking if file exists: ${OUTPUT_FOLDER}/${PROCESSED_BASENAME}"
if [ -f "${OUTPUT_FOLDER}/${PROCESSED_BASENAME}" ]; then
    echo "File already exists: ${OUTPUT_FOLDER}/${PROCESSED_BASENAME}"
    exit 1
fi

echo "Making ${OUTPUT_FOLDER}"
mkdir -p ${OUTPUT_FOLDER}

echo "Processing ${FILENAME}"
python main.py --input "${FILENAME}" --output "${PROCESSED_FILENAME}" --model_size ${MODEL_SIZE} --skip_frames ${SKIP_FRAMES} --smoothing_window ${SMOOTHING_WINDOW} --conf_threshold ${CONF_THRESHOLD} ${USE_SALIENCY} --max_workers ${MAX_WORKERS} --target_ratio ${TARGET_RATIO}

echo "Trimming ${PROCESSED_FILENAME}"
ffmpeg -i ${PROCESSED_FILENAME} -ss 5 -to $(ffmpeg -i ${PROCESSED_FILENAME} 2>&1 | awk -F: '/Duration/ {print $2*3600 + $3*60 + $4 - 5.5}') -c copy ${TRIMMED_FILENAME}

echo "Moving ${TRIMMED_FILENAME} to ${OUTPUT_FOLDER}"
mv ${TRIMMED_FILENAME} ${OUTPUT_FOLDER}

echo "Removing ${PROCESSED_FILENAME}"
# rm ${PROCESSED_FILENAME}

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
