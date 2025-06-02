#!/bin/bash
# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                                                                              │
# │         Use AI to crop a video to a different aspect ratio.                  │
# │                                                                              │
# ╰──────────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────────────────────────────────────╮
# │                        VARIABLES                         │
# ╰──────────────────────────────────────────────────────────╯

# Ratio of the target size to the original size
# 4:3 aspect ratio is 0.75
# 16:9 aspect ratio is 0.5625
# 1:1 aspect ratio is 1.0
TARGET_RATIO=0.5625

# Number of workers for processing
# This should be set to the number of CPU cores available
MAX_WORKERS=6

# Object detection model to use (yolo, ssd, or faster_rcnn)
DETECTOR="yolo"

# Number of frames to skip for saliency detection
SKIP_FRAMES=10

# Confidence threshold for saliency detection
# 0.0 to 1.0
CONF_THRESHOLD=0.4

# Size of the model to use. Options: "s", "m", "l", "x"
MODEL_SIZE="x"

# List of object class IDs to track (0=person, 1=bicycle, etc.)
OBJECT_CLASSES="0"

# Number of objects to track in frame
TRACK_COUNT=1

# Padding ratio for crop window (adds space around detected objects)
PADDING_RATIO=0.1

# Weight for object size in crop calculation (larger objects are more important)
SIZE_WEIGHT=0.4

# Weight for object center in crop calculation (centered objects preferred)
CENTER_WEIGHT=0.3

# Weight for object motion in crop calculation (prioritizes moving objects)
MOTION_WEIGHT=0.3

# Weight for object history in crop calculation (smoothness over time)
HISTORY_WEIGHT=0.1

# Weight for saliency in crop calculation (visual importance of regions)
SALIENCY_WEIGHT=0.4

# Enable face detection for crop calculation
FACE_DETECTION=false

# Enable weighted center for crop calculation
WEIGHTED_CENTER=false

# Enable blending of saliency map with detected objects for crop calculation
BLEND_SALIENCY=false

# Enable temporal smoothing for crop windows
APPLY_SMOOTHING=true

# Number of frames for temporal video smoothing
SMOOTHING_WINDOW=30

# Position inertia for smoothing (how much crop position sticks to previous frame)
POSITION_INERTIA=0.8

# Size inertia for smoothing (how much crop size sticks to previous frame)
SIZE_INERTIA=0.9

# Enable debug mode
DEBUG=false

# ╭──────────────────────────────────────────────────────────╮
# │                          Usage.                          │
# ╰──────────────────────────────────────────────────────────╯

usage()
{
    if [ "$#" -lt 1 ]; then
        printf "ℹ️ Usage:\n $0 -i <INPUT_FILE> [OPTIONS]\n\n" >&2 

        printf "Summary:\n"
        printf "Convert the aspect ratio of a video to a different aspect ratio.\n\n"

        printf "Required Arguments:\n"
        printf " -i | --input <INPUT_FILE>\n"
        printf "\tThe name of the input file.\n\n"

        printf "Optional Arguments:\n"
        printf " --target_ratio <RATIO>\n"
        printf "\tTarget aspect ratio (default: 0.5625 for 16:9)\n\n"
        printf " --max_workers <NUM>\n"
        printf "\tNumber of worker threads (default: 6)\n\n"
        printf " --detector <MODEL>\n"
        printf "\tObject detection model (default: yolo)\n\n"
        printf " --skip_frames <NUM>\n"
        printf "\tFrames to skip for detection (default: 10)\n\n"
        printf " --conf_threshold <NUM>\n"
        printf "\tConfidence threshold 0.0-1.0 (default: 0.4)\n\n"
        printf " --model_size <SIZE>\n"
        printf "\tModel size: s/m/l/x (default: x)\n\n"
        printf " --object_classes <CLASSES>\n"
        printf "\tObject classes to track (default: 0)\n\n"
        printf " --track_count <NUM>\n"
        printf "\tNumber of objects to track (default: 1)\n\n"
        printf " --padding_ratio <RATIO>\n"
        printf "\tPadding around objects (default: 0.1)\n\n"
        printf " --size_weight <WEIGHT>\n"
        printf "\tWeight for object size (default: 0.4)\n\n"
        printf " --center_weight <WEIGHT>\n"
        printf "\tWeight for center position (default: 0.3)\n\n"
        printf " --motion_weight <WEIGHT>\n"
        printf "\tWeight for object motion (default: 0.3)\n\n"
        printf " --history_weight <WEIGHT>\n"
        printf "\tWeight for object history (default: 0.1)\n\n"
        printf " --saliency_weight <WEIGHT>\n"
        printf "\tWeight for saliency (default: 0.4)\n\n"
        printf " --face_detection\n"
        printf "\tEnable face detection (flag)\n\n"
        printf " --weighted_center\n"
        printf "\tEnable weighted center (flag)\n\n"
        printf " --blend_saliency\n"
        printf "\tEnable saliency blending (flag)\n\n"
        printf " --apply_smoothing\n"
        printf "\tEnable temporal smoothing (flag)\n\n"
        printf " --smoothing_window <NUM>\n"
        printf "\tSmoothing window size (default: 10)\n\n"
        printf " --position_inertia <WEIGHT>\n"
        printf "\tPosition inertia weight (default: 0.8)\n\n"
        printf " --size_inertia <WEIGHT>\n"
        printf "\tSize inertia weight (default: 0.9)\n\n"
        printf " --debug\n"
        printf "\tEnable debug mode (flag)\n\n"

        exit 1
    fi
}

# ╭──────────────────────────────────────────────────────────╮
# │         Take the arguments from the command line         │
# ╰──────────────────────────────────────────────────────────╯
function arguments()
{
    POSITIONAL_ARGS=()

    while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            FILENAME=$(realpath "$2")
            shift
            shift
            ;;
        --target_ratio)
            TARGET_RATIO="$2"
            shift
            shift
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift
            shift
            ;;
        --detector)
            DETECTOR="$2"
            shift
            shift
            ;;
        --skip_frames)
            SKIP_FRAMES="$2"
            shift
            shift
            ;;
        --conf_threshold)
            CONF_THRESHOLD="$2"
            shift
            shift
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift
            shift
            ;;
        --object_classes)
            OBJECT_CLASSES="$2"
            shift
            shift
            ;;
        --track_count)
            TRACK_COUNT="$2"
            shift
            shift
            ;;
        --padding_ratio)
            PADDING_RATIO="$2"
            shift
            shift
            ;;
        --size_weight)
            SIZE_WEIGHT="$2"
            shift
            shift
            ;;
        --center_weight)
            CENTER_WEIGHT="$2"
            shift
            shift
            ;;
        --motion_weight)
            MOTION_WEIGHT="$2"
            shift
            shift
            ;;
        --history_weight)
            HISTORY_WEIGHT="$2"
            shift
            shift
            ;;
        --saliency_weight)
            SALIENCY_WEIGHT="$2"
            shift
            shift
            ;;
        --face_detection)
            FACE_DETECTION=true
            shift
            ;;
        --weighted_center)
            WEIGHTED_CENTER=true
            shift
            ;;
        --blend_saliency)
            BLEND_SALIENCY=true
            shift
            ;;
        --apply_smoothing)
            APPLY_SMOOTHING=true
            shift
            ;;
        --smoothing_window)
            SMOOTHING_WINDOW="$2"
            shift
            shift
            ;;
        --position_inertia)
            POSITION_INERTIA="$2"
            shift
            shift
            ;;
        --size_inertia)
            SIZE_INERTIA="$2"
            shift
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
    done
}

# ╭──────────────────────────────────────────────────────────╮
# │                                                          │
# │                      Main Function                       │
# │                                                          │
# ╰──────────────────────────────────────────────────────────╯
function main()
{
    # Check if the input file is provided
    if [ -z "${FILENAME}" ]; then
        usage
        exit 1
    fi

    BASE_FILENAME=$(basename "${FILENAME}")
    ABSOLUTE_PATH=$(realpath "$FILENAME")
    ABSOLUTE_DIR=$(dirname "$ABSOLUTE_PATH")
    OUTPUT_FOLDER="${ABSOLUTE_DIR}/portrait"
    PROCESSED_FILENAME="${FILENAME%.*}_processed.${FILENAME##*.}"
    PROCESSED_BASENAME=$(basename "${PROCESSED_FILENAME}")

    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    echo "Checking if file exists: ${OUTPUT_FOLDER}/${PROCESSED_FILENAME}"
    if [ -f "${OUTPUT_FOLDER}/${PROCESSED_BASENAME}" ]; then
        echo "File already exists: ${OUTPUT_FOLDER}/${PROCESSED_BASENAME}"
        exit 1
    fi

    echo "Making ${OUTPUT_FOLDER}"
    mkdir -p ${OUTPUT_FOLDER}

    # Build command with conditional flags
    CMD="python main.py --input \"${FILENAME}\" \
                    --output \"${PROCESSED_FILENAME}\" \
                    --target_ratio ${TARGET_RATIO} \
                    --max_workers ${MAX_WORKERS} \
                    --detector ${DETECTOR} \
                    --skip_frames ${SKIP_FRAMES} \
                    --conf_threshold ${CONF_THRESHOLD} \
                    --model_size ${MODEL_SIZE} \
                    --object_classes ${OBJECT_CLASSES} \
                    --track_count ${TRACK_COUNT} \
                    --padding_ratio ${PADDING_RATIO} \
                    --size_weight ${SIZE_WEIGHT} \
                    --center_weight ${CENTER_WEIGHT} \
                    --motion_weight ${MOTION_WEIGHT} \
                    --history_weight ${HISTORY_WEIGHT} \
                    --saliency_weight ${SALIENCY_WEIGHT} \
                    --smoothing_window ${SMOOTHING_WINDOW} \
                    --position_inertia ${POSITION_INERTIA} \
                    --size_inertia ${SIZE_INERTIA}"

    # Add boolean flags only if they are true
    if [ "${FACE_DETECTION}" = true ]; then
        CMD="${CMD} --face_detection"
    fi
    if [ "${WEIGHTED_CENTER}" = true ]; then
        CMD="${CMD} --weighted_center"
    fi
    if [ "${BLEND_SALIENCY}" = true ]; then
        CMD="${CMD} --blend_saliency"
    fi
    if [ "${APPLY_SMOOTHING}" = true ]; then
        CMD="${CMD} --apply_smoothing"
    fi
    if [ "${DEBUG}" = true ]; then
        CMD="${CMD} --debug"
    fi

    echo "Processing:"
    echo "${CMD}"

    eval "${CMD}"

    mv ${PROCESSED_FILENAME} ${OUTPUT_FOLDER}/${BASE_FILENAME}
    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}

usage $@
arguments $@
main $@