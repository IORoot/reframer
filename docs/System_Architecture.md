# Content-Aware Video Cropping: System Architecture

This document outlines the architecture and process flow of the Content-Aware Video Cropping system, which automatically converts landscape videos to portrait format while focusing on the most important subjects.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INITIALIZATION                                    │
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │ Load Command│   │  Initialize │   │  Initialize │   │   Initialize    │  │
│  │   Line Args │─▶│ YOLOv8 Model│──▶│   Tracker   │─▶│ Crop Calculator │  │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VIDEO LOADING & ANALYSIS                            │
│                                                                             │
│  ┌─────────────┐   ┌─────────────────────────────────────────────────────┐  │
│  │ Load Input  │   │              PHASE 1: DETECTION                     │  │
│  │    Video    │─▶│                                                      │  │
│  └─────────────┘   │  ┌──────────────────────────────────────────────┐    │  │
│                    │  │         ThreadPoolExecutor                   │    │  │
│                    │  │  ┌─────────┐   ┌─────────────┐   ┌─────────┐ │    │  │
│                    │  │  │ Extract │   │   Detect    │   │  Track  │ │    │  │
│                    │  │  │ Keyframe│─▶│   Objects   │──▶│ Objects │ │    │  │
│                    │  │  └─────────┘   └─────────────┘   └─────────┘ │    │  │
│                    │  │        │              │               │      │    │  │
│                    │  │        └──────────────┴───────────────┘      │    │  │
│                    │  │                      ▲                       │    │  │
│                    │  │                      │                       │    │  │
│                    │  │                      │ Every Nth Frame       │    │  │
│                    │  │                      │                       │    │  │
│                    │  │  ┌───────────────────┴──────────────────┐    │    │  │
│                    │  │  │       Frame Generator Loop           │    │    │  │
│                    │  │  └──────────────────────────────────────┘    │    │  │
│                    │  └──────────────────────────────────────────────┘    │  │
│                    └──────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CROP WINDOW CALCULATION                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                PHASE 2: KEYFRAME PROCESSING                         │    │
│  │                                                                     │    │
│  │  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐   │    │
│  │  │ Get Tracked │   │ Detect Faces &  │   │ Calculate Crop Window│   │    │
│  │  │   Objects   │──▶│ Saliency Regions│──▶│ with Prioritization │   │    │
│  │  └─────────────┘   └─────────────────┘   └──────────────────────┘   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              PHASE 3: INTERPOLATION                                 │    │
│  │                                                                     │    │
│  │  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐   │    │
│  │  │Find Nearest │   │ Calculate       │   │ Apply Numpy-based    │   │    │
│  │  │ Keyframes   │─▶│ Interpolation   │──▶│ Linear Interpolation │   │    │
│  │  └─────────────┘   │   Factor        │   │                      │   │    │
│  │                    └─────────────────┘   └──────────────────────┘   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              PHASE 4: TEMPORAL SMOOTHING                            │    │
│  │                                                                     │    │
│  │  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────────┐    │    │
│  │  │Apply Moving │   │ Balance         │   │ Generate Final      │    │    │
│  │  │  Average    │─▶│ Responsiveness  │──▶│ Smoothed Windows    │    │    │
│  │  └─────────────┘   │ & Stability     │   │                     │    │    │
│  │                    └─────────────────┘   └─────────────────────┘    │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VIDEO GENERATION                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                PHASE 5: OUTPUT CREATION                             │    │
│  │                                                                     │    │
│  │  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────────┐    │    │
│  │  │ Reset Video │   │ Create Video    │   │ Process Each Frame  │    │    │
│  │  │ to Beginning│─▶│   Writer        │──▶│ with Crop Window    │    │    │
│  │  └─────────────┘   └─────────────────┘   └─────────────────────┘    │    │
│  │                                                  │                  │    │
│  │                                                  ▼                  │    │
│  │                                        ┌─────────────────────┐      │    │
│  │                                        │ Write Frame to      │      │    │
│  │                                        │ Output Video        │      │    │
│  │                                        └─────────────────────┘      │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FINALIZATION                                      │
│                                                                             │
│  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────────────────┐    │
│  │ Release     │   │ Calculate       │   │ Display Processing Summary  │    │
│  │ Resources   │─▶│ Elapsed Time    │──▶│ and Output File Location    │    │
│  └─────────────┘   └─────────────────┘   └─────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Process Flow Description

### 1. Initialization

The system begins by setting up all required components:

- **Command Line Arguments**: Parses user inputs such as input/output paths, model size, and processing parameters.
- **YOLOv8 Model**: Initializes the object detection model with the specified size (nano, small, medium, large, or xlarge).
- **Object Tracker**: Sets up the tracking mechanism that will maintain object identity across frames.
- **Crop Calculator**: Prepares the component that will determine optimal crop positions.

### 2. Video Loading & Analysis (Phase 1)

This phase handles the initial processing of the video:

- **Video Loading**: Opens the input video file and extracts metadata (resolution, frame rate, total frames).
- **Parallel Processing**: Uses ThreadPoolExecutor to process keyframes in parallel.
- **Keyframe Extraction**: Processes every Nth frame based on the `skip_frames` parameter to reduce computation.
- **Object Detection**: Uses YOLOv8 to identify subjects of interest in each keyframe.
- **Object Tracking**: Maintains identity of detected objects across consecutive keyframes.
- **Data Storage**: Organizes tracked objects by frame index for later processing.

### 3. Crop Window Calculation

This multi-phase process determines the optimal crop window for each frame:

#### Phase 2: Keyframe Processing
- Retrieves tracked objects for each keyframe
- Performs additional face detection if enabled
- Calculates saliency maps to identify visually interesting regions
- Calculates optimal crop window based on subject prioritization rules
- Assigns importance values to subjects based on class, size, position, and confidence
- Blends object-based and saliency-based interest points when appropriate

#### Phase 3: Interpolation
- For non-keyframes, identifies the nearest keyframes before and after
- Calculates an interpolation factor based on the frame's position
- Applies numpy-based linear interpolation between keyframe crop windows for faster processing
- Ensures smooth transitions between keyframes

#### Phase 4: Temporal Smoothing
- Applies moving average smoothing to crop windows
- Balances responsiveness (following subjects) and stability (avoiding jerky movements)
- Generates final smoothed crop windows for all frames
- Applies inertia to resist sudden changes

### 4. Video Generation (Phase 5)

This phase creates the final output video:

- **Video Reset**: Returns to the beginning of the input video
- **Writer Initialization**: Creates a video writer with the target dimensions (portrait format)
- **Frame Processing**:
  - For each frame, applies the corresponding crop window
  - Extracts the relevant portion of the frame
  - Maintains the target aspect ratio (9:16)
- **Output Creation**: Writes each cropped frame to the output video file

### 5. Finalization

The final phase completes the process:

- **Resource Release**: Properly closes video reader and writer
- **Performance Measurement**: Calculates and displays total processing time
- **Summary Display**: Shows processing statistics and the location of the output file

## Data Flow

The system processes data through the following enhanced sequence:

```
Input Video → Parallel Frame Extraction → Object Detection → Saliency Detection → 
Object Tracking → Subject Prioritization → Crop Window Calculation → 
Numpy-based Interpolation → Temporal Smoothing → Frame Cropping → Output Video
```

## Key Components Interaction

The system architecture consists of the following key components:

- **Object Detector**: Utilizes YOLOv8 to identify subjects in frames and provides detection results to the Object Tracker
- **Saliency Detector**: Identifies visually interesting regions when objects are few or missing
- **Object Tracker**: Maintains subject identity across frames and feeds tracked objects to the Crop Calculator
- **Crop Calculator**: Determines optimal crop windows based on subject importance and saliency, then passes them to the Smoother
- **Smoother**: Applies temporal smoothing to ensure stable transitions and provides final crop windows to the Video Processor
- **Video Processor**: Applies crop windows to frames and generates the final portrait-oriented output video
- **ThreadPoolExecutor**: Manages parallel processing of keyframes for improved performance

## Advanced Features

- **Subject Prioritization**: Intelligently weights subjects based on class (people, faces have higher priority), size, position, and confidence score
- **Saliency Detection**: Identifies visually interesting regions when objects are few or missing
- **Multi-subject Handling**: Makes informed decisions about which subjects to keep in frame when multiple subjects are present
- **Temporal Consistency**: Ensures smooth tracking and transitions even when subjects temporarily disappear from view
- **Adaptive Processing**: Adjusts processing parameters based on scene complexity and subject movement

## Performance Optimization

- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent keyframe analysis
- **Keyframe Processing**: Analyzes only a subset of frames to reduce computational load
- **Numpy-based Interpolation**: Uses numpy for faster interpolation calculations
- **Saliency Optimization**: Resizes frames for faster saliency detection
- **GPU Acceleration**: Utilizes GPU for faster object detection when available
- **Progressive Processing**: Implements a phased approach with progress reporting

## How to Run the Code

### 1. Environment Setup
First, ensure you have Python 3.8+ installed. Then set up your environment:

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install ultralytics
pip install opencv-python
pip install numpy
pip install tqdm
```

### 2. Download YOLOv8 Models
Run the model downloader to get the YOLOv8 weights:

```bash
python download_models.py
```
This will create a `models` directory and download all YOLOv8 model variants (n, s, m, l, x).

### 3. Run the Main Program
The main program supports various command-line arguments for customization:

```bash
python main.py --input <input_video_path> --output <output_video_path> [options]
```

Example commands:

```bash
# Basic usage with default settings (nano model)
python main.py --input videos/landscape.mp4 --output videos/portrait.mp4

# Use medium-sized model for better accuracy
python main.py --input videos/landscape.mp4 --output videos/portrait.mp4 --model_size m

# Advanced usage with custom parameters
python main.py --input videos/landscape.mp4 --output videos/portrait.mp4 \
    --model_size l \
    --skip_frames 5 \
    --conf_threshold 0.6 \
    --use_saliency \
    --max_workers 8
```

### Available Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--input` | Input video path | Required | Any video file path |
| `--output` | Output video path | Required | Any video file path |
| `--model_size` | YOLOv8 model size | 'n' | 'n', 's', 'm', 'l', 'x' |
| `--skip_frames` | Process every nth frame | 10 | Any positive integer |
| `--conf_threshold` | Detection confidence threshold | 0.5 | 0.0 to 1.0 |
| `--use_saliency` | Enable saliency detection | False | Flag |
| `--max_workers` | Number of parallel workers | 4 | Any positive integer |
| `--target_ratio` | Output aspect ratio | 9/16 | Any positive float |
| `--smoothing_window` | Frames for temporal smoothing | 30 | Any positive integer |

### Performance Tips
- For faster processing: Use `--model_size n` and higher `--skip_frames`
- For better quality: Use `--model_size m` or `l` and lower `--skip_frames`
- For multi-core CPUs: Increase `--max_workers`
- For smoother motion: Increase `--smoothing_window`
- For better detection: Adjust `--conf_threshold` (higher = more selective)