# Content-Aware Video Cropping

An intelligent system that automatically converts landscape videos to portrait format (9:16) while keeping the most important subjects in frame.

![Project Banner](video_cropped.png)

## Overview

This project implements a content-aware video cropping system that uses computer vision and machine learning to:
- Detect and track subjects of interest in videos
- Prioritize subjects based on importance (people, faces, etc.)
- Calculate optimal crop windows that follow the action
- Generate smooth transitions between frames
- Output portrait-oriented videos optimized for mobile viewing

## Features

- **Intelligent Subject Detection**: Uses YOLOv8 to identify people, animals, vehicles, and other objects
- **Face Prioritization**: Specifically detects and prioritizes human faces
- **Saliency Detection**: Identifies visually interesting regions when subjects are few or missing
- **Multi-Subject Handling**: Intelligently decides which subjects to focus on when multiple are present
- **Smooth Tracking**: Maintains stable framing with temporal smoothing
- **Parallel Processing**: Utilizes multi-threading for faster keyframe analysis
- **Configurable Parameters**: Adjust model size, processing speed, and smoothing for different needs
- **Progress Reporting**: Provides detailed progress updates during processing

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU recommended for faster processing

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/content-aware-video-cropping.git
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

## Project Evolution

### Initial Approach
We started with a basic implementation using YOLOv3 for object detection and a simple tracking mechanism. This approach had limitations:
- Slower detection speed
- Less accurate object identification
- Basic tracking that struggled with multiple subjects
- Simple crop window calculation without sophisticated prioritization

### Improvements and Iterations
1. **Upgraded to YOLOv8**: 
   - Significantly improved detection accuracy
   - Better performance, especially for small objects
   - More robust in challenging lighting conditions

2. **Enhanced Subject Prioritization**:
   - Added dedicated face detection using Haar Cascades
   - Implemented weighted importance calculation based on class, size, and position
   - Added historical weighting for stability

3. **Improved Tracking**:
   - Implemented IoU-based tracking with Hungarian algorithm
   - Added handling for subjects that temporarily disappear

4. **Optimized Processing Pipeline**:
   - Added keyframe processing with interpolation
   - Implemented phased processing with progress updates
   - Added robust error handling

5. **Enhanced Smoothing**:
   - Implemented multi-factor temporal smoothing
   - Added inertia to resist sudden changes
   - Balanced responsiveness with stability

6. **Latest Improvements**:
   - Added saliency detection to identify visually interesting regions
   - Implemented parallel processing with ThreadPoolExecutor
   - Enhanced importance calculation with higher weights for class and size
   - Optimized interpolation using numpy for faster processing


## Project Structure

```
content-aware-video-cropping/
│
├── main.py                  # Entry point for the application
├── video_processor.py       # Video input/output operations
├── object_detector.py       # Object detection using YOLOv8
├── object_tracker.py        # Tracking objects across frames
├── crop_calculator.py       # Calculate optimal crop window
├── smoothing.py             # Temporal smoothing for crop windows
├── utils.py                 # Utility functions
└── requirements.txt         # Dependencies
```

## Usage

### Basic Usage

```bash
python main.py --input "path/to/input/video.mp4" --output "path/to/output/video.mp4"
```

### Advanced Options

```bash
python main.py --input "path/to/input/video.mp4" --output "path/to/output/video.mp4" --model_size m --skip_frames 10 --smoothing_window 30 --conf_threshold 0.5 --use_saliency --max_workers 4
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Path to input video file | Required |
| `--output` | Path to output video file | Required |
| `--target_ratio` | Target aspect ratio (width/height) | 0.5625 (9:16) |
| `--model_size` | YOLOv8 model size (n, s, m, l, x) | n (nano) |
| `--skip_frames` | Process every Nth frame for detection | 10 |
| `--smoothing_window` | Number of frames for temporal smoothing | 30 |
| `--conf_threshold` | Confidence threshold for detections | 0.5 |
| `--use_saliency` | Enable saliency detection | False |
| `--max_workers` | Maximum number of worker threads | 4 |

### Model Size Selection Guide

| Model Size | Flag | Description | Use Case |
|------------|------|-------------|----------|
| Nano (n) | `--model_size n` | Smallest and fastest model | Testing or low-power devices |
| Small (s) | `--model_size s` | Good balance for mobile devices | Mobile applications |
| Medium (m) | `--model_size m` | Balanced model | General purpose detection |
| Large (l) | `--model_size l` | Higher accuracy, slower speed | When accuracy is more important |
| XLarge (x) | `--model_size x` | Highest accuracy, slowest speed | When maximum accuracy is required |

### Processing Multiple Videos

To process multiple videos in a directory, you can use a simple shell script:

```bash
#!/bin/bash
INPUT_DIR="/path/to/videos"
OUTPUT_DIR="/path/to/outputs"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each MP4 file in the input directory
for video in "$INPUT_DIR"/*.mp4; do
    # Get just the filename without path
    filename=$(basename "$video")
    # Run the processing script
    python main.py --input "$video" --output "$OUTPUT_DIR/${filename%.mp4}_cropped.mp4" --model_size m --use_saliency --max_workers 4
    echo "Processed $filename"
done
```

## Architecture

The system follows a modular pipeline architecture:

1. **Video Processing**: Handles video I/O operations
2. **Object Detection**: Identifies subjects using YOLOv8
3. **Saliency Detection**: Identifies visually interesting regions
4. **Object Tracking**: Maintains subject identity across frames
5. **Crop Window Calculation**: Determines optimal crop position
6. **Temporal Smoothing**: Ensures smooth transitions
7. **Video Generation**: Creates the final portrait video

## Algorithm Flow

1. **Keyframe Analysis**:
   - Process every Nth frame to reduce computation
   - Detect objects using YOLOv8
   - Track objects across consecutive keyframes

2. **Subject Prioritization**:
   - Assign importance scores based on class, size, position, and confidence
   - Prioritize human faces and people
   - Use saliency detection when objects are few or missing

3. **Crop Window Calculation**:
   - Calculate weighted center point based on subject importance
   - Blend object-based and saliency-based interest points
   - Apply historical weighting for stability
   - Ensure crop window maintains target aspect ratio

4. **Interpolation and Smoothing**:
   - Interpolate crop windows for non-keyframes
   - Apply temporal smoothing to avoid jerky movements

5. **Video Generation**:
   - Apply calculated crop windows to original frames
   - Generate output video in portrait format

## Technical Implementation Details

### Object Detection
We use YOLOv8 from the Ultralytics package for object detection:
```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')  # or 's', 'm', 'l', 'x' for different sizes

# Run detection
results = model(frame)

# Process results
for det in results.boxes.data.cpu().numpy():
    x1, y1, x2, y2, conf, cls = det
    # Process detection...
```

### Saliency Detection
We use OpenCV's saliency detection to find visually interesting regions:
```python
# Initialize saliency detector
saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

# Calculate saliency map
success, saliency_map = saliency_detector.computeSaliency(frame)

# Find most salient region
_, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
```

We use OpenCV's saliency detection with fallback options:
```python
# Initialize saliency detector with fallback
try:
    saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
except:
    try:
        saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    except:
        print("Warning: Saliency detection not available. Continuing without saliency...")
        saliency_detector = None
```

### Model Loading
YOLOv8 models are loaded with local file priority:
```python
model_path = f'models/yolov8{model_size}.pt'
if not os.path.exists(model_path):
    print(f"Model {model_path} not found, using default model from Ultralytics")
    model_path = f'yolov8{model_size}.pt'
```

### Parallel Processing
We use ThreadPoolExecutor for parallel keyframe processing:
```python
with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = []
    for frame_idx in keyframes:
        future = executor.submit(
            process_keyframe, 
            frame_idx, 
            frame, 
            detector, 
            tracker, 
            tracked_objects_by_frame
        )
        futures.append(future)
```

### Enhanced Importance Calculation
We calculate subject importance with higher weights for class and size:
```python
importance = (
    class_weight * 1.5 *  # Increased class weight influence
    confidence * 
    (size_weight * size_factor * 1.2 + center_weight * center_factor)
)
```

## Performance Considerations

- Processing time depends on video length, resolution, and chosen model size
- Using larger model sizes (l, x) significantly increases processing time but improves detection accuracy
- Increasing `skip_frames` speeds up processing but may reduce tracking accuracy
- Parallel processing with `max_workers` can significantly improve speed on multi-core systems
- GPU acceleration significantly improves performance

## Troubleshooting

### Common Issues and Solutions

1. **Video Writer Error**:
   - Try using a different codec with `cv2.VideoWriter_fourcc(*'XVID')`
   - Ensure the output directory exists
   - Check that the crop dimensions are valid

2. **Slow Processing**:
   - Increase `skip_frames` parameter
   - Use a smaller model size
   - Increase `max_workers` if you have more CPU cores
   - Ensure GPU acceleration is working if available

3. **Poor Tracking Quality**:
   - Use a larger model size for better detection
   - Decrease `skip_frames` for more frequent detection
   - Enable saliency detection with `--use_saliency`
   - Increase `conf_threshold` to filter out low-confidence detections

4. **Memory Issues**:
   - Process shorter video segments
   - Use a smaller model size
   - Reduce video resolution before processing
   - Decrease `max_workers` to reduce memory usage

5. **Saliency Detection Issues**:
   - The system will continue working even if saliency detection is unavailable
   - Different OpenCV versions may support different saliency detection methods
   - Use `--use_saliency` only if your OpenCV installation supports it

## Future Improvements

1. **GPU Acceleration**: Optimize for GPU processing to improve speed
2. **Advanced Tracking**: Implement more sophisticated tracking algorithms (DeepSORT)
3. **Scene Understanding**: Add scene context awareness for better cropping decisions
4. **Subject Re-identification**: Improve handling of subjects that leave and re-enter the frame
5. **Adaptive Processing**: Dynamically adjust processing parameters based on video content

## Demo

![Demo GIF](https://via.placeholder.com/640x360)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for video processing capabilities
- [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) for optimal assignment in tracking
