# Content-Aware Video Cropping

## Approach & Design Decisions

### Core Approach
1. **Object Detection & Tracking**
   - Using YOLOv8 for reliable object detection
   - Multiple model sizes available (n, s, m, l, x) to balance speed vs accuracy
   - Implemented object tracking to maintain subject consistency across frames

2. **Intelligent Crop Window Selection**
   - Prioritizes subjects based on importance (people > faces > animals > objects)
   - Uses weighted scoring system considering:
     - Subject type (class weight)
     - Size of subject
     - Position in frame
     - Detection confidence

3. **Performance Optimization**
   - Processes keyframes instead of every frame
   - Uses parallel processing for faster analysis
   - Interpolates crop windows between keyframes
   - Implements temporal smoothing for stable transitions

### Key Design Decisions

1. **YOLOv8 Selection**
   - Chosen for balance of speed and accuracy
   - Provides good detection even for small objects
   - Multiple model sizes allow flexibility based on needs

2. **Keyframe Processing**
   - Process every Nth frame instead of all frames
   - Reduces computational load significantly
   - Interpolate between keyframes for smooth transitions

3. **Error Handling**
   - Fallback mechanisms for saliency detection
   - Local model loading with remote fallback
   - Graceful handling of missing frames or detection failures

## Setup & Running Instructions

### 1. System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU recommended (but not required)
- 8GB RAM minimum (16GB recommended)

### 2. Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install ultralytics opencv-python numpy tqdm
```

### 3. Download Models
```bash
# Download YOLOv8 models
python download_models.py
```

### 4. Running the Code

Basic usage:
```bash
python main.py --input <input_video> --output <output_video>
```

Advanced usage:
```bash
python main.py --input input.mp4 --output output.mp4 \
    --model_size m \          # Model size (n/s/m/l/x)
    --skip_frames 5 \         # Process every 5th frame
    --conf_threshold 0.6 \    # Detection confidence threshold
    --max_workers 4          # Number of parallel workers
```

### 5. Common Command Arguments

| Argument | Purpose | Default | Options |
|----------|---------|---------|----------|
| --input | Input video path | Required | Path to video |
| --output | Output video path | Required | Path to save |
| --model_size | YOLOv8 model size | 'n' | 'n','s','m','l','x' |
| --skip_frames | Frame skip rate | 10 | Any positive int |
| --conf_threshold | Detection confidence | 0.5 | 0.0 to 1.0 |
| --max_workers | Parallel workers | 4 | Any positive int |

### 6. Optimization Tips

For Speed:
- Use smaller model size (`--model_size n`)
- Increase frame skip (`--skip_frames 15`)
- Adjust workers based on CPU (`--max_workers`)

For Quality:
- Use larger model (`--model_size m` or `l`)
- Decrease frame skip (`--skip_frames 5`)
- Increase confidence (`--conf_threshold 0.6`)

### 7. Troubleshooting

If you encounter:
- **Memory issues**: Reduce `max_workers` or use smaller model
- **Slow processing**: Increase `skip_frames` or use smaller model
- **Poor detection**: Use larger model size or decrease confidence threshold
- **Unstable output**: Increase smoothing window size 