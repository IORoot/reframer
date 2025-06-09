# Proxy Video Processing Guide

## Overview

The proxy video feature allows you to significantly speed up the detection and tracking phase by processing a lower-resolution version of your video, then applying the results to the original high-quality video.

## How It Works

1. **Proxy Creation**: A lower-resolution version of your video is created using FFmpeg
2. **Fast Processing**: Object detection and tracking run on the smaller proxy video
3. **Coordinate Scaling**: Crop coordinates are scaled back to the original video dimensions
4. **High-Quality Output**: The final video is cropped from the original high-quality source
5. **Cleanup**: Proxy file is automatically removed (unless `--keep_proxy` is specified)

## Performance Benefits

| Proxy Resolution | Speed Improvement | File Size Reduction | Use Case |
|------------------|-------------------|---------------------|----------|
| 360p | ~16x faster | ~95% smaller | Very fast processing, may miss small objects |
| 480p | ~9x faster | ~90% smaller | Good balance for most videos |
| 720p | ~4x faster | ~75% smaller | Recommended default |
| 1080p | ~1.5x faster | ~50% smaller | Minimal speed gain, high accuracy |
| 25% | Variable | ~94% smaller | Percentage of original |
| 50% | Variable | ~75% smaller | Percentage of original |

## Command Line Options

### Basic Proxy Usage
```bash
python main.py --input video.mp4 --output cropped.mp4 --use_proxy
```

### Advanced Proxy Options
```bash
python main.py \
  --input video.mp4 \
  --output cropped.mp4 \
  --use_proxy \
  --proxy_resolution 480p \
  --proxy_quality medium \
  --keep_proxy
```

### Proxy Arguments

- `--use_proxy`: Enable proxy video processing
- `--proxy_resolution`: Choose proxy resolution
  - `360p`: 640x360 (very fast)
  - `480p`: 854x480 (fast)
  - `720p`: 1280x720 (recommended)
  - `1080p`: 1920x1080 (minimal speed gain)
  - `25%`: 25% of original dimensions
  - `50%`: 50% of original dimensions
- `--proxy_quality`: Choose proxy quality
  - `low`: Fastest creation, smallest file
  - `medium`: Balanced (recommended)
  - `high`: Slower creation, better quality
- `--keep_proxy`: Keep proxy file after processing (default: auto-remove)

## Usage Examples

### Quick Processing (Fastest)
```bash
python main.py \
  --input long_video.mp4 \
  --output cropped.mp4 \
  --use_proxy \
  --proxy_resolution 360p \
  --proxy_quality low
```

### Balanced Processing (Recommended)
```bash
python main.py \
  --input video.mp4 \
  --output cropped.mp4 \
  --use_proxy \
  --proxy_resolution 720p \
  --proxy_quality medium
```

### High Accuracy Processing
```bash
python main.py \
  --input video.mp4 \
  --output cropped.mp4 \
  --use_proxy \
  --proxy_resolution 1080p \
  --proxy_quality high
```

### Custom Percentage Resolution
```bash
python main.py \
  --input video.mp4 \
  --output cropped.mp4 \
  --use_proxy \
  --proxy_resolution 25% \
  --proxy_quality medium
```

## Technical Details

### Coordinate Scaling
The system automatically calculates scale factors between proxy and original video:
- `scale_factor_x = original_width / proxy_width`
- `scale_factor_y = original_height / proxy_height`

Crop coordinates are scaled using:
- `scaled_x = proxy_x * scale_factor_x`
- `scaled_y = proxy_y * scale_factor_y`
- `scaled_width = proxy_width * scale_factor_x`
- `scaled_height = proxy_height * scale_factor_y`

### Aspect Ratio Preservation
Proxy videos maintain the original aspect ratio to ensure accurate coordinate scaling.

### Fallback Behavior
If proxy creation fails, the system automatically falls back to processing the original video with a warning message.

## File Management

### Proxy File Location
Proxy files are created in the same directory as the input video with `_proxy.mp4` suffix:
- Input: `video.mp4`
- Proxy: `video_proxy.mp4`

### Automatic Cleanup
By default, proxy files are automatically removed after processing. Use `--keep_proxy` to retain them for debugging or reuse.

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and in your PATH
2. **Proxy creation fails**: Check available disk space and FFmpeg installation
3. **Poor detection quality**: Try higher proxy resolution or quality
4. **Slow proxy creation**: Use lower quality setting or smaller resolution

### Debug Mode
Enable debug mode to see detailed information about proxy creation and coordinate scaling:
```bash
python main.py --input video.mp4 --output cropped.mp4 --use_proxy --debug
```

## Best Practices

1. **Start with 720p**: Good balance of speed and accuracy
2. **Use medium quality**: Fast creation with good detection quality
3. **Test with small videos first**: Verify settings work for your use case
4. **Monitor disk space**: Proxy files can be large for long videos
5. **Keep proxy for debugging**: Use `--keep_proxy` if you need to troubleshoot

## Performance Tips

- **Long videos**: Use 360p or 480p for maximum speed
- **High-resolution videos**: Use percentage-based resolutions (25%, 50%)
- **Real-time processing**: Use low quality and small resolution
- **Batch processing**: Keep proxy files if processing multiple videos with same settings 