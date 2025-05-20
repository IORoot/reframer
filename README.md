# AI Video Reframe

Batch convert landscape to portrait videos using AI.

forked from [https://github.com/Sagar-lab03/AI-Content-Aware-Video-Cropping](https://github.com/Sagar-lab03/AI-Content-Aware-Video-Cropping)

---

## Intro

This software will detect motion in the video and crop the content dynamically keeping the main subject in the frame.
There are many settings, but the main ones are:
- Speed vs Quality : You can select how many frames to use AI on. The more you do, the better the tracking but the longer it takes.
- Aspect Ratio : Specify the size of the reframe.
- FFMPEG : Used to keep audio and do custom functions like trimming the footage.

---

## Requirements
1. Python 3
2. [FFMPEG](https://github.com/FFmpeg/FFmpeg)

> This runs linux/macos shell scripts - but it easily can be run in windows if needed. Just read the `run.sh` file and run commands by hand.

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/IORoot/AI_Video_Reframe
```

2. Setup virtual environment
```bash
cd AI_Video_Reframe
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirement.txt
#or individually:
pip install ultralytics opencv-python numpy tqdm
```

## Run a single file

To accept the defaults and convert landscape 16:9 videos to 3:4 videos, do this:
```bash
./run.sh /Location/of/MP4/file.mp4
```

## Changing the settings

The main line to run the code is:
```bash
python main.py --input "${FILENAME}" --output "${PROCESSED_FILENAME}" --model_size m --skip_frames 3 --smoothing_window 30 --conf_threshold 0.5 --use_saliency --max_workers 6 --target_ratio 0.75
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


---

### Scenarios:

I've found that if you want highly accurate (but very slow processing) video reframing, you need to do the following flags:
```bash
--skip_frames 0   # Use AI to detect movement on EVERY Frame
--model_size x    # Use biggest AI Model
--max_workers 8   # Or the number of CPU cores you have
```

Do a good job of tracking, but fast movements might not be caught:
```bash
--skip_frames 3   # Skip 3 frames, and then use AI on 1. Repeat.
--model_size m    # use the medium AI model
--max_workers 6   # 75% of All cores
```

Fast, but inaccurate tracking - good for low movement or interview videos:
```bash
--skip_frames 30  # on a 30fps video, use 1 frame per second.
--model_size s    # use a small AI Model
--max_workers 6   # 75% of cores
```

There are a lot more settings and the python code can be changed to make even more alterations too.

---

## Batch Runs

Use like so:

```bash
./run_batch.sh /folder/with/videos/in/ 
```

This will find all `mp4` files in any subdirectory within that folder. It will then create a new file
called `run_all_found_files.sh` which lists every file and the run command against each one. 

The reason that I prefer using this method rather than just a loop over each file and running it is
because you can open up the `run_all_found_files.sh` file and check how far the batch has got through.
It also allow you to cancel the process at any time and then start again (it will skip any already done)
without a problem.

Once all videos are converted, the `run_all_found_files.sh` file is removed.


## Output

The reframed videos will be in a subfolder within the directory of the found video file.

FFMPEG is used to copy the audio from the original to the reframed version since the main
python code does not do that.


## Credit

This code was originally from [https://github.com/Sagar-lab03/AI-Content-Aware-Video-Cropping](https://github.com/Sagar-lab03/AI-Content-Aware-Video-Cropping) and all the AI work is theirs.
I've slightly adapted it to include the `bash` scripts and FFMPEG bits for my own usage.
