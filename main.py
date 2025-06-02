import argparse
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from video_processor import VideoProcessor
from object_detector import ObjectDetector
from object_tracker import ObjectTracker
from crop_calculator import CropCalculator
from smoothing import CropWindowSmoother








def parse_args():
    parser = argparse.ArgumentParser(description='Content-aware video cropping')

    # Main args
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--target_ratio', type=float, default=9/16, help='Target aspect ratio (width/height)')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads for parallel processing')

    # Detector args
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'ssd', 'faster_rcnn'], help='Object detection model to use')
    parser.add_argument('--skip_frames', type=int, default=10, help='Process every nth frame for detection (1 = process all frames)')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--object_classes', type=int, nargs='+', default=[0], help='List of object class IDs to track. E.g., 0 1 4 (Default is [0] = person)')

    # Tracker args
    parser.add_argument('--track_count', type=int, default=1, help='Number of objects to track in frame (Default 1)')

    # Crop calculator args
    parser.add_argument('--padding_ratio', type=float, default=0.1, help='Padding ratio for crop window (Default 0.1)')
    parser.add_argument('--size_weight', type=float, default=0.4, help='Weight for object size in crop calculation (Default 0.4)')
    parser.add_argument('--center_weight', type=float, default=0.3, help='Weight for object center in crop calculation (Default 0.3)')
    parser.add_argument('--motion_weight', type=float, default=0.3, help='Weight for object motion in crop calculation (Default 0.3)')
    parser.add_argument('--history_weight', type=float, default=0.1, help='Weight for object history in crop calculation (Default 0.1)')
    parser.add_argument('--saliency_weight', type=float, default=0.4, help='Weight for saliency in crop calculation (Default 0.4)')
    parser.add_argument('--face_detection', action='store_true', default=False, help='Enable face detection for crop calculation')
    parser.add_argument('--weighted_center', action='store_true', default=False, help='Enable weighted center calculation for crop window')
    parser.add_argument('--blend_saliency', action='store_true', default=False, help='Enable blending of saliency map with detected objects for crop calculation')

    # Smoother args
    parser.add_argument('--apply_smoothing', action='store_true', default=False, help='Enable temporal smoothing for crop windows')
    parser.add_argument('--smoothing_window', type=int, default=30,  help='Number of frames for temporal smoothing')
    parser.add_argument('--position_inertia', type=float, default=0.8, help='Position inertia for smoothing (Default 0.8)')
    parser.add_argument('--size_inertia', type=float, default=0.9, help='Size inertia for smoothing (Default 0.9)')

    # debugging
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debuging mode for CLI. See debug_logs folder.')


    return parser.parse_args()





def process_keyframe(frame_idx, frame, detector, tracker, tracked_objects_by_frame, track_count=1):
    """Process a keyframe with detection and tracking."""
    # print(f"🐤 main : process_keyframe: {frame_idx}")

    # Detect objects in frame
    detected_objects = detector.detect(
        frame,              # 🕵️‍♂️ Detect objects in the frame
        top_n=track_count,  # 🏷️ Get the top N detection (highest confidence) 
    )
    
    # Update tracker with new detections
    tracked_objects = tracker.update(frame, detected_objects)
    tracked_objects_by_frame[frame_idx] = tracked_objects
    
    return frame_idx





def main(args=None):

    if args is None:
        args = parse_args()
    
    # Initialize components with YOLOv8
    video_processor = VideoProcessor()


    detector = ObjectDetector(
        confidence_threshold=args.conf_threshold,   # 🕵️‍♂️ Confidence threshold for object detection (0-1).
        model_size=args.model_size,                # 📏 Size of the YOLOv8 model (n=small, s=medium, m=large, l=xlarge).
        classes=args.object_classes,               # 🏷️ Classes to detect (0=person, 1=bicycle, 4=car, 7=truck, etc...).
        debug=args.debug,                               # 🐛 If True, saves debug images and logs to help you visualize decisions.
    )

    tracker = ObjectTracker(
        max_disappeared=30,     # 🕵️‍♂️ Number of frames an object can be missing before being considered lost.
        max_distance=50         # 🔍 Maximum distance for re-identifying lost objects (in pixels).
    )


    crop_calculator = CropCalculator(
        target_ratio=args.target_ratio,         # 📐 Desired aspect ratio of the crop (e.g., 16/9 for widescreen). Example: 1.77
        padding_ratio=args.padding_ratio,       # ➡️ Add 10% padding around the detected area to avoid tight crops.
        size_weight=args.size_weight,           # 📏 How much object size matters (larger objects are more important).
        center_weight=args.center_weight,       # 🎯 How much being close to the frame center matters (centered objects preferred).
        motion_weight=args.motion_weight,       # 🎥 How much moving objects are prioritized (good for tracking action).
        history_weight=args.history_weight,     # 🕰️ How much previous frames affect the crop (smoothness over time). Set 0 to ignore history.
        saliency_weight=args.saliency_weight,   # 👀 How much visual "importance" (saliency maps) matters (e.g., bright or attention-grabbing regions).
        debug=args.debug,                       # 🐛 If True, saves debug images and logs to help you visualize decisions.
        face_detection=args.face_detection,     # 👤 If True, uses face to enhance detection in the crop. Uses weighted averages.
        weighted_center=args.weighted_center,   # ⚖️ If True, uses weighted average of detected objects' centers for crop center.
        blend_saliency=args.blend_saliency,     # 🌈 If True, blends saliency map with detected objects to enhance crop.
    )


    smoother = CropWindowSmoother(
        window_size=args.smoothing_window,      # 📅 Number of frames for smoothing (e.g., 30 for 1 second at 30 FPS).
        position_inertia=args.position_inertia, # 🔄 How much the position of the crop should "stick" to the previous frame (0-1).
        size_inertia=args.size_inertia          # 📏 How much the size of the crop should "stick" to the previous frame (0-1).
    )
    
    # Load video and get properties
    video_info = video_processor.load_video(args.input)
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    print(f"Processing video: {args.input}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    # Process frames
    tracked_objects_by_frame = {}
    
    start_time = time.time()
    






    # First pass: detect and track objects on keyframes only
    print("Phase 1: Detecting and tracking objects...")
    
    # Determine keyframes
    keyframes = list(range(0, total_frames, args.skip_frames))
    
    # Use ThreadPoolExecutor for parallel processing of keyframes
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a list to store futures
        futures = []
        
        # Process keyframes
        for frame_idx in keyframes:
            # Set position to keyframe
            video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_processor.cap.read()
            
            if not ret:
                continue
            
            # Submit task to executor
            future = executor.submit(
                process_keyframe, 
                frame_idx, 
                frame, 
                detector, 
                tracker, 
                tracked_objects_by_frame,
                args.track_count
            )
            futures.append(future)
            
            # Print progress every 10 keyframes
            if len(futures) % 10 == 0:
                print(f"\r🤖 Submitted {len(futures)}/{len(keyframes)} keyframes for processing", end='', flush=True)
        
        print(f"\n")

        # Wait for all futures to complete
        for i, future in enumerate(futures):
            future.result()  # This will raise any exceptions that occurred
            if (i + 1) % 10 == 0:
                print(f"\r🚀 Processed {i + 1}/{len(keyframes)} keyframes", end='', flush=True)
    









    # Second pass: calculate crop windows for keyframes
    print("Phase 2: Calculating crop windows for keyframes...")
    
    # Pre-allocate crop windows array
    crop_windows = [None] * total_frames
    
    # Process keyframes
    for frame_idx in keyframes:
        if frame_idx not in tracked_objects_by_frame:
            continue
            
        objects = tracked_objects_by_frame[frame_idx]
        
        # Get the actual frame for additional analysis
        video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_processor.cap.read()
        
        if not ret:
            continue
        
        # Calculate optimal crop window
        crop_window = crop_calculator.calculate(objects, width, height, frame)
        crop_windows[frame_idx] = crop_window
        
        if frame_idx % 100 == 0:
            print(f"\r✂️ Calculated crop window for keyframe {frame_idx}/{total_frames}", end='', flush=True)
    










    # Phase 3: Interpolate crop windows for non-keyframes
    print("Phase 3: Interpolating crop windows for non-keyframes...")
    
    # Fast interpolation using numpy
    keyframe_indices = np.array(keyframes)
    keyframe_crop_windows = np.array([crop_windows[i] for i in keyframes if crop_windows[i] is not None])
    
    if len(keyframe_crop_windows) > 1:
        # For each frame, find the nearest keyframes and interpolate
        for i in range(total_frames):
            if crop_windows[i] is not None:
                continue
                
            # Find nearest keyframes
            next_idx = keyframe_indices[keyframe_indices > i]
            prev_idx = keyframe_indices[keyframe_indices < i]
            
            if len(next_idx) == 0 and len(prev_idx) > 0:
                # After last keyframe, use last keyframe
                crop_windows[i] = crop_windows[prev_idx[-1]]
            elif len(prev_idx) == 0 and len(next_idx) > 0:
                # Before first keyframe, use first keyframe
                crop_windows[i] = crop_windows[next_idx[0]]
            elif len(prev_idx) > 0 and len(next_idx) > 0:
                # Interpolate between keyframes
                prev_frame = prev_idx[-1]
                next_frame = next_idx[0]
                
                if crop_windows[prev_frame] is not None and crop_windows[next_frame] is not None:
                    # Calculate interpolation factor
                    alpha = (i - prev_frame) / (next_frame - prev_frame)
                    
                    # Linear interpolation
                    prev_crop = np.array(crop_windows[prev_frame])
                    next_crop = np.array(crop_windows[next_frame])
                    interp_crop = prev_crop * (1 - alpha) + next_crop * alpha
                    crop_windows[i] = [int(x) for x in interp_crop]
    
    # Fill any remaining None values with center crop
    for i in range(total_frames):
        if crop_windows[i] is None:
            # Use center crop as fallback
            crop_height = height
            crop_width = int(crop_height * args.target_ratio)
            if crop_width > width:
                crop_width = width
                crop_height = int(crop_width / args.target_ratio)
            x = int((width - crop_width) / 2)
            y = int((height - crop_height) / 2)
            crop_windows[i] = [x, y, crop_width, crop_height]
    










    # Apply temporal smoothing to crop windows
    print("Phase 4: Applying temporal smoothing..." if args.apply_smoothing else "Phase 4: No smoothing applied")

    if args.apply_smoothing:
        smoothed_windows = smoother.smooth(crop_windows)
    else:
        smoothed_windows = crop_windows

    










    # Generate output video with cropped frames
    print("Phase 5: Generating output video...")
    video_processor.generate_output_video(
        output_path=args.output,
        crop_windows=smoothed_windows,
        fps=fps
    )
    
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time:.2f} seconds")
    print(f"Output saved to: {args.output}")





if __name__ in {"__main__", "__mp_main__"}:
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        start_gui()