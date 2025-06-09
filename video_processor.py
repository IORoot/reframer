import cv2
import numpy as np
import os
import subprocess

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.writer = None
        self.frames = None
        self.current_frame_idx = 0
        self.video_info = {}
        self.proxy_info = {}
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
    
    def load_video(self, video_path):
        """Load video file and extract basic information."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        # self.cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        self.video_info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'path': video_path
        }
        
        self.current_frame_idx = 0
        return self.video_info
    
    def create_proxy_video(self, proxy_resolution='720p', proxy_quality='medium'):
        """Create a proxy video for faster processing."""
        if self.cap is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        # Calculate proxy dimensions
        original_width = self.video_info['width']
        original_height = self.video_info['height']
        
        if proxy_resolution == '360p':
            target_width, target_height = 640, 360
        elif proxy_resolution == '480p':
            target_width, target_height = 854, 480
        elif proxy_resolution == '720p':
            target_width, target_height = 1280, 720
        elif proxy_resolution == '1080p':
            target_width, target_height = 1920, 1080
        elif proxy_resolution == '25%':
            target_width = int(original_width * 0.25)
            target_height = int(original_height * 0.25)
        elif proxy_resolution == '50%':
            target_width = int(original_width * 0.5)
            target_height = int(original_height * 0.5)
        else:
            raise ValueError(f"Unknown proxy resolution: {proxy_resolution}")
        
        # Maintain aspect ratio
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        if original_ratio > target_ratio:
            # Original is wider, fit to width
            proxy_width = target_width
            proxy_height = int(target_width / original_ratio)
        else:
            # Original is taller, fit to height
            proxy_height = target_height
            proxy_width = int(target_height * original_ratio)
        
        # Ensure even dimensions for video encoding
        proxy_width = proxy_width - (proxy_width % 2)
        proxy_height = proxy_height - (proxy_height % 2)
        
        # Calculate scale factors for coordinate conversion
        self.scale_factor_x = original_width / proxy_width
        self.scale_factor_y = original_height / proxy_height
        
        # Set quality parameters
        if proxy_quality == 'low':
            crf = 35
            preset = 'ultrafast'
        elif proxy_quality == 'medium':
            crf = 28
            preset = 'fast'
        elif proxy_quality == 'high':
            crf = 23
            preset = 'medium'
        else:
            raise ValueError(f"Unknown proxy quality: {proxy_quality}")
        
        # Create proxy file path
        input_path = self.video_info['path']
        proxy_path = input_path.rsplit('.', 1)[0] + '_proxy.mp4'
        
        # Create proxy using ffmpeg
        ffmpeg_command = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f'scale={proxy_width}:{proxy_height}',
            '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
            '-an',  # No audio for proxy
            proxy_path
        ]
        
        print(f"Creating proxy video: {proxy_path}")
        print(f"Proxy resolution: {proxy_width}x{proxy_height} (scale factors: {self.scale_factor_x:.2f}x, {self.scale_factor_y:.2f}x)")
        
        try:
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            print("✅ Proxy video created successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create proxy video: {e.stderr}")
            return None
        
        # Store proxy information
        self.proxy_info = {
            'path': proxy_path,
            'width': proxy_width,
            'height': proxy_height,
            'scale_factor_x': self.scale_factor_x,
            'scale_factor_y': self.scale_factor_y
        }
        
        return self.proxy_info
    
    def switch_to_proxy_video(self):
        """Switch video capture to use the proxy video for detection and tracking."""
        if not self.proxy_info or not os.path.exists(self.proxy_info['path']):
            raise ValueError("No proxy video available. Call create_proxy_video() first.")
        
        # Store original video info
        self.original_video_info = self.video_info.copy()
        
        # Close current video capture
        if self.cap:
            self.cap.release()
        
        # Open proxy video
        self.cap = cv2.VideoCapture(self.proxy_info['path'])
        
        # Verify the video was opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open proxy video: {self.proxy_info['path']}")
        
        # Update video info to proxy dimensions
        self.video_info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'path': self.proxy_info['path']
        }
        
        print(f"🔄 Switched to proxy video: {self.proxy_info['path']}")
        print(f"📊 Proxy dimensions: {self.video_info['width']}x{self.video_info['height']}")
        print(f"📊 Proxy frames: {self.video_info['total_frames']}")
        
        return self.video_info
    
    def switch_to_original_video(self):
        """Switch video capture back to the original video for final processing."""
        if not hasattr(self, 'original_video_info'):
            raise ValueError("No original video info available.")
        
        # Close current video capture
        if self.cap:
            self.cap.release()
        
        # Open original video
        self.cap = cv2.VideoCapture(self.original_video_info['path'])
        
        # Verify the video was opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open original video: {self.original_video_info['path']}")
        
        # Restore original video info
        self.video_info = self.original_video_info.copy()
        
        print(f"🔄 Switched back to original video: {self.video_info['path']}")
        print(f"📊 Original dimensions: {self.video_info['width']}x{self.video_info['height']}")
        print(f"📊 Original frames: {self.video_info['total_frames']}")
        
        return self.video_info
    
    def scale_coordinates_to_original(self, crop_window):
        """Scale crop coordinates from proxy back to original video dimensions."""
        if not self.proxy_info:
            return crop_window  # No scaling if no proxy was used
        
        x, y, w, h = crop_window
        
        # Scale coordinates and dimensions
        scaled_x = int(x * self.scale_factor_x)
        scaled_y = int(y * self.scale_factor_y)
        scaled_w = int(w * self.scale_factor_x)
        scaled_h = int(h * self.scale_factor_y)
        
        return [scaled_x, scaled_y, scaled_w, scaled_h]
    
    def cleanup_proxy(self):
        """Remove proxy file if it exists."""
        if self.proxy_info and os.path.exists(self.proxy_info['path']):
            try:
                os.remove(self.proxy_info['path'])
                print(f"🗑️ Removed proxy file: {self.proxy_info['path']}")
            except OSError as e:
                print(f"⚠️ Warning: Could not remove proxy file: {e}")
    
    def frame_generator(self):
        """Generator that yields frames from the video."""
        if self.cap is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame
            self.current_frame_idx += 1
    
    def apply_crop(self, frame, crop_window):
        """Apply crop window to a frame."""
        x, y, w, h = crop_window
        
        # Ensure crop window is within frame boundaries
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))
        
        # Apply crop
        return frame[y:y+h, x:x+w]
    
    def convert_to_h264(self, input_path):
        """Convert the given video to H.264 format using FFmpeg."""

        temp_h264_output = "temp_h264_output.mp4"

        ffmpeg_command = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            temp_h264_output
        ]
        
        print(f"Encoding video to H.264: {temp_h264_output}")
        subprocess.run(ffmpeg_command, check=True)

        # Overwrite original video with the new one
        os.replace(temp_h264_output, input_path)


    def merge_audio(self, video_path, original_video):
        """Merge original audio into the processed video using FFmpeg."""
        audio_path = "temp_audio.mp3"
        temp_output = "temp_output.mp4"  # Temporary output file

        # Extract audio from the original video
        extract_cmd = [
            "ffmpeg", "-i", original_video, "-q:a", "0", "-map", "0:a?", audio_path, "-y"
        ]
        result = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not os.path.exists(audio_path):
            print("Error extracting audio:", result.stderr.decode())
            return  # Exit if audio extraction fails

        # Merge extracted audio into a temporary file
        merge_cmd = [
            "ffmpeg", "-i", video_path, "-i", audio_path, 
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", temp_output, "-y"
        ]
        result = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print("Error merging audio:", result.stderr.decode())
            return  # Exit if merging fails

        # Overwrite original video with the new one
        os.replace(temp_output, video_path)

        # Cleanup
        os.remove(audio_path)

    
    
    def generate_output_video(self, output_path, crop_windows, fps=None):
        """Generate output video with the specified crop windows."""
        if self.cap is None:
            raise ValueError("No video loaded. Call load_video() first.")
        
        if fps is None:
            fps = self.video_info['fps']
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get first crop window to determine output dimensions
        first_crop = crop_windows[0]
        crop_width, crop_height = first_crop[2], first_crop[3]
        
        # Create video writer with mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (crop_width, crop_height)
        )
        
        # Check if writer was successfully created
        if not self.writer.isOpened():
            print(f"Error: Could not create video writer for {output_path}")
            print(f"Trying alternative codec...")
            # Try with a different codec
            self.writer = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'XVID'), 
                fps, 
                (crop_width, crop_height)
            )
            if not self.writer.isOpened():
                raise ValueError(f"Failed to create video writer with multiple codecs")
        
        # Process each frame
        for i, frame in enumerate(self.frame_generator()):
            if i >= len(crop_windows):
                break
            
            # Apply crop with boundary checking
            crop_window = crop_windows[i]
            try:
                cropped_frame = self.apply_crop(frame, crop_window)
                
                # Ensure cropped frame has the expected dimensions
                if cropped_frame.shape[1] != crop_width or cropped_frame.shape[0] != crop_height:
                    cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
                    
                # Write to output video
                self.writer.write(cropped_frame)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue
            
            if i % 100 == 0:
                print(f"\r🎥 Processed {i}/{len(crop_windows)} frames for output video", end='', flush=True)
        
        # Release resources
        if self.writer:
            self.writer.release()

        # Merge audio
        self.merge_audio(output_path, self.video_info['path'])

        self.convert_to_h264(output_path)

    
    def __del__(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release() 