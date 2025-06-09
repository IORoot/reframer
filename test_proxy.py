#!/usr/bin/env python3
"""
Test script for proxy video functionality
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor

def test_proxy_creation():
    """Test proxy video creation with different settings."""
    
    # Test video path (you'll need to provide a real video file)
    test_video = "test_video.mp4"  # Replace with your test video
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please provide a test video file to test proxy functionality.")
        return
    
    print("🧪 Testing proxy video functionality...")
    
    # Initialize video processor
    processor = VideoProcessor()
    
    # Load original video
    print(f"\n📹 Loading original video: {test_video}")
    video_info = processor.load_video(test_video)
    print(f"Original resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS: {video_info['fps']}, Total frames: {video_info['total_frames']}")
    
    # Test different proxy resolutions
    resolutions = ['360p', '480p', '720p', '25%', '50%']
    qualities = ['low', 'medium', 'high']
    
    for resolution in resolutions:
        for quality in qualities:
            print(f"\n🎬 Testing proxy: {resolution} resolution, {quality} quality")
            
            try:
                proxy_info = processor.create_proxy_video(
                    proxy_resolution=resolution,
                    proxy_quality=quality
                )
                
                if proxy_info:
                    print(f"✅ Proxy created: {proxy_info['path']}")
                    print(f"   Resolution: {proxy_info['width']}x{proxy_info['height']}")
                    print(f"   Scale factors: {proxy_info['scale_factor_x']:.2f}x, {proxy_info['scale_factor_y']:.2f}x")
                    
                    # Test coordinate scaling
                    test_crop = [100, 100, 200, 200]  # Example crop window
                    scaled_crop = processor.scale_coordinates_to_original(test_crop)
                    print(f"   Test crop {test_crop} -> scaled {scaled_crop}")
                    
                    # Cleanup
                    processor.cleanup_proxy()
                else:
                    print("❌ Failed to create proxy")
                    
            except Exception as e:
                print(f"❌ Error: {e}")

def test_coordinate_scaling():
    """Test coordinate scaling functionality."""
    print("\n🧮 Testing coordinate scaling...")
    
    processor = VideoProcessor()
    
    # Simulate proxy info
    processor.proxy_info = {
        'scale_factor_x': 2.0,
        'scale_factor_y': 2.0
    }
    
    test_crops = [
        [100, 100, 200, 200],
        [50, 50, 100, 100],
        [0, 0, 640, 360]
    ]
    
    for crop in test_crops:
        scaled = processor.scale_coordinates_to_original(crop)
        print(f"Crop {crop} -> Scaled {scaled}")

if __name__ == "__main__":
    print("🚀 Proxy Video Test Suite")
    print("=" * 50)
    
    test_coordinate_scaling()
    
    # Uncomment to test actual proxy creation (requires test video)
    # test_proxy_creation()
    
    print("\n✅ Test completed!") 