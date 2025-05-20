import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_detections(frame, objects, show_labels=True):
    """
    Visualize detected objects on a frame.
    
    Args:
        frame: The input frame
        objects: List of detected objects
        show_labels: Whether to show class labels and confidence scorles
    
    Returns:
        Frame with visualized detections
    """
    vis_frame = frame.copy()
    
    # Define colors for different classes
    colors = {
        'person': (0, 255, 0),    # Green
        'face': (0, 0, 255),      # Red
        'car': (255, 0, 0),       # Blue
        'dog': (255, 255, 0),     # Cyan
        'cat': (255, 0, 255),     # Magenta
        'default': (128, 128, 128) # Gray
    }
    
    for obj in objects:
        # Get object properties
        box = obj['box']
        class_name = obj.get('class_name', 'unknown')
        confidence = obj.get('confidence', None)
        importance = obj.get('importance', None)
        
        # Get color for this class
        color = colors.get(class_name, colors['default'])
        
        # Draw bounding box
        x, y, w, h = box
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label if requested
        if show_labels:
            label = class_name
            if confidence is not None:
                label += f" {confidence:.2f}"
            if importance is not None:
                label += f" (imp: {importance:.2f})"
                
            # Draw label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(y, label_size[1])
            cv2.rectangle(vis_frame, (x, y_label - label_size[1]), 
                         (x + label_size[0], y_label + baseline), (255, 255, 255), cv2.FILLED)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x, y_label), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return vis_frame

def visualize_crop_window(frame, crop_window, color=(0, 255, 255)):
    """
    Visualize crop window on a frame.
    
    Args:
        frame: The input frame
        crop_window: Crop window as [x, y, width, height]
        color: Color for the crop window rectangle
    
    Returns:
        Frame with visualized crop window
    """
    vis_frame = frame.copy()
    
    # Draw crop window
    x, y, w, h = crop_window
    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw center point
    center_x = x + w // 2
    center_y = y + h // 2
    cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
    
    return vis_frame

def create_debug_visualization(frame, objects, crop_window, output_path=None):
    """
    Create a debug visualization showing detections and crop window.
    
    Args:
        frame: The input frame
        objects: List of detected objects
        crop_window: Crop window as [x, y, width, height]
        output_path: Path to save the visualization (if None, just returns the image)
    
    Returns:
        Visualization image
    """
    # Visualize detections
    vis_frame = visualize_detections(frame, objects)
    
    # Visualize crop window
    vis_frame = visualize_crop_window(vis_frame, crop_window)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_frame)
    
    return vis_frame

def plot_crop_window_trajectory(crop_windows, frame_width, frame_height, output_path=None):
    """
    Plot the trajectory of crop windows over time.
    
    Args:
        crop_windows: List of crop windows, each as [x, y, width, height]
        frame_width: Width of the original frame
        frame_height: Height of the original frame
        output_path: Path to save the plot (if None, just displays the plot)
    """
    # Extract center points of crop windows
    centers_x = [x + w // 2 for x, y, w, h in crop_windows]
    centers_y = [y + h // 2 for x, y, w, h in crop_windows]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot frame boundaries
    plt.plot([0, frame_width, frame_width, 0, 0], [0, 0, frame_height, frame_height, 0], 'k-')
    
    # Plot crop window centers
    plt.plot(centers_x, centers_y, 'b-', alpha=0.7, label='Crop Window Center')
    plt.scatter(centers_x, centers_y, c=range(len(centers_x)), cmap='viridis', s=10)
    
    # Add colorbar to show time progression
    cbar = plt.colorbar()
    cbar.set_label('Frame Number')
    
    # Set plot properties
    plt.xlim(-frame_width * 0.1, frame_width * 1.1)
    plt.ylim(frame_height * 1.1, -frame_height * 0.1)  # Inverted y-axis to match image coordinates
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Crop Window Trajectory')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or show plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def download_models():
    """
    Download required model files if they don't exist.
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # YOLO model files
    yolo_files = {
        'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    for filename, url in yolo_files.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename} successfully.")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                print(f"Please download manually from {url} and place in {models_dir}") 