import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    """Class for tracking objects across video frames."""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # Dictionary: object_id -> {'box': [x, y, w, h], 'class_id': class_id, ...}
        self.disappeared = {}  # Dictionary: object_id -> count of frames where object disappeared
        
        self.max_disappeared = max_disappeared  # Maximum number of frames an object can be missing
        self.max_distance = max_distance  # Maximum distance for considering it the same object
    




    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Extract coordinates
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection coordinates
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        # Check if there is an intersection
        if xx2 < xx1 or yy2 < yy1:
            return 0.0
        
        # Calculate area of intersection
        intersection_area = (xx2 - xx1) * (yy2 - yy1)
        
        # Calculate area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou
    






    def _calculate_distance(self, box1, box2):
        """Calculate distance between centers of two bounding boxes."""
        # Extract coordinates
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate centers
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        
        # Calculate Euclidean distance
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    





    def update(self, frame, detections):
        """
        Update object tracking with new detections.
        
        Args:
            frame: Current video frame
            detections: List of detected objects from ObjectDetector
            
        Returns:
            Dictionary of tracked objects with their IDs
        """
        # If no objects are being tracked yet, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
        
        # If no detections in current frame, mark all existing objects as disappeared
        elif len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # If object has been missing for too long, deregister it
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
        
        # Otherwise, match detections with existing objects
        else:
            # Get IDs and boxes of existing objects
            object_ids = list(self.objects.keys())
            object_boxes = [self.objects[object_id]['box'] for object_id in object_ids]
            
            # Get boxes of new detections
            detection_boxes = [detection['box'] for detection in detections]
            
            # Calculate distance/similarity matrix
            distance_matrix = np.zeros((len(object_boxes), len(detection_boxes)))
            
            for i, object_box in enumerate(object_boxes):
                for j, detection_box in enumerate(detection_boxes):
                    # Use a combination of IoU and center distance for matching
                    iou = self._calculate_iou(object_box, detection_box)
                    distance = self._calculate_distance(object_box, detection_box)
                    
                    # Convert IoU to a distance-like metric (1-IoU)
                    iou_distance = 1.0 - iou
                    
                    # Normalize distance by max_distance
                    normalized_distance = min(distance / self.max_distance, 1.0)
                    
                    # Combine metrics (weighted average)
                    combined_distance = 0.7 * iou_distance + 0.3 * normalized_distance
                    
                    distance_matrix[i, j] = combined_distance
            
            # Use Hungarian algorithm to find optimal assignment
            row_indices, col_indices = linear_sum_assignment(distance_matrix)
            
            # Keep track of matched object IDs
            used_object_ids = set()
            used_detection_indices = set()
            
            # Update matched objects
            for row_idx, col_idx in zip(row_indices, col_indices):
                if distance_matrix[row_idx, col_idx] < 0.7:
                    object_id = object_ids[row_idx]
                    detection = detections[col_idx]
                    
                    # Preserve all detection properties
                    self.objects[object_id] = detection.copy()  # Make a copy to avoid reference issues
                    self.disappeared[object_id] = 0
                    
                    used_object_ids.add(object_id)
                    used_detection_indices.add(col_idx)
            
            # Check for disappeared objects
            for object_id in object_ids:
                if object_id not in used_object_ids:
                    self.disappeared[object_id] += 1
                    
                    # If object has been missing for too long, deregister it
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._deregister(object_id)
            
            # Register new detections
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    self._register(detection)
        
        # Return current tracked objects
        return {object_id: self.objects[object_id] for object_id in self.objects 
                if self.disappeared[object_id] <= self.max_disappeared}
    





    def _register(self, detection):
        """Register a new object."""
        self.objects[self.next_object_id] = detection
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    





    def _deregister(self, object_id):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id] 