import numpy as np
import cv2

class CropCalculator:
    """Enhanced class for calculating optimal crop windows based on detected objects and saliency."""
    
    def __init__(self, 
                 target_ratio=9/16, 
                 padding_ratio=0.1, 
                 class_weights=None, 
                 size_weight=0.4, 
                 center_weight=0.3, 
                 motion_weight=0.3, 
                 history_weight=0.7, 
                 saliency_weight=0.4,
                 debug=False,
                 face_detection=False,
                 weighted_center=False,
                 blend_saliency=False,
                ):
        """
        Initialize the crop calculator with enhanced parameters.
        
        Args:
            target_ratio: Target aspect ratio (width/height) for the output video
            padding_ratio: Padding around objects as a ratio of frame dimensions
            class_weights: Dictionary mapping class names to importance weights
            size_weight: Weight for object size in importance calculation
            center_weight: Weight for object center proximity in importance calculation
            motion_weight: Weight for object motion in importance calculation
            history_weight: Weight for historical crop positions (stability)
            saliency_weight: Weight for saliency map in importance calculation
        """
        self.target_ratio = target_ratio
        self.padding_ratio = padding_ratio
        self.size_weight = size_weight
        self.center_weight = center_weight
        self.motion_weight = motion_weight
        self.history_weight = history_weight
        self.saliency_weight = saliency_weight
        self.debug = debug
        self.face_detection = face_detection
        self.weighted_center = weighted_center
        self.blend_saliency = blend_saliency
        
        # Enhanced class weights with more categories and higher contrast
        self.class_weights = class_weights or {
            'person': 40,       # Higher priority for people
            'face': 40,         # Highest priority for faces
            'dog': 30,
            'cat': 30,
            'sports ball': 20,  # Higher priority for sports activities
            'bicycle': 10,
            'motorcycle': 10,
            'car': 10,
            'bird': 10,
            'default': 0.5       # Lower default to increase contrast
        }
        
        # Keep track of previous crop windows for stability
        self.prev_crop_window = None
        self.prev_weighted_center = None
        self.prev_saliency_map = None
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize saliency detector (lightweight and fast)
        self.saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    
    def calculate(self, objects, frame_width, frame_height, frame=None):
        """
        Calculate optimal crop window based on detected objects and saliency.
        
        Args:
            objects: List of detected/tracked objects
            frame_width: Width of the original frame
            frame_height: Height of the original frame
            frame: Optional original frame for additional analysis (face detection, saliency, etc.)
            
        Returns:
            Crop window as [x, y, width, height]
        """
        # Ensure objects is a list
        if isinstance(objects, dict):
            objects = list(objects.values())
        elif not isinstance(objects, list):
            objects = []
        
        # If frame is provided, enhance detection
        saliency_center = None
        if frame is not None:
            if self.face_detection:
                # Detect faces to add to objects
                faces = self._detect_faces(frame)
                for face in faces:
                    face_obj = {
                        'box': face,
                        'class_name': 'face',
                        'confidence': 0.9,
                        'class_id': -1  # Special ID for faces
                    }
                    objects.append(face_obj)
            
            # Calculate saliency map (only if we have few or no objects)
            if len(objects) < 3:
                saliency_center = self._calculate_saliency(frame, frame_width, frame_height)
            
            if objects and objects[0]['box'] is not None:
                if self.debug:
                    print(f"✂️ Inputs Box: [x={objects[0]['box'][0]}, y={objects[0]['box'][1]}, w={objects[0]['box'][2]}, h={objects[0]['box'][3]}], confidence: {objects[0]['confidence']:.2f}")
                    with open("debug_logs/log2a_cropinputs.txt", "a") as f:
                        f.write(f"Crop Inputs Box: [x={objects[0]['box'][0]}, y={objects[0]['box'][1]}, w={objects[0]['box'][2]}, h={objects[0]['box'][3]}], confidence: {objects[0]['confidence']:.2f}\n")


        # If no objects detected and no saliency, use previous crop window or center of frame
        if not objects and saliency_center is None:
            if self.prev_crop_window is not None:
                return self.prev_crop_window
            return self._get_center_crop(frame_width, frame_height)
        
        # Calculate importance for each object
        for obj in objects:
            obj['importance'] = self._calculate_importance(obj, frame_width, frame_height)
        
        # Sort objects by importance (descending)
        sorted_objects = sorted(objects, key=lambda x: x['importance'], reverse=True)
        
        # Calculate weighted center of attention from objects
        if sorted_objects:
            if self.weighted_center:
                total_weight = sum(obj['importance'] for obj in sorted_objects)
                weighted_x = sum(self._get_center_x(obj['box']) * obj['importance'] for obj in sorted_objects) / total_weight
                weighted_y = sum(self._get_center_y(obj['box']) * obj['importance'] for obj in sorted_objects) / total_weight
            else:
                weighted_x = self._get_center_x(obj['box'])
                weighted_y = self._get_center_y(obj['box'])
        else:
            # If no objects but we have saliency
            weighted_x, weighted_y = saliency_center or (frame_width / 2, frame_height / 2)
        
        # If we have both objects and saliency, blend them
        if sorted_objects and saliency_center and self.blend_saliency:
            saliency_x, saliency_y = saliency_center
            # Blend based on number and importance of objects
            if len(sorted_objects) <= 2:
                # With few objects, give more weight to saliency
                blend_factor = 0.3
                weighted_x = weighted_x * (1 - blend_factor) + saliency_x * blend_factor
                weighted_y = weighted_y * (1 - blend_factor) + saliency_y * blend_factor
        
        # If we have a previous weighted center, apply history weight for stability
        if self.prev_weighted_center is not None:
            prev_x, prev_y = self.prev_weighted_center
            weighted_x = prev_x * self.history_weight + weighted_x * (1 - self.history_weight)
            weighted_y = prev_y * self.history_weight + weighted_y * (1 - self.history_weight)
        
        # Store current weighted center for next frame
        self.prev_weighted_center = (weighted_x, weighted_y)
        
        # Calculate crop dimensions
        crop_height = frame_height
        crop_width = int(crop_height * self.target_ratio)
        
        # If target crop is wider than the frame, adjust dimensions
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = int(crop_width / self.target_ratio)
        
        # Calculate crop position
        x = int(weighted_x - crop_width / 2)
        y = int(weighted_y - crop_height / 2)
        
        # Ensure crop window is within frame boundaries
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))
        
        # Create current crop window
        current_crop = [x, y, crop_width, crop_height]
        
        # If we have a previous crop window, apply smoothing
        if self.prev_crop_window is not None:
            current_crop = self._smooth_transition(self.prev_crop_window, current_crop)
        
        # Store current crop window for next frame
        self.prev_crop_window = current_crop

        if self.debug:
            print(f"✂️ CropCalc Box: [x={x}, y={y}, w={crop_width}, h={crop_height}]")
            with open("debug_logs/log2b_cropcalc.txt", "a") as f:
                f.write(f"CropCalc Box: [x={x}, y={y}, w={crop_width}, h={crop_height}]\n")
        
        return current_crop
    
    def _detect_faces(self, frame):
        """Detect faces in the frame using Haar Cascade."""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Convert to [x, y, width, height] format
        return [[x, y, w, h] for (x, y, w, h) in faces]
    
    def _calculate_saliency(self, frame, frame_width, frame_height):
        """Calculate saliency map and return the center of the most salient region."""
        # Resize frame for faster processing
        max_dim = 400
        scale = min(max_dim / frame_width, max_dim / frame_height)
        if scale < 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small_frame = frame
        
        # Calculate saliency map
        success, saliency_map = self.saliency_detector.computeSaliency(small_frame)
        
        if not success:
            return None
        
        # Normalize and convert to 8-bit
        saliency_map = (saliency_map * 255).astype(np.uint8)
        
        # Apply threshold to get the most salient regions
        _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours of salient regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the center of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        # Calculate center and scale back to original size
        center_x = int(M["m10"] / M["m00"] / scale)
        center_y = int(M["m01"] / M["m00"] / scale)
        
        return (center_x, center_y)
    
    def _calculate_importance(self, obj, frame_width, frame_height):
        """Calculate importance score for an object with enhanced metrics."""
        box = obj['box']
        class_name = obj.get('class_name', 'default')
        confidence = obj.get('confidence', 1.0)
        
        # Get class weight
        class_weight = self.class_weights.get(class_name, self.class_weights['default'])
        
        # Calculate size factor (normalized by frame area)
        size = box[2] * box[3]
        frame_area = frame_width * frame_height
        size_factor = size / frame_area
        
        # Calculate center proximity factor
        center_x = frame_width / 2
        center_y = frame_height / 2
        obj_center_x = self._get_center_x(box)
        obj_center_y = self._get_center_y(box)
        
        # Normalize distance to frame center
        max_distance = np.sqrt((frame_width/2)**2 + (frame_height/2)**2)
        distance = np.sqrt((center_x - obj_center_x)**2 + (center_y - obj_center_y)**2)
        center_factor = 1 - (distance / max_distance)
        
        # Combine factors with higher weight for class and size
        importance = (
            class_weight * 1.5 *  # Increase class weight influence
            confidence * 
            (self.size_weight * size_factor * 1.2 + self.center_weight * center_factor)
        )
        
        return importance
    
    def _smooth_transition(self, prev_crop, current_crop):
        """Apply smoothing between consecutive crop windows."""
        # Extract coordinates
        prev_x, prev_y, prev_w, prev_h = prev_crop
        curr_x, curr_y, curr_w, curr_h = current_crop
        
        # Calculate smoothed coordinates
        smooth_x = int(prev_x * self.history_weight + curr_x * (1 - self.history_weight))
        smooth_y = int(prev_y * self.history_weight + curr_y * (1 - self.history_weight))
        smooth_w = int(prev_w * self.history_weight + curr_w * (1 - self.history_weight))
        smooth_h = int(prev_h * self.history_weight + curr_h * (1 - self.history_weight))
        
        return [smooth_x, smooth_y, smooth_w, smooth_h]
    
    def _get_center_x(self, box):
        """Get x-coordinate of box center."""
        return box[0] + box[2] / 2
    
    def _get_center_y(self, box):
        """Get y-coordinate of box center."""
        return box[1] + box[3] / 2
    
    def _get_center_crop(self, frame_width, frame_height):
        """Get crop window centered in the frame."""
        crop_height = frame_height
        crop_width = int(crop_height * self.target_ratio)
        
        # If target crop is wider than the frame, adjust dimensions
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = int(crop_width / self.target_ratio)
        
        # Center the crop window
        x = int((frame_width - crop_width) / 2)
        y = int((frame_height - crop_height) / 2)
        
        return [x, y, crop_width, crop_height] 