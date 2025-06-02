from ultralytics import YOLO
import cv2
import numpy as np
import os
import urllib.request
from PIL import Image
import time  

class ObjectDetector:
    """Class for detecting objects in video frames using YOLOv8."""
    





    def __init__(self, confidence_threshold=0.5, model_size='n', classes=[0], debug=False):
        """
        Initialize YOLOv8 detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_size = model_size
        self._initialize_model()
        self.classes = classes
        self.debug = debug
    






    def _initialize_model(self):
        """Initialize the YOLOv8 model."""
        try:
            model_filename = f'yolo11{self.model_size}.pt'
            local_model_path = os.path.join('models', model_filename)

            if not os.path.exists(local_model_path):
                os.makedirs('models', exist_ok=True)
                download_url = f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_filename}'
                print(f"Downloading YOLOv11 model from {download_url}...")
                urllib.request.urlretrieve(download_url, local_model_path)
                print(f"Model downloaded to {local_model_path}")

            # All classes are loaded by default, but we can filter them later
            # The classes are:
            # 0: person
            # 1: bicycle
            # 2: car
            # 3: motorcycle
            # 4: airplane
            # 5: bus
            # see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
            #
            # Arguments: https://docs.ultralytics.com/modes/predict/#inference-sources
            self.model = YOLO(local_model_path)
            print(f"YOLOv11-{self.model_size} model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
    






    def detect(self, frame, top_n=1):
        """
        Detect objects in a frame.
        
        Args:
            frame: The input frame (numpy array)
            top_n: Number of top detections to return, based on confidence. 
                If top_n=1, only the highest confidence detection is returned.
                If top_n=2, the top 2 detections are returned, and so on.
            
        Returns:
            List of detected objects, each as a dictionary with:
            - 'box': [x, y, width, height]
            - 'confidence': detection confidence
            - 'class_name': class name
            - 'class_id': class ID
        """

        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Run YOLOv11 inference
        results = self.model(frame, 
                            verbose=False, 
                            classes=self.classes,
                            )[0]
        
        # Process detections
        detections = []

        # Collect all valid detections
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det

            # Skip if the confidence is below the threshold
            if conf < self.confidence_threshold:
                continue

            width = x2 - x1
            height = y2 - y1

            x = int(x1)
            y = int(y1)
            w = int(width)
            h = int(height)
            class_id = int(cls)
            class_name = results.names[int(cls)]
            confidence = float(conf)

            detection = {
                'box': [x, y, w, h],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            }

            detections.append(detection)

        # Sort detections by confidence, in descending order
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # If top_n is specified, keep only the top 'n' detections
        if top_n > 0:
            detections = detections[:top_n]

        # DEBUG DETECTIONS!
        if self.debug and detections:
            timestamp = int(time.time() * 1000)
            debug_folder = f"debug_logs"
            os.makedirs(debug_folder, exist_ok=True)
            # Save the detection result into the debug folder
            results.save(os.path.join(debug_folder, f"detect_results_{timestamp}.jpg"))
            for detection in detections:
                with open("debug_logs/log1_detections.txt", "a") as f:
                    f.write(f"Detector Box : {detection['box']} class_name: {detection['class_name']}, "
                            f"Confidence: {detection['confidence']}\n")

        return detections
    








    def get_class_names(self):
        """Get list of class names the model can detect."""
        return self.model.names if self.model else {}








    # def _detect_yolo(self, frame):
    #     """YOLO object detection implementation."""
    #     height, width, _ = frame.shape
        
    #     # Prepare image for YOLO
    #     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #     self.model.setInput(blob)
        
    #     # Get detections
    #     outs = self.model.forward(self.output_layers)
        
    #     # Process detections
    #     class_ids = []
    #     confidences = []
    #     boxes = []
        
    #     for out in outs:
    #         for detection in out:
    #             scores = detection[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]
                
    #             if confidence > self.confidence_threshold:
    #                 # Object detected
    #                 center_x = int(detection[0] * width)
    #                 center_y = int(detection[1] * height)
    #                 w = int(detection[2] * width)
    #                 h = int(detection[3] * height)
                    
    #                 # Rectangle coordinates
    #                 x = int(center_x - w / 2)
    #                 y = int(center_y - h / 2)
                    
    #                 boxes.append([x, y, w, h])
    #                 confidences.append(float(confidence))
    #                 class_ids.append(class_id)


    #                 # 🖨️ Print detection details
    #                 print(f"Detected class_name: {class_name}, class ID: {class_id}, Confidence: {confidence:.2f}, Box: [x={x}, y={y}, w={w}, h={h}]")
        
    #     # Apply non-maximum suppression
    #     indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
    #     detected_objects = []
    #     for i in indices:
    #         if isinstance(i, list):  # Handle different OpenCV versions
    #             i = i[0]
                
    #         box = boxes[i]
    #         confidence = confidences[i]
    #         class_id = class_ids[i]
    #         class_name = self.classes[class_id]
            
    #         detected_objects.append({
    #             'box': box,  # [x, y, width, height]
    #             'confidence': confidence,
    #             'class_id': class_id,
    #             'class_name': class_name
    #         })
        
    #     return detected_objects
    





    # def _detect_ssd(self, frame):
    #     """SSD object detection implementation."""
    #     # Implementation for SSD model
    #     pass
    




    
    # def _detect_faster_rcnn(self, frame):
    #     """Faster R-CNN object detection implementation."""
    #     # Implementation for Faster R-CNN model
    #     pass 