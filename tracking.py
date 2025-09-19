import numpy as np
import cv2
from config import MIN_CONF, NMS_THRESH 

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

def detect_human(net, ln, frame, encoder, tracker, time):
    """
    Detect humans in frame using YOLO and track them using DeepSORT
    
    Args:
        net: YOLO network
        ln: Layer names
        frame: Input frame
        encoder: DeepSORT encoder
        tracker: DeepSORT tracker
        time: Current time for tracking
        
    Returns:
        [tracked_bboxes, expired]: List of current tracks and expired tracks
    """
    
    # Get the dimension of the frame
    (frame_height, frame_width) = frame.shape[:2]
    
    # Initialize lists needed for detection
    boxes = []
    centroids = []
    confidences = []

    try:
        # Construct a blob from the input frame 
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Perform forward pass of YOLO
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        # FIXED: Process detections properly
        for output in layer_outputs:
            for detection in output:
                # Extract the class ID and confidence 
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # FIXED: Class ID for person is 0, check if the confidence meets threshold
                if class_id == 0 and confidence > MIN_CONF:
                    # Scale the bounding box coordinates back to the size of the image
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (center_x, center_y, width, height) = box.astype("int")
                    
                    # Derive the coordinates for the top left corner of the bounding box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    
                    # FIXED: Ensure coordinates are within frame boundaries
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, frame_width - x)
                    height = min(height, frame_height - y)
                    
                    # Only add valid boxes
                    if width > 0 and height > 0:
                        boxes.append([x, y, int(width), int(height)])
                        centroids.append((int(center_x), int(center_y)))
                        confidences.append(float(confidence))

        # Debug output
        print(f"Initial detections: {len(boxes)} people detected")

    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return [[], []]

    # Initialize return values
    tracked_bboxes = []
    expired = []

    # FIXED: Perform Non-maxima suppression
    if len(boxes) > 0:
        try:
            # Apply NMS to suppress overlapping boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
            
            # FIXED: Handle NMS results properly
            if len(idxs) > 0:
                # Convert to flat array if needed
                if isinstance(idxs, tuple):
                    idxs = idxs[0] if len(idxs) > 0 else []
                elif hasattr(idxs, 'flatten'):
                    idxs = idxs.flatten()
                
                # Filter boxes based on NMS results
                filtered_boxes = []
                filtered_centroids = []
                filtered_confidences = []
                
                for i in idxs:
                    filtered_boxes.append(boxes[i])
                    filtered_centroids.append(centroids[i])
                    filtered_confidences.append(confidences[i])
                
                boxes = np.array(filtered_boxes)
                centroids = np.array(filtered_centroids)
                confidences = np.array(filtered_confidences)
                
                print(f"After NMS: {len(boxes)} people remaining")

                # FIXED: Create DeepSORT detections
                if len(boxes) > 0:
                    try:
                        # Generate features using DeepSORT encoder
                        features = encoder(frame, boxes)
                        
                        # Create Detection objects
                        detections = []
                        for bbox, confidence, centroid, feature in zip(boxes, confidences, centroids, features):
                            detection = Detection(bbox, confidence, centroid, feature)
                            detections.append(detection)
                        
                        # Update tracker
                        tracker.predict()
                        expired = tracker.update(detections, time)

                        # Get confirmed tracks
                        for track in tracker.tracks:
                            if track.is_confirmed() and track.time_since_update <= 1:
                                tracked_bboxes.append(track)
                        
                        print(f"Active tracks: {len(tracked_bboxes)}")
                        
                    except Exception as e:
                        print(f"Error in DeepSORT tracking: {e}")
                        # Fallback: return simple detections without tracking
                        class SimpleTrack:
                            def __init__(self, bbox, track_id, centroid):
                                self.track_id = track_id
                                self.positions = [centroid]
                                self._bbox = bbox
                            
                            def to_tlbr(self):
                                x, y, w, h = self._bbox
                                return np.array([x, y, x + w, y + h])
                            
                            def is_confirmed(self):
                                return True
                        
                        tracked_bboxes = []
                        for i, (bbox, centroid) in enumerate(zip(boxes, centroids)):
                            track = SimpleTrack(bbox, i, centroid)
                            tracked_bboxes.append(track)
            else:
                print("NMS returned no valid detections")
        
        except Exception as e:
            print(f"Error in NMS processing: {e}")
            # Fallback: use all detections without NMS
            if len(boxes) > 0:
                class SimpleTrack:
                    def __init__(self, bbox, track_id, centroid):
                        self.track_id = track_id
                        self.positions = [centroid]
                        self._bbox = bbox
                    
                    def to_tlbr(self):
                        x, y, w, h = self._bbox
                        return np.array([x, y, x + w, y + h])
                    
                    def is_confirmed(self):
                        return True
                
                tracked_bboxes = []
                for i, (bbox, centroid) in enumerate(zip(boxes[:5], centroids[:5])):  # Limit to 5 for safety
                    track = SimpleTrack(bbox, i, centroid)
                    tracked_bboxes.append(track)
    
    else:
        print("No initial detections found")

    return [tracked_bboxes, expired]