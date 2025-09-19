from config import YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE

if FRAME_SIZE > 1920:
    print("Frame size is too large!")
    quit()
elif FRAME_SIZE < 480:
    print("Frame size is too small! You won't see anything")
    quit()

import datetime
import time
import numpy as np
import imutils
import cv2
import os
import csv
import json
from video_process import video_process
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the paths to YOLO files - try different possible locations
possible_config_paths = [
    os.path.join(BASE_DIR, 'YOLOv4-tiny', 'yolov4-tiny.cfg'),
    os.path.join(BASE_DIR, 'yolov4-tiny.cfg'),
    os.path.join(BASE_DIR, 'cfg', 'yolov4-tiny.cfg'),
    os.path.join(BASE_DIR, YOLO_CONFIG["CONFIG_PATH"])
]

possible_weights_paths = [
    os.path.join(BASE_DIR, 'YOLOv4-tiny', 'yolov4-tiny.weights'),
    os.path.join(BASE_DIR, 'yolov4-tiny.weights'),
    os.path.join(BASE_DIR, 'weights', 'yolov4-tiny.weights'),
    os.path.join(BASE_DIR, YOLO_CONFIG["WEIGHTS_PATH"])
]

# Find the actual config file
CONFIG_PATH = None
for path in possible_config_paths:
    if os.path.exists(path):
        CONFIG_PATH = path
        print(f"Found config file at: {CONFIG_PATH}")
        break

# Find the actual weights file
WEIGHTS_PATH = None
for path in possible_weights_paths:
    if os.path.exists(path):
        WEIGHTS_PATH = path
        print(f"Found weights file at: {WEIGHTS_PATH}")
        break

# Check if files were found
if not CONFIG_PATH:
    print("ERROR: Could not find yolov4-tiny.cfg file!")
    print("Please make sure the file exists in one of these locations:")
    for path in possible_config_paths:
        print(f"  - {path}")
    quit()

if not WEIGHTS_PATH:
    print("ERROR: Could not find yolov4-tiny.weights file!")
    print("Please make sure the file exists in one of these locations:")
    for path in possible_weights_paths:
        print(f"  - {path}")
    quit()

# Read from video
IS_CAM = VIDEO_CONFIG["IS_CAM"]

# Improved video capture handling
def initialize_video_capture():
    """Initialize video capture with proper error handling"""
    cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
    
    if not cap.isOpened():
        print("Error: Could not open video source!")
        # Try different backends if the first one fails
        if not IS_CAM:
            print("Trying alternative video backends...")
            cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"], cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"], cv2.CAP_ANY)
    
    if not cap.isOpened():
        print("Failed to open video source with all backends!")
        quit()
    
    return cap

cap = cv2.VideoCapture(r"C:\Users\Rutuja\Desktop\SEM 7 CROWWD VS\Crowd-Analysis\assets\videoplayback.mp4")

# Load the YOLO network
print("Loading YOLO network...")
try:
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    print("YOLO network loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO network: {e}")
    quit()

# Set the preferable backend to CPU since we are not using GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of all the layers in the network
ln = net.getLayerNames()
# Filter out the layer names we dont need for YOLO
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Tracker parameters
max_cosine_distance = 0.7
nn_budget = None

# IMPROVED FPS CALCULATION with robust fallbacks
def get_video_fps(cap, is_cam=False):
    """Get video FPS with multiple fallback methods"""
    if is_cam:
        return max(1, VIDEO_CONFIG.get("CAM_APPROX_FPS", 30))
    
    # Method 1: Direct FPS property
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        print(f"FPS from video properties: {fps}")
        return fps
    
    # Method 2: Calculate from frame count and duration
    print("Direct FPS failed, trying frame count method...")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if total_frames > 0:
        # Save current position
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Go to end to get duration
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Reset to original position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if duration_ms > 0:
            calculated_fps = total_frames / (duration_ms / 1000.0)
            if calculated_fps > 0:
                print(f"FPS calculated from frame count: {calculated_fps}")
                return calculated_fps
    
    # Method 3: Manual timing (sample a few frames)
    print("Frame count method failed, trying manual timing...")
    try:
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        start_time = time.time()
        frame_count = 0
        
        for i in range(10):  # Sample 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        
        end_time = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)  # Reset position
        
        if frame_count > 0 and (end_time - start_time) > 0:
            manual_fps = frame_count / (end_time - start_time)
            print(f"FPS from manual timing: {manual_fps}")
            return manual_fps
    except Exception as e:
        print(f"Manual FPS calculation failed: {e}")
    
    # Method 4: Default fallback
    print("All FPS detection methods failed, using default FPS of 30")
    return 30

# Calculate VID_FPS with improved error handling
VID_FPS = get_video_fps(cap, IS_CAM)
VID_FPS = max(1, VID_FPS)  # Ensure at least 1 FPS
print(f"Final FPS: {VID_FPS}")

# Calculate max_age based on video type
if IS_CAM:
    max_age = int(VID_FPS * TRACK_MAX_AGE)
else:
    max_age = DATA_RECORD_RATE * TRACK_MAX_AGE

# Ensure reasonable max_age values
max_age = max(1, min(max_age, 30))
print(f"Tracker max_age: {max_age}")

# Add proper path handling for model file with better error messages
model_filename = os.path.join(BASE_DIR, 'model_data', 'mars-small128.pb')
if not os.path.exists(model_filename):
    # Try alternative locations
    alternative_paths = [
        os.path.join(BASE_DIR, 'mars-small128.pb'),
        os.path.join(BASE_DIR, 'deep_sort', 'model_data', 'mars-small128.pb'),
        os.path.join(BASE_DIR, 'model_data', 'mars-small128.pb')
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            model_filename = alt_path
            print(f"Found model file at: {model_filename}")
            break
    else:
        print(f"ERROR: Model file not found!")
        print("Searched in the following locations:")
        print(f"  - {model_filename}")
        for path in alternative_paths:
            print(f"  - {path}")
        print("\nPlease download it from: https://github.com/nwojke/deep_sort/raw/master/deep_sort/model_data/mars-small128.pb")
        print("And place it in the 'model_data' directory")
        quit()

# Initialize encoder and tracker with error handling
try:
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=max_age)
    print("Deep SORT tracker initialized successfully!")
except Exception as e:
    print(f"Error initializing tracker: {e}")
    quit()

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
    print("Created 'processed_data' directory")

# Initialize CSV files with proper error handling
try:
    movement_data_file = open('processed_data/movement_data.csv', 'w', newline='', encoding='utf-8') 
    crowd_data_file = open('processed_data/crowd_data.csv', 'w', newline='', encoding='utf-8')

    movement_data_writer = csv.writer(movement_data_file)
    crowd_data_writer = csv.writer(crowd_data_file)

    # Write headers if files are empty
    if os.path.getsize('processed_data/movement_data.csv') == 0:
        movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
    if os.path.getsize('processed_data/crowd_data.csv') == 0:
        crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance violate', 'Restricted Entry', 'Abnormal Activity'])
    
    print("CSV files initialized successfully!")
except Exception as e:
    print(f"Error initializing CSV files: {e}")
    quit()

START_TIME = time.time()

# Calculate DATA_RECORD_FRAME with proper bounds checking
DATA_RECORD_FRAME = max(1, int(VID_FPS / DATA_RECORD_RATE)) if DATA_RECORD_RATE > 0 else 1
print(f"Data record frame interval: {DATA_RECORD_FRAME}")

# Main processing with comprehensive error handling
try:
    print("Starting video processing...")
    processing_FPS = video_process(cap, FRAME_SIZE, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer, VID_FPS, DATA_RECORD_FRAME)
    print("Video processing completed!")
except Exception as e:
    print(f"Error during video processing: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Ensure cleanup happens even if there's an error
    cv2.destroyAllWindows()
    
    # Close files safely
    try:
        movement_data_file.close()
        crowd_data_file.close()
        print("Data files closed successfully!")
    except:
        pass
    
    try:
        cap.release()
        print("Video capture released!")
    except:
        pass

END_TIME = time.time()
PROCESS_TIME = END_TIME - START_TIME
print(f"Total processing time: {PROCESS_TIME:.2f} seconds")

# Calculate final FPS statistics
if IS_CAM:
    final_fps = processing_FPS if processing_FPS is not None and processing_FPS > 0 else VID_FPS
    print(f"Camera processing FPS: {final_fps:.2f}")
else:
    try:
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        processed_fps = total_frames / PROCESS_TIME if PROCESS_TIME > 0 and total_frames > 0 else VID_FPS
        print(f"Video processing FPS: {processed_fps:.2f}")
        print(f"Original video FPS: {VID_FPS:.2f}")
    except:
        processed_fps = VID_FPS
        print(f"Using fallback FPS: {processed_fps:.2f}")

# Create comprehensive video metadata
try:
    video_data = {
        "IS_CAM": IS_CAM,
        "DATA_RECORD_FRAME": DATA_RECORD_FRAME,
        "VID_FPS": float(VID_FPS),
        "PROCESSED_FRAME_SIZE": FRAME_SIZE,
        "TRACK_MAX_AGE": TRACK_MAX_AGE,
        "DATA_RECORD_RATE": DATA_RECORD_RATE,
        "PROCESSING_TIME": PROCESS_TIME,
        "START_TIME": datetime.datetime.fromtimestamp(START_TIME).strftime("%d/%m/%Y, %H:%M:%S"),
        "END_TIME": datetime.datetime.fromtimestamp(END_TIME).strftime("%d/%m/%Y, %H:%M:%S"),
        "CONFIG_PATH": CONFIG_PATH,
        "WEIGHTS_PATH": WEIGHTS_PATH,
        "MODEL_PATH": model_filename
    }

    with open('processed_data/video_data.json', 'w') as video_data_file:
        json.dump(video_data, video_data_file, indent=2)
    
    print("Video metadata saved successfully!")
except Exception as e:
    print(f"Error saving video metadata: {e}")

print("\n" + "="*50)
print("PROCESSING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Data files saved in 'processed_data' directory:")
print(f"  - movement_data.csv")
print(f"  - crowd_data.csv") 
print(f"  - video_data.json")
print("="*50)