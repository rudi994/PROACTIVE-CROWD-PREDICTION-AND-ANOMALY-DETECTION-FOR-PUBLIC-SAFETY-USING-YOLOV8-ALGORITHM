import os
import cv2

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'YOLOv4-tiny', 'yolov4-tiny.cfg')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'YOLOv4-tiny', 'yolov4-tiny.weights')

print(f"Current directory: {BASE_DIR}")
print(f"Config path: {CONFIG_PATH}")
print(f"Weights path: {WEIGHTS_PATH}")

# Check if files exist
print(f"Config file exists: {os.path.exists(CONFIG_PATH)}")
print(f"Weights file exists: {os.path.exists(WEIGHTS_PATH)}")

# If files exist, check if they can be read
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            first_line = f.readline()
            print(f"Config file first line: {first_line.strip()}")
    except Exception as e:
        print(f"Error reading config file: {e}")

if os.path.exists(WEIGHTS_PATH):
    file_size = os.path.getsize(WEIGHTS_PATH)
    print(f"Weights file size: {file_size} bytes (should be around 23-24MB)")