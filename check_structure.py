import os

def check_directory_structure():
    print("Current working directory:", os.getcwd())
    print("\nDirectory contents:")
    for item in os.listdir():
        if os.path.isdir(item):
            print(f"[DIR]  {item}/")
        else:
            print(f"[FILE] {item}")
    
    # Check if YOLOv4-tiny folder exists
    yolov4_dir = "YOLOv4-tiny"
    if os.path.exists(yolov4_dir) and os.path.isdir(yolov4_dir):
        print(f"\nContents of {yolov4_dir}/:")
        for item in os.listdir(yolov4_dir):
            print(f"  {item}")
    else:
        print(f"\n{yolov4_dir} directory does not exist!")

if __name__ == "__main__":
    check_directory_structure()