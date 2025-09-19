import time
import datetime
import numpy as np
import imutils
import cv2
import time
from math import ceil
from scipy.spatial.distance import euclidean
from tracking import detect_human
from util import rect_distance, progress, kinetic_energy
from colors import RGB_COLORS
from config import SHOW_DETECT, DATA_RECORD, RE_CHECK, RE_START_TIME, RE_END_TIME, SD_CHECK, SHOW_VIOLATION_COUNT, SHOW_TRACKING_ID, SOCIAL_DISTANCE,\
    SHOW_PROCESSING_OUTPUT, YOLO_CONFIG, VIDEO_CONFIG, DATA_RECORD_RATE, ABNORMAL_CHECK, ABNORMAL_ENERGY, ABNORMAL_THRESH, ABNORMAL_MIN_PEOPLE
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

IS_CAM = VIDEO_CONFIG["IS_CAM"]
HIGH_CAM = VIDEO_CONFIG["HIGH_CAM"]

def _record_movement_data(movement_data_writer, movement):
    """Record movement data to CSV file"""
    try:
        track_id = movement.track_id 
        entry_time = movement.entry 
        exit_time = movement.exit            
        positions = movement.positions
        positions = np.array(positions).flatten()
        positions = list(positions)
        data = [track_id] + [entry_time] + [exit_time] + positions
        movement_data_writer.writerow(data)
    except Exception as e:
        print(f"Error recording movement data: {e}")

def _record_crowd_data(time, human_count, violate_count, restricted_entry, abnormal_activity, crowd_data_writer):
    """Record crowd data to CSV file"""
    try:
        data = [time, human_count, violate_count, int(restricted_entry), int(abnormal_activity)]
        crowd_data_writer.writerow(data)
    except Exception as e:
        print(f"Error recording crowd data: {e}")

def _end_video(tracker, frame_count, movement_data_writer):
    """Handle end of video processing"""
    try:
        for t in tracker.tracks:
            if t.is_confirmed():
                t.exit = frame_count
                _record_movement_data(movement_data_writer, t)
    except Exception as e:
        print(f"Error ending video processing: {e}")

def video_process(cap, frame_size, net, ln, encoder, tracker, movement_data_writer, crowd_data_writer, VID_FPS=30, DATA_RECORD_FRAME=1):
    """
    Main video processing function with robust error handling
    
    Args:
        cap: Video capture object
        frame_size: Target frame size for processing
        net: YOLO network
        ln: Layer names
        encoder: Deep SORT encoder
        tracker: Deep SORT tracker
        movement_data_writer: CSV writer for movement data
        crowd_data_writer: CSV writer for crowd data
        VID_FPS: Video FPS (default: 30)
        DATA_RECORD_FRAME: Frame recording interval (default: 1)
    
    Returns:
        Final FPS value
    """
    
    def _calculate_FPS():
        """Calculate FPS for camera input"""
        nonlocal VID_FPS
        try:
            if t0 is not None:
                t1 = time.time() - t0
                if t1 > 0:
                    calculated_fps = frame_count / t1
                    VID_FPS = calculated_fps
                    print(f"Calculated FPS: {VID_FPS:.2f}")
                else:
                    print("Warning: Invalid time elapsed for FPS calculation")
        except Exception as e:
            print(f"Error calculating FPS: {e}")

    # FIXED: Robust parameter validation
    if VID_FPS is None or VID_FPS <= 0:
        print(f"Warning: Invalid FPS parameter ({VID_FPS}), using default: 30")
        VID_FPS = 30

    if DATA_RECORD_FRAME is None or DATA_RECORD_FRAME <= 0:
        print(f"Warning: Invalid DATA_RECORD_FRAME parameter ({DATA_RECORD_FRAME}), using default: 1")
        DATA_RECORD_FRAME = 1

    # FIXED: Calculate TIME_STEP safely with validation
    try:
        TIME_STEP = DATA_RECORD_FRAME / VID_FPS
        if TIME_STEP <= 0:
            TIME_STEP = 1/30  # Fallback
            print(f"Warning: Invalid TIME_STEP calculated, using fallback: {TIME_STEP}")
        else:
            print(f"Time step: {TIME_STEP:.4f} seconds")
    except (ZeroDivisionError, TypeError) as e:
        TIME_STEP = 1/30  # Safe fallback
        print(f"Error calculating TIME_STEP: {e}, using fallback: {TIME_STEP}")

    # Initialize timing variables
    if IS_CAM:
        t0 = time.time()
    else:
        t0 = None

    # Initialize counters
    frame_count = 0
    display_frame_count = 0
    re_warning_timeout = 0
    sd_warning_timeout = 0
    ab_warning_timeout = 0

    # Initialize flags
    RE = False
    ABNORMAL = False

    print(f"Starting video processing...")
    print(f"- Frame size: {frame_size}")
    print(f"- FPS: {VID_FPS}")
    print(f"- Record interval: {DATA_RECORD_FRAME} frames")
    print(f"- Camera mode: {IS_CAM}")

    try:
        while True:
            # Read frame from video
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"Error reading frame: {e}")
                break

            # Stop the loop when video ends
            if not ret:
                print("End of video reached or failed to read frame")
                _end_video(tracker, frame_count, movement_data_writer)
                if IS_CAM and t0 is not None:
                    _calculate_FPS()
                break

            # Update frame count with overflow protection
            if frame_count > 1000000:
                if IS_CAM and t0 is not None:
                    _calculate_FPS()
                    t0 = time.time()  # Reset timer
                frame_count = 0
                display_frame_count = 0
                print("Frame counter reset due to overflow protection")
            
            frame_count += 1
            
            # Skip frames according to given rate
            if frame_count % DATA_RECORD_FRAME != 0:
                continue

            display_frame_count += 1

            # Progress indicator for non-visual processing
            if not SHOW_PROCESSING_OUTPUT and display_frame_count % 100 == 0:
                print(f"Processed {display_frame_count} frames...")

            # Resize Frame to given size with error handling
            try:
                if frame is not None and frame.size > 0:
                    frame = imutils.resize(frame, width=frame_size)
                else:
                    print("Warning: Invalid frame detected, skipping...")
                    continue
            except Exception as e:
                print(f"Error resizing frame: {e}")
                continue

            # Get current time
            try:
                current_datetime = datetime.datetime.now()
            except Exception as e:
                print(f"Error getting current time: {e}")
                current_datetime = datetime.datetime.now()

            # Determine record time based on input type
            if IS_CAM:
                record_time = current_datetime
            else:
                record_time = frame_count
            
            # Run tracking algorithm with error handling
            try:
                [humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, record_time)
            except Exception as e:
                print(f"Error in human detection: {e}")
                humans_detected = []
                expired = []

            # Record movement data for expired tracks
            try:
                for movement in expired:
                    _record_movement_data(movement_data_writer, movement)
            except Exception as e:
                print(f"Error recording expired movements: {e}")
            
            # Check for restricted entry
            if RE_CHECK:
                try:
                    RE = False
                    if (current_datetime.time() > RE_START_TIME) and (current_datetime.time() < RE_END_TIME):
                        if len(humans_detected) > 0:
                            RE = True
                except Exception as e:
                    print(f"Error checking restricted entry: {e}")
                    RE = False
                
            # Process detections for visualization and analysis
            if SHOW_PROCESSING_OUTPUT or SHOW_DETECT or SD_CHECK or RE_CHECK or ABNORMAL_CHECK:
                try:
                    # Initialize violation tracking
                    violate_set = set()
                    violate_count = np.zeros(len(humans_detected)) if len(humans_detected) > 0 else np.array([])

                    # Initialize abnormal activity tracking
                    abnormal_individual = []
                    ABNORMAL = False
                    
                    # Process each detected person
                    for i, track in enumerate(humans_detected):
                        try:
                            # Get object bounding box
                            bbox = track.to_tlbr()
                            if len(bbox) >= 4:
                                [x, y, w, h] = list(map(int, bbox.tolist()))
                            else:
                                continue

                            # Get object centroid - safely handle positions
                            if hasattr(track, 'positions') and len(track.positions) > 0:
                                try:
                                    [cx, cy] = list(map(int, track.positions[-1]))
                                except (IndexError, ValueError, TypeError):
                                    cx, cy = int((x + w) / 2), int((y + h) / 2)  # Fallback to bbox center
                            else:
                                cx, cy = int((x + w) / 2), int((y + h) / 2)  # Fallback to bbox center

                            # Get object id
                            idx = getattr(track, 'track_id', i)
                            
                            # Check for social distance violation
                            if SD_CHECK and len(humans_detected) >= 2:
                                for j, track_2 in enumerate(humans_detected[i+1:], start=i+1):
                                    try:
                                        if HIGH_CAM:
                                            # Use centroid distance for high camera
                                            if hasattr(track_2, 'positions') and len(track_2.positions) > 0:
                                                [cx_2, cy_2] = list(map(int, track_2.positions[-1]))
                                            else:
                                                bbox_2 = track_2.to_tlbr()
                                                [x_2, y_2, w_2, h_2] = list(map(int, bbox_2.tolist()))
                                                cx_2, cy_2 = int((x_2 + w_2) / 2), int((y_2 + h_2) / 2)
                                            distance = euclidean((cx, cy), (cx_2, cy_2))
                                        else:
                                            # Use bounding box distance for normal camera
                                            bbox_2 = track_2.to_tlbr()
                                            [x_2, y_2, w_2, h_2] = list(map(int, bbox_2.tolist()))
                                            distance = rect_distance((x, y, w, h), (x_2, y_2, w_2, h_2))
                                        
                                        if distance < SOCIAL_DISTANCE:
                                            violate_set.add(i)
                                            violate_count[i] += 1
                                            violate_set.add(j)
                                            violate_count[j] += 1
                                    except Exception as e:
                                        print(f"Error checking social distance for tracks {i}, {j}: {e}")

                            # Compute energy level for abnormal activity detection
                            if ABNORMAL_CHECK and hasattr(track, 'positions') and len(track.positions) >= 2:
                                try:
                                    ke = kinetic_energy(track.positions[-1], track.positions[-2], TIME_STEP)
                                    if ke > ABNORMAL_ENERGY:
                                        abnormal_individual.append(getattr(track, 'track_id', i))
                                except Exception as e:
                                    print(f"Error calculating kinetic energy for track {i}: {e}")

                            # Draw bounding boxes and labels
                            try:
                                # Restricted entry visualization
                                if RE:
                                    cv2.rectangle(frame, (x + 5, y + 5), (w - 5, h - 5), RGB_COLORS["red"], 5)

                                # Social distance violation visualization
                                if i in violate_set:
                                    cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["yellow"], 2)
                                    if SHOW_VIOLATION_COUNT:
                                        cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["yellow"], 2)
                                elif SHOW_DETECT and not RE:
                                    cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["green"], 2)
                                    if SHOW_VIOLATION_COUNT:
                                        cv2.putText(frame, str(int(violate_count[i])), (x, y - 10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
                                
                                # Tracking ID visualization
                                if SHOW_TRACKING_ID:
                                    cv2.putText(frame, str(int(idx)), (x, y - 30), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["blue"], 2)
                            except Exception as e:
                                print(f"Error drawing bounding box for track {i}: {e}")

                        except Exception as e:
                            print(f"Error processing track {i}: {e}")
                            continue
                    
                    # Check for overall abnormal activity level
                    if ABNORMAL_CHECK and len(humans_detected) > ABNORMAL_MIN_PEOPLE and len(abnormal_individual) > 0:
                        try:
                            abnormal_ratio = len(abnormal_individual) / len(humans_detected)
                            if abnormal_ratio > ABNORMAL_THRESH:
                                ABNORMAL = True
                        except ZeroDivisionError:
                            ABNORMAL = False

                except Exception as e:
                    print(f"Error in detection processing loop: {e}")
                    violate_set = set()
                    ABNORMAL = False

            # Display warnings and information on frame
            try:
                # Social distance violation warning
                if SD_CHECK:
                    if len(violate_set) > 0:
                        sd_warning_timeout = 10
                    else: 
                        sd_warning_timeout -= 1
                    
                    if sd_warning_timeout > 0:
                        text = f"Violation count: {len(violate_set)}"
                        cv2.putText(frame, text, (200, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Restricted entry warning
                if RE_CHECK:
                    if RE:
                        re_warning_timeout = 10
                    else: 
                        re_warning_timeout -= 1
                    
                    if re_warning_timeout > 0:
                        if display_frame_count % 3 != 0:  # Blinking effect
                            cv2.putText(frame, "RESTRICTED ENTRY", (200, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, RGB_COLORS["red"], 3)

                # Abnormal activity warning
                if ABNORMAL_CHECK:
                    if ABNORMAL:
                        ab_warning_timeout = 10
                        # Highlight abnormal individuals
                        for track in humans_detected:
                            if hasattr(track, 'track_id') and track.track_id in abnormal_individual:
                                try:
                                    bbox = track.to_tlbr()
                                    [x, y, w, h] = list(map(int, bbox.tolist()))
                                    cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["blue"], 5)
                                except:
                                    pass
                    else:
                        ab_warning_timeout -= 1
                    
                    if ab_warning_timeout > 0:
                        if display_frame_count % 3 != 0:  # Blinking effect
                            cv2.putText(frame, "ABNORMAL ACTIVITY", (130, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, RGB_COLORS["blue"], 5)

                # Display crowd count
                if SHOW_DETECT:
                    text = f"Crowd count: {len(humans_detected)}"
                    cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            except Exception as e:
                print(f"Error adding text overlays: {e}")

            # Record crowd data to file
            if DATA_RECORD:
                try:
                    _record_crowd_data(record_time, len(humans_detected), len(violate_set), RE, ABNORMAL, crowd_data_writer)
                except Exception as e:
                    print(f"Error recording crowd data: {e}")

            # Display video output or show progress
            try:
                if SHOW_PROCESSING_OUTPUT:
                    cv2.imshow("Processed Output", frame)
                    # Press 'Q' to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("User requested quit")
                        break
                else:
                    progress(display_frame_count)
            except Exception as e:
                print(f"Error displaying output: {e}")

    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Unexpected error in main processing loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup and final processing
        try:
            _end_video(tracker, frame_count, movement_data_writer)
        except Exception as e:
            print(f"Error in final cleanup: {e}")
        
        if IS_CAM and t0 is not None:
            _calculate_FPS()
        
        cv2.destroyAllWindows()
        print(f"Video processing completed. Total frames processed: {display_frame_count}")
    
    return VID_FPS