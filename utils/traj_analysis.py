# [ ] READ THESE
# https://rpg.ifi.uzh.ch/docs/IROS16_Kueng.pdf
# https://www.readcube.com/library/30766591-50a3-4e36-bba2-d382326275bf:439fa67d-3639-4c97-856e-10003c603523

# [x] load the images and events
# [x] detect and track the red circles (using tracking-by-detection with Kalman filter)
# [-] annotate data using my little annotator, train a Mask-R CNN on this for detection
# [ ] track the events between the ROIs in the event camera frames
# [ ] do it in the event cloud as well and make a continuous 3D curve

import os
import glob
import tqdm
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import utils

flir_frames, flir_t, res, K, dist = utils.load_frames("/home/matt/projects/vme_research/calibrate/calibrate_data_2025_08_24_data/procs/sequence_000003")

print(f"Loaded {flir_frames.shape[0]} frames.")

# --- Tracking-by-Detection Setup with Kalman Filter ---

class KalmanFilter2D:
    """2D Kalman filter for tracking position and velocity"""
    def __init__(self, initial_position, dt=1.0):
        # State vector: [x, y, vx, vy]
        self.state = np.array([initial_position[0], initial_position[1], 0, 0], dtype=np.float32)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float32)
        
        # Process noise covariance
        q = 0.1  # Process noise magnitude
        self.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                          [0, dt**4/4, 0, dt**3/2],
                          [dt**3/2, 0, dt**2, 0],
                          [0, dt**3/2, 0, dt**2]], dtype=np.float32) * q
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * 5.0  # Measurement noise
        
        # State covariance matrix
        self.P = np.eye(4, dtype=np.float32) * 100
        
    def predict(self):
        """Predict next state"""
        # Predict state
        self.state = self.F @ self.state
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Return predicted position
        return (self.state[0], self.state[1])
    
    def update(self, measurement):
        """Update state with measurement"""
        z = np.array(measurement, dtype=np.float32)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return (self.state[0], self.state[1])

class Track:
    """Track class with Kalman filter for state estimation"""
    def __init__(self, track_id, position, frame_idx):
        self.track_id = track_id
        self.positions = [position]  # List of (x, y) positions for visualization
        self.last_seen = frame_idx
        self.color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for visualization
        self.active = True
        self.kalman = KalmanFilter2D(position)  # Initialize Kalman filter
        self.age = 0
        self.hits = 1  # Number of successful updates
        
    def predict(self):
        """Predict next position using Kalman filter"""
        predicted_pos = self.kalman.predict()
        self.age += 1
        return predicted_pos
        
    def update(self, position, frame_idx):
        """Update track with new detection"""
        self.kalman.update(position)
        self.positions.append(position)
        self.last_seen = frame_idx
        self.hits += 1

# Tracking parameters
MAX_DISTANCE_THRESHOLD = 100  # Maximum distance in pixels to associate a detection with a track
MAX_FRAMES_TO_SKIP = 10  # Maximum frames a track can be lost before deletion
MIN_HITS = 3  # Minimum hits to consider a track confirmed

# Initialize tracking variables
tracks = []
next_track_id = 0
frame_count = 0

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area
params.filterByArea = True
params.minArea = 50
params.maxArea = 5000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = False

# Filter by Inertia
params.filterByInertia = False

# Create detector with parameters
detector = cv2.SimpleBlobDetector_create(params)

# --- Main Loop for Detection and Tracking ---
for i, frame in enumerate(flir_frames):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_frame = frame.copy()
    
    # --- Step 1: Blob Detection ---
    print(f"Frame {i}: Running Blob detection...")
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Wider, more robust red range
    lower_red1 = np.array([0, 200, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

    lower_red2 = np.array([170, 200, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    red_mask = mask1 + mask2
    blurred_mask = cv2.GaussianBlur(red_mask, (9, 9), 2)

    # Display the blurred_mask
    cv2.imshow('Blurred Mask', blurred_mask)

    # Binarize the image
    _, binary_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    # Invert the image for the blob detector
    inverted_mask = cv2.bitwise_not(binary_mask)
    
    # Find blobs in the image
    keypoints = detector.detect(inverted_mask)

    # Display the inverted_mask
    cv2.imshow('Inverted Mask', inverted_mask)
    
    # Extract detection positions
    detections = []
    if keypoints:
        for kp in keypoints:
            detections.append(kp.pt)
        print(f"Detected {len(detections)} blobs")
    else:
        print("No blobs detected")
    
    # --- Step 2: Kalman Prediction for all tracks ---
    
    # Predict new positions for all active tracks
    predicted_positions = []
    active_track_indices = []
    for idx, track in enumerate(tracks):
        if track.active:
            pred_pos = track.predict()
            predicted_positions.append(pred_pos)
            active_track_indices.append(idx)
    
    # --- Step 3: Hungarian Algorithm for Optimal Assignment ---
    
    if len(detections) > 0 and len(predicted_positions) > 0:
        # Compute cost matrix (distances between detections and predictions)
        detections_array = np.array(detections)
        predictions_array = np.array(predicted_positions)
        
        # Calculate pairwise distances
        cost_matrix = cdist(detections_array, predictions_array)
        
        # Apply Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Process assignments
        assigned_detections = set()
        assigned_tracks = set()
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            # Check if assignment is within threshold
            if cost_matrix[det_idx, track_idx] < MAX_DISTANCE_THRESHOLD:
                # Update track with detection
                actual_track_idx = active_track_indices[track_idx]
                tracks[actual_track_idx].update(detections[det_idx], i)
                assigned_detections.add(det_idx)
                assigned_tracks.add(track_idx)
            else:
                print(f"Assignment rejected: distance {cost_matrix[det_idx, track_idx]:.2f} > {MAX_DISTANCE_THRESHOLD}")
        
        # Create new tracks for unassigned detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in assigned_detections:
                new_track = Track(next_track_id, detection, i)
                tracks.append(new_track)
                next_track_id += 1
                print(f"Created new track {new_track.track_id}")
                
    elif len(detections) > 0:
        # No existing tracks, create new ones for all detections
        for detection in detections:
            new_track = Track(next_track_id, detection, i)
            tracks.append(new_track)
            next_track_id += 1
            print(f"Created new track {new_track.track_id}")
    
    # --- Step 4: Track Management (delete lost tracks) ---
    for track in tracks:
        if track.active and (i - track.last_seen) > MAX_FRAMES_TO_SKIP:
            track.active = False
            print(f"Lost track {track.track_id} (last seen: frame {track.last_seen})")
    
    # --- Step 5: Visualization ---
    
    # Draw current detections
    for detection in detections:
        x, y = int(detection[0]), int(detection[1])
        cv2.circle(output_frame, (x, y), 3, (255, 255, 255), -1)
    
    # Draw predicted positions (for debugging)
    for idx, track in enumerate(tracks):
        if track.active:
            pred_x, pred_y = int(track.kalman.state[0]), int(track.kalman.state[1])
            cv2.circle(output_frame, (pred_x, pred_y), 3, (0, 255, 255), 1)  # Yellow circle for prediction
    
    # Draw tracks
    for track in tracks:
        if track.active and track.hits >= MIN_HITS:  # Only show confirmed tracks
            # Draw track history
            if len(track.positions) > 1:
                points = np.array(track.positions, dtype=np.int32)
                for j in range(1, len(points)):
                    cv2.line(output_frame, tuple(points[j-1]), tuple(points[j]), 
                            track.color, 2)
            
            # Draw current position with track ID
            if len(track.positions) > 0:
                x, y = int(track.positions[-1][0]), int(track.positions[-1][1])
                cv2.circle(output_frame, (x, y), 5, track.color, -1)
                
                # Add velocity arrow if significant
                vx, vy = track.kalman.state[2], track.kalman.state[3]
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 0.5:  # Only show if moving
                    end_x = int(x + vx * 10)  # Scale velocity for visualization
                    end_y = int(y + vy * 10)
                    cv2.arrowedLine(output_frame, (x, y), (end_x, end_y), 
                                  track.color, 2, tipLength=0.3)
                
                # Draw track info
                info_text = f"ID:{track.track_id} H:{track.hits}"
                cv2.putText(output_frame, info_text, 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, track.color, 2)
    
    # Display tracking info
    active_tracks = sum(1 for t in tracks if t.active)
    confirmed_tracks = sum(1 for t in tracks if t.active and t.hits >= MIN_HITS)
    cv2.putText(output_frame, f"Frame: {i} | Tracks: {confirmed_tracks}/{active_tracks} | Detections: {len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Legend
    cv2.putText(output_frame, "White: Detection | Yellow: Prediction | Arrow: Velocity", 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Display the result
    cv2.imshow('Tracking', output_frame)
    
    # Play as a video (e.g., 100ms delay). Press 'q' to quit.
    key = cv2.waitKey(100) & 0xff
    if key == ord('q'):
        break
    elif key == ord(' '):  # Pause on spacebar
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Print final tracking statistics
print(f"\nTracking Statistics:")
print(f"Total tracks created: {next_track_id}")
print(f"Active tracks at end: {sum(1 for t in tracks if t.active)}")
print(f"Confirmed tracks: {sum(1 for t in tracks if t.hits >= MIN_HITS)}")

# Detailed track information
for track in tracks:
    if len(track.positions) > 5:  # Only show tracks that lasted more than 5 frames
        avg_speed = np.sqrt(track.kalman.state[2]**2 + track.kalman.state[3]**2)
        print(f"Track {track.track_id}: {len(track.positions)} frames, {track.hits} hits, avg speed: {avg_speed:.2f} px/frame")