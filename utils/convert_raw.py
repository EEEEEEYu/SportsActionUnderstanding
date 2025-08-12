import os
import sys
import cv2
import time
import numpy as np
import tqdm
import argparse
import multiprocessing
from multiprocessing import Value
from scipy.spatial import cKDTree

from vme_research.messaging.shared_ndarray import SharedNDArrayPipe
from vme_research.hardware.prophesee_camera import PropheseeCamera
from vme_research.hardware.record import Record, Load


# Attempt to import the necessary custom library from your environment
try:
    from vme_research.hardware.record import Load, LoadEventStream
except ImportError as e:
    print(f"Failed to import vme_research library: {e}")
    print("Please ensure the 'vme_research' package is installed and in your PYTHONPATH.")
    # Define dummy classes to allow the script to be parsed, but it will fail on use.
    class Load: pass
    class LoadEventStream: pass

# --- Constants and Configuration ---
FLIR_DIR = 'flir_23604512'
PROPH_DIR = 'proph_00051463'
PROPH_EXP_DIR = 'proph_00051463_exported'


class TimeZeroSource:
    def __init__(self, t0=None):
        self.t0 = t0
        if self.t0 is None: self.t0 = time.time()

    def time(self):
        return time.time() - self.t0

#@jit(nopython=True)
def events_image(image, x, y, p):
    image[...] = 0.5
    # 2*p-1 converts from [0, 1] to [-1, 1]
    # divide by 15 to get [-1/15, 1/15], making 15 be the maximum value
    p_polarized = (2*p - 1) / 15
    for i in range(y.shape[0]):
        image[y[i], x[i]] += p_polarized[i]
    image = np.clip(image, 0, 1)

    return image

def convert_raw_to_npy(cam_folder, crop=True):
    stop = Value('i', 0)
    time_source = TimeZeroSource()

    event_sample = (np.zeros((1,), dtype=np.int64), np.zeros((1,), dtype=np.uint16), np.zeros((1,), dtype=np.uint16), np.zeros((1,), dtype=np.int16))
    e_buffer_size = int(2 * 150e6) # 2 seconds at 150 Mev/s
    e_pubsub = SharedNDArrayPipe(sample_data=event_sample, max_messages=e_buffer_size, check_overflow=False)

    loader = Load(cam_folder)

    sequence_folder, cam_name = os.path.split(os.path.normpath(cam_folder))
    npy_recorder = Record(os.path.join(sequence_folder, cam_name + '_exported'), time_source=time_source)
    camera = PropheseeCamera(stop, time_source, npy_recorder=npy_recorder, loader=loader, pub_sub=e_pubsub)
    # TODO: make the ROI configurable (make the camera a selection in the editor)

    try:
        no_data_count = 0
        camera.start()

        while True:
            time.sleep(0.01)

            data = e_pubsub.get(N=-1)
            if data is not None:
                image = np.full((720, 1280), 0.5, dtype=np.float32)
                no_data_count = 0

                et, ex, ey, ep = data
                et = et.flatten()
                ex = ex.flatten()
                ey = ey.flatten()
                ep = ep.flatten()
                
                ex = (1280 - 1) - ex

                image = events_image(image, ex, ey, ep)

                cv2.imshow('events image', image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                no_data_count += 1
                if no_data_count % 100 == 0:
                    print('Found', no_data_count, 'empty frames')
                if no_data_count > 1000:
                    break

    except KeyboardInterrupt: pass
    finally:
        stop.value = 1
        camera.join()

def get_calibration_parameters(calib_path, crop):
    """
    Loads calibration parameters. Returns:
    1. A map to warp the FLIR image to the Prophesee perspective.
    2. The intrinsic matrix (K) and distortion coefficients (dist) for the Prophesee camera.
    """
    print("Loading calibration data...")
    calib_flir_path = os.path.join(calib_path, FLIR_DIR, 'calibration_joint')
    calib_proph_path = os.path.join(calib_path, PROPH_EXP_DIR.replace('_exported', ''), 'calibration_joint')

    loader_cal_flir = Load(calib_flir_path)
    loader_cal_proph = Load(calib_proph_path)
    calib_data_flir = loader_cal_flir.get_appended()
    calib_data_proph = loader_cal_proph.get_appended()

    K_flir = np.array(calib_data_flir['K']).reshape(3, 3)
    dist_flir = np.array(calib_data_flir['dist'])
    R_vc_c_flir = np.array(calib_data_flir['R_vc_c']).reshape(3, 3)

    K_proph = np.array(calib_data_proph['K']).reshape(3, 3)
    dist_proph = np.array(calib_data_proph['dist'])
    R_vc_c_proph = np.array(calib_data_proph['R_vc_c']).reshape(3, 3)

    res_proph = (720, 720) if crop else (1280, 720)
    R_proph_flir = R_vc_c_proph.T @ R_vc_c_flir

    # Create map to warp FLIR image to (undistorted) Prophesee coordinates
    flir_map1, flir_map2 = cv2.initUndistortRectifyMap(
        K_flir, dist_flir, R_proph_flir, K_proph, res_proph, cv2.CV_32FC1)

    return flir_map1, flir_map2, K_proph, dist_proph

def stochastic_round_points(coords_float):
    """
    Rounds floating point coordinates stochastically to preserve statistical distribution.
    """
    # Generate random numbers for the stochastic choice
    rand_vals = np.random.rand(coords_float.shape[0], 2)
    
    # Get the integer floor and the fractional part
    coords_floor = np.floor(coords_float)
    coords_frac = coords_float - coords_floor
    
    # If the fractional part is greater than a random number, add 1.
    coords_int = coords_floor + (coords_frac > rand_vals)
    
    return coords_int.astype(np.int32)

class AlignedDataSaver:
    def __init__(self, seq_path, calib_path, crop=False):
        self.seq_path = seq_path
        self.calib_path = calib_path
        self.crop = crop
        self.crop_roi = (0, 0, 720, 720) if self.crop else None # ROI for cropped coordinates

        self.flir_camera_dir = os.path.join(self.seq_path, FLIR_DIR)
        self.proph_events_dir = os.path.join(self.seq_path, PROPH_EXP_DIR)

        self.output_events_dir = os.path.join(self.seq_path, "proc", 'events')
        self.output_flir_dir = os.path.join(self.seq_path, "proc", 'flir')
        os.makedirs(self.output_events_dir, exist_ok=True)
        os.makedirs(self.output_flir_dir, exist_ok=True)
        print(f"Output will be saved to:\n  {self.output_flir_dir}\n  {self.output_events_dir}")

        self.process_and_save()

    def process_and_save(self):
        print(f"Starting data processing for sequence: {self.seq_path}")

        # 1. Get calibration parameters
        flir_map1, flir_map2, K_proph, dist_proph = get_calibration_parameters(self.calib_path, self.crop)

        # 2. Load Data Streams & Synchronize
        loader_frame = Load(self.flir_camera_dir)
        loader_event = LoadEventStream(self.proph_events_dir)
        print("Performing time synchronization...")
        Load.time_synchronization(loader_frame, loader_event)

        flir_t_file = os.path.join(self.flir_camera_dir, 't.npy')
        if os.path.exists(flir_t_file):
            flir_t = np.load(flir_t_file)
            print(f"Found FLIR timestamps: {flir_t.shape}")
        # shift the time to be aligned with the events
        flir_t_shift = -flir_t[0]
        flir_t = (flir_t + flir_t_shift) * 1e6

        flir_data = loader_frame.get_all()
        frame_names = flir_data['frame']

        # --- Assume you have these defined ---
        # K_proph: Camera matrix
        # dist_proph: Distortion coefficients
        # h, w: Sensor height and width (e.g., 720, 720)
        h, w = 720, 720 # Example dimensions

        print("Performing one-time setup for event undistortion...")

        # 1. Calculate the inverse map from undistorted to distorted space
        # map1 holds the x_src coords, map2 holds the y_src coords
        map1, map2 = cv2.initUndistortRectifyMap(K_proph, dist_proph, None, K_proph, (w, h), cv2.CV_32FC1)

        # 2. Create the set of "ideal" source points
        # These are the (x, y) source locations that correspond to integer grid
        # points in the destination image.
        # We reshape them into a (H*W, 2) array for the k-d tree.
        ideal_source_points = np.stack((map1.ravel(), map2.ravel()), axis=-1)

        # 3. Build the k-d tree from these ideal points. This is the search structure.
        # This might take a second or two, but you only do it once.
        kdtree = cKDTree(ideal_source_points)
        print("Setup complete. Ready to process events.")


        event_t = loader_event.get_appended()['events_t']
        event_xy = loader_event.get_appended()['events_xy']
        event_p = loader_event.get_appended()['events_p']
        # TODO: maybe we can avoid using the vme driver for raw to npy conversion
        #       and just use metavision_sdk directly
        event_xy[:, 0] = (1280 - 1) - event_xy[:, 0]
        if self.crop:
            roi = (290,0,1010,720)
            event_xy = event_xy.astype(np.int32)
            event_xy[:, 0] = np.maximum(event_xy[:, 0] - roi[0], 0)
            event_xy[:, 1] = np.maximum(event_xy[:, 1] - roi[1], 0)
            event_xy = event_xy.astype(np.uint16)

        # convert these into event frames, remap, and double check there is no aliasing
        for i in range(0, event_xy.shape[0], 10000):
            # create event image (add up xy)
            event_image = np.zeros((h, w), dtype=np.float32)
            for j in range(i, min(i + 10000, event_xy.shape[0])):
                x, y = event_xy[j]
                print(x,y)
                event_image[y, x] += 1
            # undistort the events,
            # save the events,
            # and then go through the frames and rectify them
            event_image = cv2.remap(event_image, map1, map2, cv2.INTER_LINEAR)
            # TODO: using cv2.remap works fine, maybe they do antialiasing?
            # display image
            cv2.imshow("Event Image", event_image)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)

        # --- DIAGNOSTIC VISUALIZATION ---
        # print("Running diagnostic visualization... Close the window to continue.")
        # # 1. Prepare the event data for visualization (take a subset for clarity)
        # diagnostic_events = event_xy[:50000].astype(np.int32)

        # # 2. Create a blank canvas
        # # The canvas should be in the cropped coordinate system
        # h_crop, w_crop = 720, 720
        # diagnostic_image = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)

        # # 3. Draw your actual (distorted, cropped) events in one color (e.g., blue)
        # for x, y in diagnostic_events:
        #     if 0 <= x < w_crop and 0 <= y < h_crop:
        #         diagnostic_image[y, x] = (255, 0, 0) # Blue for actual events

        # # 4. Draw the k-d tree's "ideal source points" in another color (e.g., red)
        # # These points represent where the undistorted grid comes from.
        # diagnostic_kdtree_points = ideal_source_points.astype(np.int32)
        # for x, y in diagnostic_kdtree_points[::100]: # Draw a sparse grid for clarity
        #     if 0 <= x < w_crop and 0 <= y < h_crop:
        #         cv2.circle(diagnostic_image, (x, y), 1, (0, 0, 255), -1) # Red for ideal points

        # # 5. Show the image
        # cv2.imshow("Diagnostic: Blue=Your Events, Red=K-D Tree Points", diagnostic_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # --- END DIAGNOSTIC ---

        # --- UNDISTORT EVENT COORDINATES DIRECTLY ---        
        # 'event_xy' is your (N, 2) array of original, distorted event coordinates (float or int)
        # 'kdtree' is the tree built in the setup step.
        # 'w' is the width of the sensor.

        # 1. For each of your actual events, query the k-d tree to find the index
        #    of the CLOSEST ideal source point.
        #    `distance` will be the distance, `index` will be the row in ideal_source_points.
        distance, index = kdtree.query(event_xy)

        # 2. The 'index' corresponds to the flattened (H*W) grid. We need to
        #    convert this flat index back into a 2D coordinate (u, v).
        #    `np.unravel_index` is perfect for this. The result is (row, col) i.e. (y, x)
        rectified_rows, rectified_cols = np.unravel_index(index, (h, w))

        # 3. Stack them back into the desired (N, 2) format for (x, y) coordinates.
        #    This is your final result.
        rectified_coords_int = np.stack((rectified_cols, rectified_rows), axis=-1)

        # 'rectified_coords_int' is now an (N, 2) integer array.
        # It has the exact same number of events as event_xy.
        # The events are now snapped to the undistorted grid, minimizing gaps.

        # save the events_xy, events_t, events_p to separate numpy files
        np.save(os.path.join(self.output_events_dir, "events_xy.npy"), rectified_coords_int)
        np.save(os.path.join(self.output_events_dir, "events_t.npy"), event_t)
        np.save(os.path.join(self.output_events_dir, "events_p.npy"), event_p)

        # Find the first flir_t that is above the first event_t
        first_event_t = event_t[0]
        first_flir_idx = np.searchsorted(flir_t, first_event_t, side='right')
        flir_t = flir_t[first_flir_idx:]
        print(f"First FLIR index above first event timestamp: {first_flir_idx}")

        # save out the new file into the proc directory
        np.save(os.path.join(self.output_flir_dir, "t.npy"), flir_t)

        # Process and Save
        print(f"Processing and saving {len(frame_names) - 1} frame pairs...")
        file_idx = 0
        frame_names = frame_names[first_flir_idx:]
        for i in tqdm.tqdm(range(1, len(frame_names))):
            # --- Rectify FLIR Frame ---
            frame_path = os.path.join(self.flir_camera_dir, frame_names[i])
            if not os.path.exists(frame_path): continue
            frame_rgb = np.load(frame_path)
            rectified_flir = cv2.remap(frame_rgb, flir_map1, flir_map2, cv2.INTER_LINEAR)

            # save the rectified flir as a png image
            cv2.imwrite(os.path.join(self.output_flir_dir, f"{file_idx:06d}.png"), rectified_flir)
            
            file_idx += 1

        print(f"\nProcessing complete. Saved {file_idx} rectified data pairs.")


def process_sequence(sequence, calib, beamsplitter):
    # only convert if the "proph_00051463_exported" does not exist
    if not os.path.exists(os.path.join(sequence, "proph_00051463_exported")):
        print('Converting', sequence)
        convert_raw_to_npy(os.path.join(sequence, PROPH_DIR),
                        crop=beamsplitter)
    else:
        print('Skipping conversion, already exists:', os.path.join(sequence, "proph_00051463_exported"))

    # align the data and save it out
    AlignedDataSaver(
        seq_path=sequence,
        calib_path=calib,
        crop=beamsplitter
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and directly rectify event coordinates.")
    parser.add_argument("--sequence", type=str, help="Path to the sequence directory.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--calib", type=str, help="Path to the calibration directory.",
                        default="../../vme_research/calibrate/calibration_default")
    parser.add_argument("--beamsplitter", action='store_true', help="Enable beamsplitter mode (sets crop resolution to 720x720).")
    args = parser.parse_args()

    if args.sequence:
        process_sequence(args.sequence, args.calib, args.beamsplitter)
    elif args.dataset_dir:
        for sequence in os.listdir(args.dataset_dir):
            # skip if it's not a directory
            if not os.path.isdir(os.path.join(args.dataset_dir, sequence)):
                print(f"Skipping non-directory: {sequence}")
                continue
            process_sequence(os.path.join(args.dataset_dir, sequence), args.calib, args.beamsplitter)
