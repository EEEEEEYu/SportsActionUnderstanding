# https://github.com/uzh-rpg/DSEC/issues/14#issuecomment-841348958

import os
import sys
import cv2
import glob
import time
import json
import numpy as np
import tqdm
import argparse
import multiprocessing
from multiprocessing import Value
from numba import jit

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

@jit(nopython=True)
def events_image(image, x, y, p):
    image[...] = 0.5
    # 2*p-1 converts from [0, 1] to [-1, 1]
    # divide by 15 to get [-1/15, 1/15], making 15 be the maximum value
    p_polarized = (2*p - 1) / 15
    for i in range(y.shape[0]):
        image[y[i], x[i]] += p_polarized[i]
    image = np.clip(image, 0, 1)

    return image

def convert_raw_to_npy(cam_folder):
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

    return flir_map1, flir_map2, K_proph, dist_proph, K_flir, dist_flir


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
        flir_map1, flir_map2, K_proph, dist_proph, K_flir, dist_flir = get_calibration_parameters(self.calib_path, self.crop)

        # 2. Load Data Streams & Synchronize
        loader_frame = Load(self.flir_camera_dir)
        loader_event = LoadEventStream(self.proph_events_dir)
        print("Performing time synchronization...")
        Load.time_synchronization(loader_frame, loader_event)

        flir_data_file = os.path.join(self.flir_camera_dir, "data.json")
        with open(flir_data_file, "r+") as f:
            flir_data = json.load(f)
        # add K_flir and dist_flir
        flir_data["append_fields"]["K"] = K_flir.tolist()
        flir_data["append_fields"]["dist"] = dist_flir.tolist()

        flir_t_file = os.path.join(self.flir_camera_dir, 't.npy')
        if os.path.exists(flir_t_file):
            flir_t = np.load(flir_t_file)
            print(f"Found FLIR timestamps: {flir_t.shape}")
        # shift the time to be aligned with the events
        flir_t_shift = -flir_t[0]
        flir_t = (flir_t + flir_t_shift) * 1e6

        frame_names = loader_frame.get_all()['frame']

        event_data_file = os.path.join(self.proph_events_dir, "data.json")
        with open(event_data_file, "r+") as f:
            event_data = json.load(f)
        event_t = loader_event.get_appended()['events_t']
        event_xy = loader_event.get_appended()['events_xy']
        event_p = loader_event.get_appended()['events_p']
        # TODO: maybe we can avoid using the vme driver for raw to npy conversion
        #       and just use metavision_sdk directly
        # TODO: move the cropping ROI information to one place
        event_xy[:, 0] = (1280 - 1) - event_xy[:, 0]
        if self.crop:
            roi = (290,0,1010,720)
            event_xy = event_xy.astype(np.int32)
            event_xy[:, 0] = np.maximum(event_xy[:, 0] - roi[0], 0)
            event_xy[:, 1] = np.maximum(event_xy[:, 1] - roi[1], 0)
            event_xy = event_xy.astype(np.uint16)
            # update resolution in data.json file
            event_data["append_fields"]["res"] = [720, 720]
            flir_data["append_fields"]["res"] = [720, 720]

        # save the events_xy, events_t, events_p to separate numpy files
        np.save(os.path.join(self.output_events_dir, "events_xy.npy"), event_xy)
        np.save(os.path.join(self.output_events_dir, "events_t.npy"), event_t)
        np.save(os.path.join(self.output_events_dir, "events_p.npy"), event_p)
        # save data.json to the new directory
        event_data["append_fields"]["K"] = K_flir.tolist()
        event_data["append_fields"]["dist"] = dist_flir.tolist()
        with open(os.path.join(self.output_events_dir, "data.json"), "w") as f:
            json.dump(event_data, f)

        # save out to output_flir_dir
        with open(os.path.join(self.output_flir_dir, "data.json"), "w") as f:
            json.dump(flir_data, f)

        # Find the first flir_t that is above the first event_t
        first_event_t = event_t[0]
        first_flir_idx = np.searchsorted(flir_t, first_event_t, side='right')
        flir_t = flir_t[first_flir_idx:]
        print(f"First FLIR index above first event timestamp: {first_flir_idx}")

        # save out the new file into the proc directory
        np.save(os.path.join(self.output_flir_dir, "flir_t.npy"), flir_t)

        # Process and Save
        print(f"Processing and saving {len(frame_names) - 1} frame pairs...")
        # makedirs for output_flir_dir/frame
        os.makedirs(os.path.join(self.output_flir_dir, "frame"), exist_ok=True)
        file_idx = 0
        frame_names = frame_names[first_flir_idx:]
        for i in tqdm.tqdm(range(1, len(frame_names))):
            # --- Rectify FLIR Frame ---
            frame_path = os.path.join(self.flir_camera_dir, frame_names[i])
            if not os.path.exists(frame_path): continue
            frame_rgb = np.load(frame_path)
            rectified_flir = cv2.remap(frame_rgb, flir_map1, flir_map2, cv2.INTER_LINEAR)

            # save the rectified flir as a png image
            cv2.imwrite(os.path.join(self.output_flir_dir, "frame", f"{file_idx:06d}.png"), rectified_flir)
            
            file_idx += 1

        print(f"\nProcessing complete. Saved {file_idx} rectified data pairs.")


def validate_sequence(sequence):
    # load FLIR data
    with open(os.path.join(sequence, "proc", "flir", "data.json")) as f:
        flir_data = json.load(f)
    flir_res = flir_data["append_fields"]["res"]
    flir_K = np.array(flir_data["append_fields"]["K"]).reshape(3, 3)
    flir_dist = np.array(flir_data["append_fields"]["dist"])

    # load event data
    with open(os.path.join(sequence, "proc", "events", "data.json")) as f:
        event_data = json.load(f)
    event_res = event_data["append_fields"]["res"]
    event_K = np.array(event_data["append_fields"]["K"]).reshape(3, 3)
    event_dist = np.array(event_data["append_fields"]["dist"])

    # Validate FLIR data
    flir_frames = sorted(glob.glob(os.path.join(sequence, "proc", "flir", "frame", "*.png")))
    # check if frames exist
    if not flir_frames:
        raise ValueError(f"No FLIR frames found in {os.path.join(sequence, 'proc', 'flir', 'frame')}")
    for frame in flir_frames:
        # check resolution
        frame_res = cv2.imread(frame).shape[:2]
        if not np.array_equal(frame_res, flir_res):
            print(f"Warning: FLIR frame {frame} has resolution {frame_res} but expected {flir_res}")

    # Validate event data
    event_xy = np.load(os.path.join(sequence, "proc", "events", "events_xy.npy"))
    # check if events_xy exists
    if event_xy is None:
        raise ValueError(f"No events_xy found in {os.path.join(sequence, 'proc', 'events', 'events_xy.npy')}")
    # check if events_xy has the correct shape
    if event_xy.shape[1] != 2:
        raise ValueError(f"Invalid events_xy shape: {event_xy.shape}")
    # check if events_xy are not negative
    if np.any(event_xy < 0):
        raise ValueError(f"events_xy coordinates {event_xy} are negative")
    # check if events_xy are within the bounds of the resolution
    if not np.all(event_xy[:, 0] < event_res[0]) or not np.all(event_xy[:, 1] < event_res[1]):
        raise ValueError(f"events_xy coordinates {event_xy} are out of bounds for resolution {event_res}")


def process_sequence(sequence, calib, beamsplitter, only_validate=False):
    if only_validate:
        validate_sequence(sequence)
        return
    
    # only convert if the "proph_00051463_exported" does not exist
    if not os.path.exists(os.path.join(sequence, "proph_00051463_exported")):
        print('Converting', sequence)
        convert_raw_to_npy(os.path.join(sequence, PROPH_DIR))
    else:
        print('Skipping conversion, already exists:', os.path.join(sequence, "proph_00051463_exported"))

    # align the data and save it out
    AlignedDataSaver(
        seq_path=sequence,
        calib_path=calib,
        crop=beamsplitter
    )

    # validate the data
    validate_sequence(sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and directly rectify event coordinates.")
    parser.add_argument("--sequence", type=str, help="Path to the sequence directory.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--calib", type=str, help="Path to the calibration directory.",
                        default="../../vme_research/calibrate/calibration_default")
    parser.add_argument("--beamsplitter", action='store_true', help="Enable beamsplitter mode (sets crop resolution to 720x720).")
    parser.add_argument("--validate", action='store_true', help="Only validate the sequence without processing.")
    args = parser.parse_args()

    if args.sequence:
        process_sequence(args.sequence, args.calib, 
                         args.beamsplitter, args.validate)
    elif args.dataset_dir:
        for sequence in tqdm.tqdm(os.listdir(args.dataset_dir)):
            if not sequence.startswith("sequence_"):
                continue
            # skip if it's not a directory
            if not os.path.isdir(os.path.join(args.dataset_dir, sequence)):
                print(f"Skipping non-directory: {sequence}")
                continue
            process_sequence(os.path.join(args.dataset_dir, sequence), args.calib, 
                             args.beamsplitter, args.validate)

