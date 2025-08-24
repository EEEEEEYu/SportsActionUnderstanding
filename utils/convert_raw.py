import os
import sys
import cv2
import glob
import time
import json
import numpy as np
import tqdm
import argparse
import shutil
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
PROPH_RES = (1280, 720)  # Prophesee resolution
CROP_ROI = (290, 0, 1010, 720)  # ROI for cropped coordinates
CROP_RES = (720, 720)


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
                image = np.full([PROPH_RES[1], PROPH_RES[0]], 0.5, dtype=np.float32)
                no_data_count = 0

                et, ex, ey, ep = data
                et = et.flatten()
                ex = ex.flatten()
                ey = ey.flatten()
                ep = ep.flatten()

                ex = (PROPH_RES[0] - 1) - ex

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

    res_proph = CROP_RES if crop else PROPH_RES
    R_proph_flir = R_vc_c_proph.T @ R_vc_c_flir

    # Create map to warp FLIR image to (undistorted) Prophesee coordinates
    flir_map1, flir_map2 = cv2.initUndistortRectifyMap(
        K_flir, dist_flir, R_proph_flir, K_proph, res_proph, cv2.CV_32FC1)

    return flir_map1, flir_map2, K_proph, dist_proph, K_flir, dist_flir


class AlignedDataSaver:
    def __init__(self, seq_path, calib_path, crop=False, crop_idx=None, crop_name=None):
        self.seq_path = seq_path
        self.calib_path = calib_path
        self.crop = crop
        self.crop_idx = crop_idx  # Index of which crop to use from metadata
        self.crop_name = crop_name  # Name of the crop for output directory
        self.crop_roi = (0, 0, CROP_RES[0], CROP_RES[1]) if self.crop else None # ROI for cropped coordinates

        self.flir_camera_dir = os.path.join(self.seq_path, FLIR_DIR)
        self.proph_events_dir = os.path.join(self.seq_path, PROPH_EXP_DIR)

        # Load crop metadata if available
        self.crop_metadata = None
        self.load_crop_metadata()

        # Set output directories based on whether we're processing a specific crop
        if self.crop_name:
            # Create proc folder with crop name within the same sequence
            proc_dir = f"proc_{self.crop_name}"
            self.output_events_dir = os.path.join(self.seq_path, proc_dir, 'events')
            self.output_flir_dir = os.path.join(self.seq_path, proc_dir, 'flir')
        else:
            # Use the default output location
            self.output_events_dir = os.path.join(self.seq_path, "proc", 'events')
            self.output_flir_dir = os.path.join(self.seq_path, "proc", 'flir')
        
        os.makedirs(self.output_events_dir, exist_ok=True)
        os.makedirs(self.output_flir_dir, exist_ok=True)
        print(f"Output will be saved to:\n  {self.output_flir_dir}\n  {self.output_events_dir}")

        self.process_and_save()
    
    def load_crop_metadata(self):
        """Load crop metadata from crop.json file if available"""
        # Try loading from crop.json in the sequence root
        crop_json_path = os.path.join(self.seq_path, "crop.json")
        if os.path.exists(crop_json_path):
            with open(crop_json_path, 'r') as f:
                crop_data = json.load(f)
                if 'crops' in crop_data:
                    crops = crop_data['crops']
                    if crops and self.crop_idx is not None:
                        # Use specified crop index
                        idx = self.crop_idx
                        if idx < len(crops):
                            self.crop_metadata = crops[idx]
                            print(f"Loaded crop metadata: {self.crop_metadata.get('name', f'crop_{idx}')}")
                            print(f"  Frames: {self.crop_metadata['start_index']} to {self.crop_metadata['end_index']}")
                            if 'flir_start_time' in self.crop_metadata:
                                print(f"  FLIR time: {self.crop_metadata['flir_start_time']:.6f} to {self.crop_metadata['flir_end_time']:.6f}")
                            if 'event_start_time' in self.crop_metadata:
                                print(f"  Event time: {self.crop_metadata['event_start_time']:.6f} to {self.crop_metadata['event_end_time']:.6f}")
                        else:
                            print(f"Warning: Crop index {idx} out of range (found {len(crops)} crops)")
                    print(f"Successfully loaded {len(crops)} crops from crop.json")

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
        print(f"Loaded {len(frame_names)} frame names and {len(flir_t)} timestamps")

        event_data_file = os.path.join(self.proph_events_dir, "data.json")
        with open(event_data_file, "r+") as f:
            event_data = json.load(f)
        event_t = loader_event.get_appended()['events_t']
        event_xy = loader_event.get_appended()['events_xy']
        event_p = loader_event.get_appended()['events_p']
        # TODO: maybe we can avoid using the vme driver for raw to npy conversion
        #       and just use metavision_sdk directly
        # TODO: move the cropping ROI information to one place
        event_xy[:, 0] = (PROPH_RES[0] - 1) - event_xy[:, 0]
        if self.crop:
            roi = CROP_ROI
            event_xy = event_xy.astype(np.int32)
            event_xy[:, 0] = np.maximum(event_xy[:, 0] - roi[0], 0)
            event_xy[:, 1] = np.maximum(event_xy[:, 1] - roi[1], 0)
            event_xy = event_xy.astype(np.uint16)
            # update resolution in data.json file
            event_data["append_fields"]["res"] = list(CROP_RES)
            flir_data["append_fields"]["res"] = list(CROP_RES)

        # Find the first flir_t that is above the first event_t (do this before cropping)
        first_event_t = event_t[0] if len(event_t) > 0 else 0
        first_flir_idx = np.searchsorted(flir_t, first_event_t, side='right')
        print(f"First FLIR index above first event timestamp: {first_flir_idx}")

        # Apply crop metadata if available
        if self.crop_metadata:
            # Crop frame names based on metadata indices
            start_idx = self.crop_metadata['start_index']
            end_idx = self.crop_metadata['end_index'] + 1  # +1 because end is inclusive
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(frame_names) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(frame_names)))
            
            # Apply cropping to frame names and timestamps
            # Note: frame_names and flir_t should have the same length initially
            frame_names = frame_names[start_idx:end_idx]
            flir_t = flir_t[start_idx:end_idx]
            
            # Crop events based on time range if available, otherwise use frame-based estimation
            if 'event_start_time' in self.crop_metadata and 'event_end_time' in self.crop_metadata:
                # Use explicit event times if available
                # Convert from seconds (in crop.json) to microseconds (event_t units)
                event_start_time = self.crop_metadata['event_start_time'] * 1e6
                event_end_time = self.crop_metadata['event_end_time'] * 1e6
            elif len(flir_t) > 0:
                # Estimate event times based on FLIR timestamps
                event_start_time = flir_t[0] if len(flir_t) > 0 else event_t[0]
                event_end_time = flir_t[-1] if len(flir_t) > 0 else event_t[-1]
            else:
                # Fall back to using all events
                event_start_time = event_t[0]
                event_end_time = event_t[-1]
            
            # Find indices for event cropping
            event_start_idx = np.searchsorted(event_t, event_start_time, side='left')
            event_end_idx = np.searchsorted(event_t, event_end_time, side='right')
            
            # Crop event data
            event_t = event_t[event_start_idx:event_end_idx]
            event_xy = event_xy[event_start_idx:event_end_idx]
            event_p = event_p[event_start_idx:event_end_idx]
            
            print(f"Applied crop '{self.crop_metadata.get('name', f'crop_{self.crop_idx}')}':")
            print(f"  Frames: {len(frame_names)} (from {start_idx} to {end_idx})")
            print(f"  FLIR timestamps: {len(flir_t)}")
            print(f"  Events: {len(event_t)} (from {event_start_idx} to {event_end_idx})")
            
            # Validate that flir_t matches frame count
            if len(flir_t) != len(frame_names):
                print(f"WARNING: flir_t length ({len(flir_t)}) doesn't match frame_names length ({len(frame_names)})")
            
            # Update data.json with crop info
            flir_data["crop_applied"] = self.crop_metadata
            event_data["crop_applied"] = self.crop_metadata
        else:
            # Adjust frame_names and timestamps based on first_flir_idx as before
            frame_names = frame_names[first_flir_idx:]
            flir_t = flir_t[first_flir_idx:]
        
        # Save the events after any cropping
        np.save(os.path.join(self.output_events_dir, "events_xy.npy"), event_xy)
        np.save(os.path.join(self.output_events_dir, "events_t.npy"), event_t)
        np.save(os.path.join(self.output_events_dir, "events_p.npy"), event_p)
        
        # Save data.json files
        event_data["append_fields"]["K"] = K_flir.tolist()
        event_data["append_fields"]["dist"] = dist_flir.tolist()
        with open(os.path.join(self.output_events_dir, "data.json"), "w") as f:
            json.dump(event_data, f)
        
        with open(os.path.join(self.output_flir_dir, "data.json"), "w") as f:
            json.dump(flir_data, f)
        
        # Save flir timestamps
        print(f"Saving {len(flir_t)} FLIR timestamps to {self.output_flir_dir}")
        np.save(os.path.join(self.output_flir_dir, "flir_t.npy"), flir_t)
        
        # Process and Save
        print(f"Processing and saving {len(frame_names)} frame pairs...")
        # makedirs for output_flir_dir/frame
        os.makedirs(os.path.join(self.output_flir_dir, "frame"), exist_ok=True)
        file_idx = 0
        for i in tqdm.tqdm(range(len(frame_names))):
            # --- Rectify FLIR Frame ---
            frame_path = os.path.join(self.flir_camera_dir, frame_names[i])
            if not os.path.exists(frame_path): continue
            frame_rgb = np.load(frame_path)
            
            # Apply zero rectangles if specified in crop metadata
            if self.crop_metadata and 'zero_rectangles' in self.crop_metadata:
                for rect in self.crop_metadata['zero_rectangles']:
                    x, y, w, h = rect
                    # Ensure rectangle is within bounds
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(frame_rgb.shape[1], x + w)
                    y_end = min(frame_rgb.shape[0], y + h)
                    # Zero out the rectangle region
                    frame_rgb[y_start:y_end, x_start:x_end] = 0
            rectified_flir = cv2.remap(frame_rgb, flir_map1, flir_map2, cv2.INTER_LINEAR)

            # save the rectified flir as a png image
            cv2.imwrite(os.path.join(self.output_flir_dir, "frame", f"{file_idx:06d}.png"), rectified_flir)
            
            file_idx += 1

        print(f"\nProcessing complete. Saved {file_idx} rectified data pairs.")


def validate_sequence_with_proc_dir(sequence, proc_dir_name="proc"):
    """Validate a sequence with a specific proc directory name"""
    # load FLIR data
    with open(os.path.join(sequence, proc_dir_name, "flir", "data.json")) as f:
        flir_data = json.load(f)
    flir_res = flir_data["append_fields"]["res"]
    flir_K = np.array(flir_data["append_fields"]["K"]).reshape(3, 3)
    flir_dist = np.array(flir_data["append_fields"]["dist"])

    # load event data
    with open(os.path.join(sequence, proc_dir_name, "events", "data.json")) as f:
        event_data = json.load(f)
    event_res = event_data["append_fields"]["res"]
    event_K = np.array(event_data["append_fields"]["K"]).reshape(3, 3)
    event_dist = np.array(event_data["append_fields"]["dist"])

    # Validate FLIR data
    flir_frames = sorted(glob.glob(os.path.join(sequence, proc_dir_name, "flir", "frame", "*.png")))
    # check if frames exist
    if not flir_frames:
        raise ValueError(f"No FLIR frames found in {os.path.join(sequence, proc_dir_name, 'flir', 'frame')}")
    for frame in flir_frames:
        # check resolution
        frame_res = cv2.imread(frame).shape[:2]
        if not np.array_equal(frame_res, flir_res):
            print(f"Warning: FLIR frame {frame} has resolution {frame_res} but expected {flir_res}")

    # Validate event data
    event_xy = np.load(os.path.join(sequence, proc_dir_name, "events", "events_xy.npy"))
    # check if events_xy exists
    if event_xy is None:
        raise ValueError(f"No events_xy found in {os.path.join(sequence, proc_dir_name, 'events', 'events_xy.npy')}")
    # check if events_xy has the correct shape
    if event_xy.shape[1] != 2:
        raise ValueError(f"Invalid events_xy shape: {event_xy.shape}")
    # check if events_xy are not negative
    if np.any(event_xy < 0):
        raise ValueError(f"events_xy coordinates {event_xy} are negative")
    # check if events_xy are within the bounds of the resolution
    if not np.all(event_xy[:, 0] < event_res[0]) or not np.all(event_xy[:, 1] < event_res[1]):
        raise ValueError(f"events_xy coordinates {event_xy} are out of bounds for resolution {event_res}")


def validate_sequence(sequence):
    """Validate a sequence with the default proc directory"""
    validate_sequence_with_proc_dir(sequence, "proc")


def get_crops_from_sequence(sequence):
    """Get all crops from a sequence's crop.json file"""
    crop_json_path = os.path.join(sequence, "crop.json")
    if os.path.exists(crop_json_path):
        with open(crop_json_path, 'r') as f:
            crop_data = json.load(f)
            if 'crops' in crop_data:
                return crop_data['crops']
    return None


def move_proc_folders_to_dataset(sequence_path, dataset_procs_dir, process_all_crops=False, crop_idx=None):
    """Move proc* folders from a sequence to the dataset procs directory based on processing mode"""
    sequence_name = os.path.basename(sequence_path)
    
    # Determine which proc folders to move based on the mode
    proc_folders = []
    moved_folders = []  # Track moved folders for return
    
    if process_all_crops or crop_idx is not None:
        # Only move crop-specific folders when crop processing is enabled
        for item in os.listdir(sequence_path):
            if item.startswith("proc_") and os.path.isdir(os.path.join(sequence_path, item)):
                # Skip the default "proc" folder when crop flags are used
                proc_folders.append(item)
    else:
        # Only move the default "proc" folder when no crop flags are used
        if os.path.exists(os.path.join(sequence_path, "proc")):
            proc_folders.append("proc")
    
    if not proc_folders:
        print(f"No relevant proc folders found in {sequence_name}")
        return []
    
    # Move each proc folder with the new naming convention
    for proc_folder in proc_folders:
        source_path = os.path.join(sequence_path, proc_folder)
        
        # Extract crop name from proc folder name
        if proc_folder == "proc":
            # Default proc folder (no specific crop)
            crop_name = "full"
        elif proc_folder.startswith("proc_"):
            # proc_<crop_name> format
            crop_name = proc_folder[5:]  # Remove "proc_" prefix
        else:
            crop_name = proc_folder
        
        # Create destination folder name: sequence_name_crop_name
        dest_folder_name = f"{sequence_name}_{crop_name}"
        dest_base_path = os.path.join(dataset_procs_dir, dest_folder_name)
        dest_proc_path = os.path.join(dest_base_path, "proc")
        
        # Remove existing destination if it exists
        if os.path.exists(dest_base_path):
            print(f"Removing existing {dest_folder_name}")
            shutil.rmtree(dest_base_path)
        
        # Create the destination structure
        os.makedirs(dest_base_path, exist_ok=True)
        
        # Move the entire proc folder
        print(f"Moving {proc_folder} to {dest_folder_name}/proc")
        shutil.move(source_path, dest_proc_path)
        moved_folders.append(dest_folder_name)
    
    print(f"Moved {len(proc_folders)} proc folder(s) from {sequence_name} to procs directory")
    return moved_folders


def process_sequence(sequence, calib, beamsplitter, only_validate=False, crop_idx=None, process_all_crops=False):
    if only_validate:
        # For validation with multiple crops, validate each proc directory
        if process_all_crops:
            crops = get_crops_from_sequence(sequence)
            if crops:
                for i, crop in enumerate(crops):
                    crop_name = crop.get('name', f'crop{i}')
                    proc_dir = os.path.join(sequence, f"proc_{crop_name}")
                    if os.path.exists(proc_dir):
                        print(f"Validating {crop_name}...")
                        # Temporarily modify validate_sequence to check the specific proc dir
                        validate_sequence_with_proc_dir(sequence, f"proc_{crop_name}")
            else:
                validate_sequence(sequence)
        else:
            validate_sequence(sequence)
        return
    
    # only convert if the "proph_00051463_exported" does not exist
    if not os.path.exists(os.path.join(sequence, PROPH_EXP_DIR)):
        print('Converting', sequence)
        convert_raw_to_npy(os.path.join(sequence, PROPH_DIR))
    else:
        print('Skipping conversion, already exists:', os.path.join(sequence, PROPH_EXP_DIR))

    # Check if we should process all crops
    if process_all_crops:
        crops = get_crops_from_sequence(sequence)
        if crops:
            print(f"Found {len(crops)} crops in sequence {sequence}")
            for i, crop in enumerate(crops):
                crop_name = crop.get('name', f'crop{i}')
                print(f"\nProcessing crop '{crop_name}' ({i+1}/{len(crops)})...")
                # Process each crop with its index and name
                AlignedDataSaver(
                    seq_path=sequence,
                    calib_path=calib,
                    crop=beamsplitter,
                    crop_idx=i,
                    crop_name=crop_name
                )
                
                # Validate the processed crop
                validate_sequence_with_proc_dir(sequence, f"proc_{crop_name}")
        else:
            print(f"No crops found in sequence {sequence}, skipping (--all_crops specified but no crops found)")
    elif crop_idx is not None:
        # Process single crop only
        crops = get_crops_from_sequence(sequence)
        if crops and crop_idx < len(crops):
            crop_name = crops[crop_idx].get('name', f'crop{crop_idx}')
            print(f"Processing only crop '{crop_name}' (index {crop_idx})")
            AlignedDataSaver(
                seq_path=sequence,
                calib_path=calib,
                crop=beamsplitter,
                crop_idx=crop_idx,
                crop_name=crop_name
            )
            validate_sequence_with_proc_dir(sequence, f"proc_{crop_name}")
        else:
            print(f"Warning: crop_idx {crop_idx} specified but no valid crop found at that index")
    else:
        # Process entire sequence only when no crop options are specified
        print(f"Processing entire sequence (no crop options specified)")
        AlignedDataSaver(
            seq_path=sequence,
            calib_path=calib,
            crop=beamsplitter,
            crop_idx=None
        )
        validate_sequence(sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and directly rectify event coordinates.")
    parser.add_argument("--sequence", type=str, help="Path to the sequence directory.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("--calib", type=str, help="Path to the calibration directory.",
                        default="../../vme_research/calibrate/calibration_default")
    parser.add_argument("--beamsplitter", action='store_true', help="Enable beamsplitter mode (sets crop resolution to 720x720).")
    parser.add_argument("--validate", action='store_true', help="Only validate the sequence without processing.")
    parser.add_argument("--crop_idx", type=int, default=None, help="Index of specific crop to process (mutually exclusive with --all_crops).")
    parser.add_argument("--all_crops", action='store_true', help="Process all crops found in data.json, creating separate folders for each.")
    parser.add_argument("--sequential_naming", action='store_true', help="Rename moved folders to sequence_000000, sequence_000001, etc. instead of using crop names.")
    args = parser.parse_args()

    # Check for mutually exclusive arguments
    if args.crop_idx is not None and args.all_crops:
        parser.error("--crop_idx and --all_crops are mutually exclusive. Choose one or neither.")

    if args.sequence:
        process_sequence(args.sequence, args.calib, 
                         args.beamsplitter, args.validate, args.crop_idx, args.all_crops)
    elif args.dataset_dir:
        # Create procs folder at the dataset level
        dataset_procs_dir = os.path.join(args.dataset_dir, "procs")
        if not args.validate:  # Only create procs folder when processing, not validating
            os.makedirs(dataset_procs_dir, exist_ok=True)
            print(f"Created/Using procs directory at: {dataset_procs_dir}")
        
        all_moved_folders = []  # Collect all moved folders for sequential renaming
        
        for sequence in tqdm.tqdm(os.listdir(args.dataset_dir)):
            if not sequence.startswith("sequence_"):
                continue
            # skip if it's not a directory
            sequence_path = os.path.join(args.dataset_dir, sequence)
            if not os.path.isdir(sequence_path):
                print(f"Skipping non-directory: {sequence}")
                continue
            
            # Process the sequence
            process_sequence(sequence_path, args.calib, 
                             args.beamsplitter, args.validate, args.crop_idx, args.all_crops)
            
            # Move proc folders to dataset procs directory (only when not validating)
            if not args.validate:
                moved_folders = move_proc_folders_to_dataset(sequence_path, dataset_procs_dir, args.all_crops, args.crop_idx)
                all_moved_folders.extend(moved_folders)
        
        # Rename folders sequentially if requested
        if not args.validate and args.sequential_naming and all_moved_folders:
            print(f"\nRenaming {len(all_moved_folders)} folders sequentially...")
            # Sort folders to ensure consistent ordering
            all_moved_folders.sort()
            
            for idx, folder_name in enumerate(all_moved_folders):
                old_path = os.path.join(dataset_procs_dir, folder_name)
                new_name = f"sequence_{idx:06d}"
                new_path = os.path.join(dataset_procs_dir, new_name)
                
                # Remove destination if it exists
                if os.path.exists(new_path):
                    print(f"Removing existing {new_name}")
                    shutil.rmtree(new_path)
                
                # Rename the folder
                print(f"Renaming {folder_name} to {new_name}")
                shutil.move(old_path, new_path)
            
            print(f"Renamed all folders to sequential format (sequence_000000 to sequence_{len(all_moved_folders)-1:06d})")

