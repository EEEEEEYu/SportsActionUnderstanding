import argparse
import glob
import os
import cv2
import numpy as np

from utils import render, get_events_between


def view_processed_data(sequence_path, window_size_ms=100, time_offset_ms=0):
    """
    Loads and displays processed FLIR and event data from a sequence directory.

    Args:
        sequence_path (str): The path to the main sequence directory which
                             contains the 'proc_flir' and 'proc_events' subfolders.
        window_size_ms (int): Time window size in milliseconds for event accumulation.
        time_offset_ms (float): Manual time offset in milliseconds to adjust event timing relative to FLIR.
                                Positive values delay events, negative values advance events.
    """
    # --- 1. Define paths and find data files ---
    flir_dir = os.path.join(sequence_path, 'proc', 'flir')
    if not os.path.isdir(flir_dir):
        print(f"Error: Could not find 'proc/flir' directory in '{sequence_path}'")
        print("Please run the processing script first.")
        return
    # Get a sorted list of the processed FLIR frames
    # Check for both NPY and PNG files
    flir_files_png = sorted(glob.glob(os.path.join(flir_dir, '*.png')))
        
    # Use PNG files if available, otherwise NPY
    if flir_files_png:
        flir_files = flir_files_png
    
    if not flir_files:
        print(f"Error: No FLIR frame files found in '{flir_dir}'")
        return
        
    flir_t = np.load(os.path.join(flir_dir, 't.npy'))

    # load flir files
    flir_frames = []
    for f in flir_files:
        # Load PNG file
        frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"Warning: Could not load PNG file {f}")
            continue
        flir_frames.append(frame)
    num_frames = len(flir_frames)

    # load events
    events_xy = np.load(os.path.join(sequence_path, 'proc', 'events', 'events_xy.npy')).astype(np.uint16)
    events_t  = np.load(os.path.join(sequence_path, 'proc', 'events', 'events_t.npy'))
    events_p  = np.load(os.path.join(sequence_path, 'proc', 'events', 'events_p.npy'))
    print(events_xy.dtype, events_t.dtype, events_p.dtype)
    print(np.min(events_xy, axis=0), np.max(events_xy, axis=0))

    all_events = np.concatenate([events_xy, events_t[..., np.newaxis], events_p[..., np.newaxis]], axis=-1)

    print(f"Controls: 'n' for next window, 'p' for previous window, 'space' for play/pause, 'q' to quit, 'r' to reset")
    print(f"Window size: {window_size_ms}ms, Time offset: {time_offset_ms}ms")
    
    # Convert window size from ms to microseconds (assuming timestamps are in microseconds)
    window_size_us = window_size_ms * 1000
    
    # Initialize current time at the first FLIR frame if events started earlier
    start_timestamp = flir_t[0]
    end_timestamp = flir_t[-1]
    current_time = flir_t[0]
    current_flir_idx = 0
    previous_flir_idx = -1
    playing = False

    print("START", start_timestamp, "END", end_timestamp, events_t[0], events_t[-1])

    # Debug: Print FLIR frame timestamps
    print(f"\nFLIR frame timestamps (first 10):")
    for i in range(min(10, len(flir_t))):
        print(f"  Frame {i}: {flir_t[i]:.2f} us ({flir_t[i]/1e6:.6f} s)")

    # --- 2. Main display loop ---
    while True:
        # Calculate the time window
        window_start = current_time
        window_end = current_time + window_size_us
        
        # Get events in the current time window
        events_in_window = get_events_between(all_events, events_t, window_start, window_end)
        print(f"Events in window ({window_start} - {window_end}): {len(events_in_window)}")

        # Get flir frame
        flir_frame = flir_frames[current_flir_idx]
        flir_height, flir_width = flir_frame.shape[:2]

        # Create event frame with same dimensions as FLIR frame
        event_frame = np.zeros((flir_height, flir_width, 3), dtype=np.uint8)
        if len(events_in_window) > 0:
            event_frame = render(events_in_window, event_frame)
        
        # --- Create visualizations ---
        # Create an overlay by blending the two images
        overlay = cv2.addWeighted(flir_frame, 0.6, event_frame, 0.8, 0)

        # Create a side-by-side view for comparison
        side_by_side = np.hstack([flir_frame, event_frame])

        # --- Display the images ---
        # Add text to show timing information
        time_info = f"Time: {(current_time - start_timestamp) / 1e6:.3f}s | "
        time_info += f"FLIR Frame: {current_flir_idx + 1}/{num_frames} | "
        time_info += f"Events: {len(events_in_window)} | "
        time_info += f"{'Playing' if playing else 'Paused'}"
        
        cv2.putText(side_by_side, time_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Side-by-side View (FLIR | Events)", side_by_side)
        cv2.imshow("Overlay View", overlay)

        # --- Handle user input ---
        if playing:
            key = cv2.waitKey(30) & 0xFF  # 30ms delay for playback
            # Auto-advance to next window
            if key == 255:  # No key pressed
                current_time += window_size_us
                if current_time > end_timestamp:
                    current_time = start_timestamp  # Loop back to beginning
                    current_flir_idx = 0
                while current_flir_idx < num_frames - 1 and current_time >= flir_t[current_flir_idx + 1]:
                    current_flir_idx += 1
            
        else:
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely when paused

        if key == ord('q'):
            break
        elif key == ord('n'):  # Next window
            old_time = current_time
            current_time = min(current_time + window_size_us, end_timestamp)
            # increment current_flir_idx
            while current_flir_idx < num_frames - 1 and current_time >= flir_t[current_flir_idx + 1]:
                if current_time > end_timestamp:
                    current_time = start_timestamp  # Loop back to beginning
                    current_flir_idx = 0
                else:
                    current_flir_idx += 1
            print(f"[TIME] Advanced from {old_time:.2f} to {current_time:.2f} us ({(current_time-old_time)/1000:.1f} ms forward)")
        elif key == ord('p'):  # Previous window
            old_time = current_time
            current_time = max(current_time - window_size_us, start_timestamp)
            # decrement current_flir_idx
            while current_flir_idx > 0 and current_time < flir_t[current_flir_idx]:
                if current_time < start_timestamp:
                    current_time = end_timestamp  # Loop back to end
                    current_flir_idx = num_frames - 1
                else:
                    current_flir_idx -= 1
            print(f"[TIME] Moved back from {old_time:.2f} to {current_time:.2f} us ({(old_time-current_time)/1000:.1f} ms backward)")
        elif key == ord(' '):  # Space bar for play/pause
            playing = not playing
            print(f"[PLAYBACK] {'Playing' if playing else 'Paused'}")
        elif key == ord('r'):  # Reset to beginning
            print(f"[RESET] Resetting to overlap start time {start_timestamp:.2f} us")
            current_time = start_timestamp
            # Find the correct FLIR frame for the start time
            current_flir_idx = 0
            while current_flir_idx < num_frames - 1 and start_timestamp >= flir_t[current_flir_idx + 1]:
                current_flir_idx += 1

    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize processed and aligned FLIR and event camera data."
    )
    parser.add_argument(
        "sequence_path",
        type=str,
        help="Path to the sequence directory containing 'proc/flir' and 'proc/events' folders."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Time window size in milliseconds for event accumulation (default: 100ms)"
    )
    parser.add_argument(
        "--time_offset",
        type=float,
        default=0,
        help="Time offset in milliseconds to adjust event timing relative to FLIR (default: 0ms). Positive delays events, negative advances events."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.sequence_path):
        print(f"Error: The provided path is not a valid directory: {args.sequence_path}")
    else:
        view_processed_data(args.sequence_path, window_size_ms=args.window_size, time_offset_ms=args.time_offset)
