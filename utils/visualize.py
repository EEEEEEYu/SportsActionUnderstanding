import argparse
import os
import cv2
import numpy as np

import utils


def view_processed_data(sequence_path, window_size_ms=100):
    # load frames
    frames, flir_t, flir_res, flir_K, flir_dist = utils.load_frames(sequence_path)
    num_frames = len(frames)
    print(f"Loaded {num_frames} FLIR frames")
    # load events
    events, events_t, events_res, events_K, events_dist = utils.load_events(sequence_path)

    # To do the forward mapping
    events = utils.undistort_event_xy_forward(events, events_K, events_dist, round=True, res=events_res)

    # To do the backward mapping
    #events = utils.undistort_events_backward(events, events_K, events_dist, events_res)

    print(f"Controls: 'n' for next window, 'p' for previous window, 'space' for play/pause, 'q' to quit, 'r' to reset")
    print(f"Window size: {window_size_ms}ms")

    # Convert window size from ms to microseconds (assuming timestamps are in microseconds)
    window_size_us = window_size_ms * 1000
    
    # Initialize current time at the first FLIR frame if events started earlier
    start_timestamp = flir_t[0]
    end_timestamp = flir_t[-1]
    current_time = flir_t[0]
    playing = False

    print("START", start_timestamp, "END", end_timestamp, events_t[0], events_t[-1])

    while True:
        # Calculate the time window
        window_start = current_time
        window_end = current_time + window_size_us
        
        # Get events in the current time window
        events_in_window = utils.get_events_between(events, events_t, window_start, window_end)
        print(f"Events in window ({window_start} - {window_end}): {len(events_in_window)}")

        # create event frame
        event_frame = utils.render(events_in_window, events_res)
        
        # To do backward mapping of the event image with interpolation
        #event_frame = utils.undistort_event_count_image(event_frame, events_K, events_dist, events_res)

        # get flir frame
        flir_frame = utils.get_frame_between(frames, flir_t, window_start, window_end)
        
        # --- Create visualizations ---
        # Create an overlay by blending the two images
        overlay = cv2.addWeighted(flir_frame, 0.6, event_frame, 0.8, 0)

        # Create a side-by-side view for comparison
        side_by_side = np.hstack([flir_frame, event_frame])

        # --- Display the images ---
        # Add text to show timing information
        time_info = f"Time: {(current_time - start_timestamp) / 1e6:.3f}s | "
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
            
        else:
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely when paused

        if key == ord('q'):
            break
        elif key == ord('n'):  # Next window
            old_time = current_time
            # increment current_time
            current_time = min(current_time + window_size_us, end_timestamp)
            if current_time > end_timestamp:
                current_time = start_timestamp  # Loop back to beginning
            print(f"[TIME] Advanced from {old_time:.2f} to {current_time:.2f} us ({(current_time-old_time)/1000:.1f} ms forward)")
        elif key == ord('p'):  # Previous window
            old_time = current_time
            # decrement current_time
            current_time = max(current_time - window_size_us, start_timestamp)
            if current_time < start_timestamp:
                current_time = end_timestamp  # Loop back to end
            print(f"[TIME] Moved back from {old_time:.2f} to {current_time:.2f} us ({(old_time-current_time)/1000:.1f} ms backward)")
        elif key == ord(' '):  # Space bar for play/pause
            playing = not playing
            print(f"[PLAYBACK] {'Playing' if playing else 'Paused'}")
        elif key == ord('r'):  # Reset to beginning
            print(f"[RESET] Resetting to overlap start time {start_timestamp:.2f} us")
            # reset current_time
            current_time = start_timestamp

    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize processed and aligned FLIR and event camera data."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Path to the sequence directory containing 'proc/flir' and 'proc/events' folders."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Time window size in milliseconds for event accumulation (default: 100ms)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.sequence):
        print(f"Error: The provided path is not a valid directory: {args.sequence}")
    else:
        view_processed_data(args.sequence, window_size_ms=args.window_size)
