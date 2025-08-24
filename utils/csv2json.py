#!/usr/bin/env python3
import csv
import json
import sys
import os
import numpy as np

def find_closest_index(times_array, target_time):
    """Find the index of the closest time in the array to the target time."""
    return np.argmin(np.abs(times_array - target_time))

def convert_csv_to_json(csv_file='markers.csv', output_file=None):
    flir_times = []
    proph_times = []
    crops = []
    
    # Load the FLIR time array
    sequence_dir = os.path.dirname(csv_file)
    flir_time_file = os.path.join(sequence_dir, 'flir_23604512/t.npy')
    try:
        flir_frame_times = np.load(flir_time_file)
    except FileNotFoundError:
        print(f"Warning: Could not find {flir_time_file}. Using default indices.")
        flir_frame_times = None
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            viewer_name = row['viewer_name']
            frame_time = float(row['frame_time'])
            
            if viewer_name == 'flir':
                flir_times.append(frame_time)
            elif viewer_name == 'proph':
                proph_times.append(frame_time)

    crop_idx = 0
    for i in range(0, len(flir_times)-1, 2):
        flir_start = flir_times[i] if i < len(flir_times) else 0
        flir_end = flir_times[i+1] if i+1 < len(flir_times) else 0

        # Find the closest indices in the FLIR frame times array
        if flir_frame_times is not None:
            start_index = find_closest_index(flir_frame_times, flir_start)
            end_index = find_closest_index(flir_frame_times, flir_end)
        else:
            start_index = 0
            end_index = 0
        
        crops.append({
            "name": f"crop{crop_idx}",
            "start_index": int(start_index),
            "end_index": int(end_index),
            "flir_start_time": flir_start,
            "flir_end_time": flir_end,
            "event_start_time": proph_times[i] if i < len(proph_times) else 0,
            "event_end_time": proph_times[i+1] if i+1 < len(proph_times) else 0,
            "zero_rectangles": []
        })
        crop_idx += 1

    if output_file:
        with open(output_file, 'w') as f:
            json.dump({"crops": crops}, f, indent=2)
        print(f"Converted {csv_file} to {output_file}")
    else:
        output_file = os.path.join(os.path.dirname(csv_file), 'crop.json')
        with open(output_file, 'w') as f:
            json.dump({"crops": crops}, f, indent=2)
        print(f"Converted {csv_file} to {output_file}")

    return {"crops": crops}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv2json.py <csv_file> [output_file]")
        print("If output_file is not specified, crop.json will be created in the same directory as the CSV file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    result = convert_csv_to_json(input_file, output_file)
    print(json.dumps(result, indent=2))