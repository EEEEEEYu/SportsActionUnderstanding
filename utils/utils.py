import os
import json
import glob
import cv2
import numpy as np
from numba import jit

from scipy.spatial import cKDTree


def load_events(sequence_path):
    # load events
    events_xy = np.load(os.path.join(sequence_path, 'proc', 'events', 'events_xy.npy')).astype(np.uint16)
    events_t  = np.load(os.path.join(sequence_path, 'proc', 'events', 'events_t.npy')).astype(np.int64)
    events_p  = np.load(os.path.join(sequence_path, 'proc', 'events', 'events_p.npy')).astype(np.uint8)
    events = np.concatenate([events_xy, events_t[..., np.newaxis], events_p[..., np.newaxis]], axis=-1)
    # load metadata
    with open(os.path.join(sequence_path, 'proc', 'events', 'data.json')) as f:
        events_metadata = json.load(f)
        res = events_metadata['append_fields']['res']
        K = np.array(events_metadata['append_fields']['K']).reshape(3, 3)
        dist = np.array(events_metadata['append_fields']['dist'])
    return events, events_t, res, K, dist


def load_frames(sequence_path):
    flir_t = np.load(os.path.join(sequence_path, 'proc', 'flir', 'flir_t.npy'))
    flir_files = sorted(glob.glob(os.path.join(sequence_path, 'proc', 'flir', 'frame', '*.png')))
    # load flir files
    flir_frames = []
    for f in flir_files:
        # Load PNG file
        frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"Warning: Could not load PNG file {f}")
            continue
        flir_frames.append(frame)
    # load metadata
    with open(os.path.join(sequence_path, 'proc', 'flir', 'data.json')) as f:
        flir_metadata = json.load(f)
        res = flir_metadata['append_fields']['res']
        K = np.array(flir_metadata['append_fields']['K']).reshape(3, 3)
        dist = np.array(flir_metadata['append_fields']['dist'])

    return flir_frames, flir_t, res, K, dist


def render(events, events_res):
    image = np.zeros((events_res[0], events_res[1], 3), dtype=np.uint8)
    if len(events) == 0:
        return image
    x,y,p = events[:,0].astype(np.int32), events[:,1].astype(np.int32), events[:,3] > 0
    height, width = image.shape[:2]
    x = np.clip(x, 0, width-1)
    y = np.clip(y, 0, height-1)
    image[y[p], x[p], 2] = 255
    image[y[p==0], x[p==0], 0] = 255
    return image


def binary_search_time(dset_t, x_time, l=None, r=None):
    l = 0 if l is None else l; r = len(dset_t) - 1 if r is None else r
    while l <= r:
        mid = l + (r - l) // 2
        midval = dset_t[mid]
        if midval == x_time: return mid
        elif midval < x_time: l = mid + 1
        else: r = mid - 1
    return l


def get_events_between(all_events, all_ts, start_time, end_time):
    start_idx = binary_search_time(all_ts, start_time)
    end_idx = binary_search_time(all_ts, end_time, l=start_idx)
    return all_events[start_idx:end_idx, ...]


def get_frame_between(all_frames, all_ts, start_time, end_time):
    # there can be multiple frames that overlap the time window,
    # want the frame that is closest to the start of the time window
    start_idx = binary_search_time(all_ts, start_time)
    if start_idx > len(all_frames) - 1:
        start_idx = len(all_frames) - 1
    return all_frames[start_idx]


@jit(nopython=True)
def event_count_image(image, x, y, p):
    image[...] = 0
    for i in range(y.shape[0]):
        image[y[i], x[i]] += 1
    return image


def undistort_event_count_image(event_count_image, K, dist, res=(720,720)):
    h, w = res
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(event_count_image, map1, map2, cv2.INTER_LINEAR)


def undistort_event_xy_forward(events, K, dist, round=False, res=(720,720)):
    # Format points for OpenCV function: (N, 1, 2)
    points_to_undistort = np.stack((events[:, 0], events[:, 1]), axis=-1).astype(np.float32)[:, np.newaxis, :]
    # Undistort points. The new camera matrix P=K gives pixel coordinates.
    rectified_points = cv2.undistortPoints(points_to_undistort, K, dist, P=K)
    # Reshape back to (N, 2)
    rectified_coords = rectified_points.reshape(-1, 2)
    # Round to closest int if desired
    if round:
        rectified_coords = np.round(rectified_coords).astype(np.uint16)
    # Set the undistorted coordinates back to the events
    events[:, :2] = rectified_coords
    return events


def undistort_events_backward(events, K, dist, res=(720,720)):
    # h, w: Sensor height and width (e.g., 720, 720)
    h, w = res

    print("Performing one-time setup for event undistortion...")

    # 1. Calculate the inverse map from undistorted to distorted space
    # map1 holds the x_src coords, map2 holds the y_src coords
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)

    # 2. Create the set of "ideal" source points
    # These are the (x, y) source locations that correspond to integer grid
    # points in the destination image.
    # We reshape them into a (H*W, 2) array for the k-d tree.
    ideal_source_points = np.stack((map1.ravel(), map2.ravel()), axis=-1)

    # 3. Build the k-d tree from these ideal points. This is the search structure.
    # This might take a second or two, but you only do it once.
    kdtree = cKDTree(ideal_source_points)
    print("Setup complete. Ready to process events.")

    # 1. For each of your actual events, query the k-d tree to find the index
    #    of the CLOSEST ideal source point.
    #    `distance` will be the distance, `index` will be the row in ideal_source_points.
    distance, index = kdtree.query(events[:, :2], k=1)

    # 2. The 'index' corresponds to the flattened (H*W) grid. We need to
    #    convert this flat index back into a 2D coordinate (u, v).
    #    `np.unravel_index` is perfect for this. The result is (row, col) i.e. (y, x)
    rectified_rows, rectified_cols = np.unravel_index(index, (h, w))

    # 3. Stack them back into the desired (N, 2) format for (x, y) coordinates.
    #    This is your final result.
    rectified_coords_int = np.stack((rectified_cols, rectified_rows), axis=-1)

    print("Processing complete")

    events[:, :2] = rectified_coords_int

    return rectified_coords_int
