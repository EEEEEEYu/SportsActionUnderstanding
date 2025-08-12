import numpy as np
from numba import jit


def render(events, image):
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
