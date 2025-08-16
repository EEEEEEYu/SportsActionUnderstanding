from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm

class Preprocessor(Dataset):
    def __init__(self, dataset_dir, accumulation_interval_ms=150):
        self.dataset_dir = dataset_dir
        self.accumulation_interval_ms = accumulation_interval_ms

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
        self.sequence_list = os.listdir(dataset_dir)
    
    def __len__(self):
        return len(self.sequence_list[:16])

    def __getitem__(self, idx):
        sequence_name = self.sequence_list[idx]
        events_p_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_p.npy')
        events_t_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_t.npy')
        events_xy_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_xy.npy')

        events_p_all = np.load(events_p_path, mmap_mode='r').astype(np.uint8)
        events_t_all = np.load(events_t_path, mmap_mode='r').astype(np.int64)
        events_xy_all = np.load(events_xy_path, mmap_mode='r').astype(np.uint16)

        # TODO: fix this hard truncate
        """mask = (events_xy_all[:, 0] >= 0) & (events_xy_all[:, 0] < 720) & \
            (events_xy_all[:, 1] >= 0) & (events_xy_all[:, 1] < 720)
        events_xy_all = events_xy_all[mask]
        events_p_all = events_p_all[mask]
        events_t_all = events_t_all[mask]"""

        """self.test_freq(events_xy_all)
        print(f"events_p_all.dtype: {events_p_all.dtype} events_t_all.dtype: {events_t_all.dtype} events_xy_all.dtype: {events_xy_all.dtype}")
        print(f"events_t_all.shape: {events_t_all.shape} first event_t: {events_t_all[0]} last event_t: {events_t_all[-1]}")
        print(f"events_xy_all.shape: {events_xy_all.shape}")
        print(f"event x range: {events_xy_all[:, 0].min()} {events_xy_all[:, 0].max()} event_y range: {events_xy_all[:, 1].min()} {events_xy_all[:, 1].max()}")"""

        # Ensure time array is sorted (should already be)
        sort_idx = np.argsort(events_t_all)
        events_t_all = events_t_all[sort_idx]
        events_xy_all = events_xy_all[sort_idx]
        events_p_all = events_p_all[sort_idx]

        events_p_sliced = []
        events_t_sliced = []
        events_xy_sliced = []

        # slicing by accumulation interval
        interval_us = self.accumulation_interval_ms * 1000.0
        t_start = events_t_all[0]
        t_end   = events_t_all[-1]

        events_p_sliced = []
        events_t_sliced = []
        events_xy_sliced = []

        cur_t0 = t_start
        while cur_t0 < t_end:
            cur_t1 = cur_t0 + interval_us
            idx0 = np.searchsorted(events_t_all, cur_t0, side='left')
            idx1 = np.searchsorted(events_t_all, cur_t1, side='left')

            if idx1 > idx0:  # non-empty slice
                events_p_sliced.append(events_p_all[idx0:idx1])
                events_t_sliced.append(events_t_all[idx0:idx1])
                events_xy_sliced.append(events_xy_all[idx0:idx1])

            cur_t0 = cur_t1  # move to next interval

        return events_t_sliced, events_xy_sliced, events_p_sliced, sequence_name[-6:]

