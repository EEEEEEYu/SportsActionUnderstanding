from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm

class Preprocessor(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError("")
        self.sequence_list = os.listdir(dataset_dir)


    @staticmethod
    def test_freq(data):
        # Build frequency map of each integer
        unique_vals, counts = np.unique(data, return_counts=True)
        freq_map = dict(zip(unique_vals, counts))

        # Query by single integer
        def query_single(val):
            return freq_map.get(val, 0)

        # Query by integer range [low, high] inclusive
        def query_range(low, high):
            return sum(count for val, count in freq_map.items() if low <= val <= high)

        print(query_single(240), query_single(0), query_single(719))   # 3
        print(query_range(0, 719), query_range(720, 65535)) # counts for 2, 3, 5, 6, 8 → 8
        

    def __len__(self):
        return len(self.sequence_list[:16])

    def __getitem__(self, idx):
        sequence_name = self.sequence_list[idx]
        events_p_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_p.npy')
        events_t_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_t.npy')
        events_xy_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'events', 'events_xy.npy')
        frame_t_path = os.path.join(self.dataset_dir, sequence_name, 'proc', 'flir', 't.npy')

        events_p_all = np.load(events_p_path, mmap_mode='r').astype(np.uint8)
        events_t_all = np.load(events_t_path, mmap_mode='r').astype(np.int64)
        events_xy_all = np.load(events_xy_path, mmap_mode='r').astype(np.uint16)

        # TODO: fix this hard truncate
        mask = (events_xy_all[:, 0] >= 0) & (events_xy_all[:, 0] < 720) & \
            (events_xy_all[:, 1] >= 0) & (events_xy_all[:, 1] < 720)
        events_xy_all = events_xy_all[mask]
        events_p_all = events_p_all[mask]
        events_t_all = events_t_all[mask]

        """self.test_freq(events_xy_all)
        print(f"events_p_all.dtype: {events_p_all.dtype} events_t_all.dtype: {events_t_all.dtype} events_xy_all.dtype: {events_xy_all.dtype}")
        print(f"events_t_all.shape: {events_t_all.shape} first event_t: {events_t_all[0]} last event_t: {events_t_all[-1]}")
        print(f"events_xy_all.shape: {events_xy_all.shape}")
        print(f"event x range: {events_xy_all[:, 0].min()} {events_xy_all[:, 0].max()} event_y range: {events_xy_all[:, 1].min()} {events_xy_all[:, 1].max()}")"""
        frame_t = np.load(frame_t_path)


        # Ensure time array is sorted (should already be)
        sort_idx = np.argsort(events_t_all)
        events_t_all = events_t_all[sort_idx]
        events_xy_all = events_xy_all[sort_idx]
        events_p_all = events_p_all[sort_idx]

        events_p_sliced = []
        events_t_sliced = []
        events_xy_sliced = []

        for i in tqdm(range(len(frame_t) - 1)):
            t0 = frame_t[i]
            t1 = frame_t[i + 1]
            event_idx_at_t0 = np.searchsorted(events_t_all, t0)
            event_idx_at_t1 = np.searchsorted(events_t_all, t1)

            if event_idx_at_t1 <= event_idx_at_t0:
                continue  # no events in this frame interval

            events_p_sliced.append(events_p_all[event_idx_at_t0: event_idx_at_t1])
            events_t_sliced.append(events_t_all[event_idx_at_t0: event_idx_at_t1])
            events_xy_sliced.append(events_xy_all[event_idx_at_t0: event_idx_at_t1])

        return events_t_sliced, events_xy_sliced, events_p_sliced, sequence_name[-6:]

