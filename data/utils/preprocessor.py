from torch.utils.data import Dataset

import os
import numpy as np

class Preprocessor(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError("")
        self.sequence_list = os.listdir(dataset_dir)
        

    def __len__(self):
        return len(self.sequence_list)

    def __get_item__(self, idx):
        sequence_name = self.sequence_list[idx]
        events_p_path = os.path.join(self.dataset_dir, sequence_name, 'events', 'events_p.npy')
        events_t_path = os.path.join(self.dataset_dir, sequence_name, 'events', 'events_t.npy')
        events_xy_path = os.path.join(self.dataset_dir, sequence_name, 'events', 'events_xy.npy')
        frame_t_path = os.path.join(self.dataset_dir, sequence_name, 'flir', 't.npy')

        events_p_all = np.load(events_p_path)
        events_t_all = np.load(events_t_path)
        events_xy_all = np.load(events_xy_path)
        frame_t = np.load(frame_t_path)

        events_p_sliced = []
        events_t_sliced = []
        events_xy_sliced = []

        for i in range(len(frame_t) - 1):
            t0 = frame_t[i]
            t1 = frame_t[i + 1]
            event_idx_at_t0 = np.searchsorted(events_t_all, t0)
            event_idx_at_t1 = np.searchsorted(events_t_all, t1)

            events_p_sliced.append(events_p_all[event_idx_at_t0: event_idx_at_t1])
            events_t_sliced.append(events_t_all[event_idx_at_t0: event_idx_at_t1])
            events_xy_sliced.append(events_xy_all[event_idx_at_t0: event_idx_at_t1])

        return events_t_sliced, events_xy_sliced, events_p_sliced

