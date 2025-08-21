import os
from numba import njit
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.utils.preprocessor import Preprocessor
import matplotlib.pyplot as plt

CLS_NAME_TO_INT = {f"{i:06d}": i for i in range(16)}

class Hats(Dataset):
    def __init__(self, dataset_dir,
                 width: int,
                 height: int,
                 tau: float,
                 R: int,
                 K: int,
                 cache_root: str,
                 purpose='train',
                 use_cache=True):
        assert purpose in ['train', 'test'], "Split must be either 'train' or 'test'."

        self.dataset_dir = dataset_dir
        self.preprocessor = Preprocessor(dataset_dir=dataset_dir)
        self.purpose = purpose
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.width = width
        self.height = height
        self.tau = tau
        self.R = R
        self.K = K
        
        self.cell_width = (width // K)
        self.cell_height = (height // K)
        self.n_cells = self.cell_width * self.cell_height
        self.n_polarities = 2

        # Initialize HATS
        self.hats = HATS_representation(
            width=self.width,
            height=self.height,
            tau=self.tau,
            R=self.R,
            K=self.K
        )

    def __len__(self):
        return len(self.preprocessor)
    
    def __getitem__(self, idx):
        events_t_list, events_xy_list, events_p_list, class_name, seq_folder = self.preprocessor[idx]
        print("Processing sequence:", seq_folder)

        if self.use_cache:
            rel_seq_path = os.path.relpath(seq_folder)
            cache_dir_path = os.path.join(self.cache_root, rel_seq_path)
            cached_hats_path = os.path.join(cache_dir_path, f"HATS_{self.tau}_{self.R}_{self.K}.pt")

            if os.path.exists(cached_hats_path):
                data = torch.load(cached_hats_path)
                print(f"Loaded cached HATS from: {cached_hats_path}")
                return data, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)

        else:
            cached_hats_path = None

        
        hats_list = []

        counter = 0
        for i in tqdm(range(len(events_xy_list)), desc="Processing event frames"):
            evt_xy = np.array(events_xy_list[i])
            evt_t = np.array(events_t_list[i]).reshape(-1, 1)
            evt_p = np.array(events_p_list[i]).reshape(-1, 1)
            evt_tensor = np.concatenate([evt_xy, evt_t, evt_p], axis=1)

            if evt_tensor.size == 0:
                hats_list.append(np.zeros((self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1)))
                continue
            
            self.hats.process_all(evt_tensor)
            hats_list.append(self.hats.histograms)

            # fig, axes = plt.subplots(9, 9, figsize=(12, 12))
            # axes = axes.flatten()
            # for j in range(81):
            #     axes[j].imshow(self.hats.histograms[j, 0], cmap='hot')
            #     axes[j].axis("off")
            
            # plt.subplots_adjust(wspace=0.05, hspace=0.05)
            # plt.savefig(f"./{class_name}_{counter}_hats.png", dpi=300, bbox_inches='tight')
            # plt.close(fig)
            counter += 1
            if counter == 2:
                break

            self.hats.reset()

        print(len(hats_list))
        print(np.stack(hats_list).shape)
        hats_list = np.stack(hats_list).squeeze(0)  # [num_frames, n_cells, n_polarities, (2R+1), (2R+1)]

        print(f"Computed HATS for sequence {seq_folder}, shape: {hats_list.shape}")

        if self.use_cache:
            if not os.path.exists(cache_dir_path):
                os.makedirs(cache_dir_path, exist_ok=True)
            torch.save(hats_list, cached_hats_path)
            print(f"Saved HATS to cache: {cached_hats_path}")

        return hats_list, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)
        
class HATS_representation:
    def __init__(self, width, height, tau, R, K):

        self.tau = tau
        self.R = R
        self.K = K

        self.cell_width = (width // K)
        self.cell_height = (height // K)
        self.n_cells = self.cell_width * self.cell_height
        self.n_polarities = 2
        self.index = {0: 0, 1: 1}

        self.get_cell = get_pixel_cell_partition_matrix(width, height, K)

        self.reset()

    def reset(self):
        self.histogram = np.zeros((self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1), dtype=np.float32)
        self.event_counter = np.zeros((self.n_cells, self.n_polarities), dtype=np.int32)
        self.cell_memory = np.empty([self.n_cells, self.n_polarities], dtype=object)
        for i in range(self.n_cells):
            for j in range(self.n_polarities):
                self.cell_memory[i, j] = None

    def process(self, event):
        cell = (int(event[1]) // self.K) * self.cell_width + (int(event[0]) // self.K)
        polarity_index = self.index[int(event[3])]

        if self.cell_memory[cell, polarity_index] is None:
            self.cell_memory[cell, polarity_index] = event[np.newaxis, :]
        else:
            self.cell_memory[cell, polarity_index] = np.vstack((self.cell_memory[cell, polarity_index], event))

        time_surface = compute_local_memory_time_surface(
            event, self.cell_memory[cell, polarity_index], self.R, self.tau
        )

        self.histogram[cell, polarity_index, :, :] += time_surface

        self.event_counter[cell, polarity_index] += 1

    def process_all(self, events):
        for event in tqdm(events, desc="Processing events"):
            self.process(event)

        self.histograms = normalize(self.histogram, self.event_counter)

def normalize(histograms: np.ndarray, event_counter: np.ndarray) -> np.ndarray:
    counts = np.clip(event_counter, 1e-6, None)[:, :, np.newaxis, np.newaxis]
    return histograms / counts

@njit(cache=True, fastmath=True)
def compute_local_memory_time_surface(event_i: np.ndarray,
                                      filtered_memory: np.ndarray,
                                      R: int,
                                      tau: float) -> np.ndarray:
    # center/event
    x_i = int(event_i[0])
    y_i = int(event_i[1])
    t_i = event_i[2]

    # past events (ints for indices, float for time)
    xs = filtered_memory[:, 0].astype(np.int32)
    ys = filtered_memory[:, 1].astype(np.int32)
    ts = filtered_memory[:, 2]  # assume microseconds

    # spatial mask to keep indices in-bounds
    x0 = x_i - R; x1 = x_i + R
    y0 = y_i - R; y1 = y_i + R

    size = 2*R + 1

    # temporal weights
    delta_t = (t_i - ts) / 1.0e6  # -> seconds
    values = np.exp(-delta_t / tau).astype(np.float32)

    # flatten (y,x) -> linear index and scatter-add with bincount
    idx = (ys - y0) * size + (xs - x0)
    out_flat = np.bincount(idx, weights=values, minlength=size*size)

    return out_flat.reshape(size, size).astype(np.float32)


def get_pixel_cell_partition_matrix(width: int, height: int, K: int, device=None) -> np.ndarray:
    cell_width = width // K

    x = np.arange(width, dtype=np.int32)
    y = np.arange(height, dtype=np.int32)

    cell_col_idx = x // K
    cell_row_idx = y // K

    matrix = cell_col_idx[np.newaxis, :] * cell_width + cell_row_idx[:, np.newaxis]

    return matrix.astype(np.int32)

def main():
    import matplotlib.pyplot as plt
    import time

    dataset_dir = '/fs/nexus-scratch/tuxunlu/git/SportsActionUnderstanding/procs'

    dataset = Hats(dataset_dir,
                   width=720, height=720,
                   tau=0.5,
                   R=80,
                   K=80,
                   cache_root="./cache/",
                   use_cache=False,
                   purpose="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dataloader_iter = iter(dataloader)

    for i in range(len(dataloader)):
        start_time = time.perf_counter()
        frames, cls_name = next(dataloader_iter)
        proc_time = time.perf_counter() - start_time
        print(f"Class: {cls_name}, Frames: {frames.shape}")
        print(f"Process time: {proc_time:.4f} s")

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
