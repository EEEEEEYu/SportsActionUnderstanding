import os
from numba import njit, prange
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data.utils.preprocessor import Preprocessor
import matplotlib.pyplot as plt  # (kept unused for speed)

CLS_NAME_TO_INT = {f"{i:06d}": i for i in range(16)}

class HatsFast(Dataset):
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
        
        self.cell_width = (width // K)   # number of cells horizontally
        self.cell_height = (height // K) # number of cells vertically
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

        cache_dir_path = None
        if self.use_cache:
            rel_seq_path = os.path.relpath(seq_folder)
            cache_dir_path = os.path.join(self.cache_root, rel_seq_path)
            cached_hats_path = os.path.join(cache_dir_path, f"HATS_{self.tau}_{self.R}_{self.K}.npy")

            if os.path.exists(cached_hats_path):
                data = np.load(cached_hats_path)
                print(f"Loaded cached HATS from: {cached_hats_path}")
                return torch.from_numpy(data), torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)
        else:
            cached_hats_path = None

        # -------- preallocate --------
        num_frames = len(events_xy_list)
        out = np.zeros(
            (num_frames, self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1),
            dtype=np.float32
        )

        for i in tqdm(range(num_frames), desc="Processing event frames"):
            if len(events_xy_list[i]) == 0:
                # already zeros
                continue

            # Build events array once (int32/int8 for coords/labels, float64 for time)
            evt_xy = np.asarray(events_xy_list[i], dtype=np.int32)
            evt_t  = np.asarray(events_t_list[i], dtype=np.float64).reshape(-1, 1)
            evt_p  = np.asarray(events_p_list[i], dtype=np.int8).reshape(-1, 1)
            evt    = np.concatenate([evt_xy, evt_t, evt_p], axis=1)  # [N,4] -> (x,y,t,p)

            self.hats.process_all(evt)
            out[i] = self.hats.histograms
            self.hats.reset()

        print(f"Computed HATS for sequence {seq_folder}, shape: {out.shape}")

        if self.use_cache:
            if not os.path.exists(cache_dir_path):
                os.makedirs(cache_dir_path, exist_ok=True)
            np.save(cached_hats_path, out)
            print(f"Saved HATS to cache: {cached_hats_path}")

        return out, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)


class HATS_representation:
    def __init__(self, width, height, tau, R, K):
        self.tau = float(tau)
        self.R = int(R)
        self.K = int(K)

        self.cell_width = (width // K)   # num cells horizontally
        self.cell_height = (height // K) # num cells vertically
        self.n_cells = self.cell_width * self.cell_height
        self.n_polarities = 2

        # Row-major cell map consistent with: (y//K)*cell_width + (x//K)
        self.cell_map = get_pixel_cell_partition_matrix(width, height, K)

        self.reset()

    def reset(self):
        self.histogram = np.zeros(
            (self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1),
            dtype=np.float32
        )
        self.event_counter = np.zeros((self.n_cells, self.n_polarities), dtype=np.int64)

    def process_all(self, events: np.ndarray):
        if events.size == 0:
            self.histograms = normalize(self.histogram, self.event_counter)
            return

        xs = events[:, 0].astype(np.int32)
        ys = events[:, 1].astype(np.int32)
        ts = events[:, 2].astype(np.float64)
        ps = events[:, 3].astype(np.int8)

        # 1) count and prefix-sum per (cell, polarity) bucket
        counts = _count_per_bucket(xs, ys, ps, self.cell_map, self.n_cells)
        offs = _exclusive_prefix_sum(counts)

        # 2) build contiguous per-bucket memory in time order (encounter order)
        mem_x, mem_y, mem_t = _build_bucket_memory(xs, ys, ts, ps, self.cell_map, self.n_cells, counts, offs)

        # 3) process buckets in parallel (each bucket sequential internally)
        _process_buckets_parallel(mem_x, mem_y, mem_t, counts, offs,
                                self.R, self.tau,
                                self.histogram, self.event_counter)

        self.histograms = normalize(self.histogram, self.event_counter)


def normalize(histograms: np.ndarray, event_counter: np.ndarray) -> np.ndarray:
    # identical logic, vectorized
    counts = np.clip(event_counter, 1, None)[:, :, np.newaxis, np.newaxis].astype(np.float32)
    return histograms / counts


# -------------------- Numba kernels (new) --------------------

@njit(cache=True)
def _build_bucket_memory(xs, ys, ts, ps, cell_map, n_cells, counts, offs):
    # Place all events into a single contiguous memory pool, bucketed by (cell,polarity)
    n_buckets = n_cells * 2
    total = offs[n_buckets]
    mem_x = np.empty(total, dtype=np.int32)
    mem_y = np.empty(total, dtype=np.int32)
    mem_t = np.empty(total, dtype=np.float64)

    # cursor tracks the next write position per bucket
    cursor = offs.copy()
    for i in range(xs.shape[0]):
        c = cell_map[ys[i], xs[i]]
        b = c * 2 + int(ps[i])
        pos = cursor[b]
        mem_x[pos] = xs[i]
        mem_y[pos] = ys[i]
        mem_t[pos] = ts[i]
        cursor[b] += 1

    return mem_x, mem_y, mem_t

@njit(parallel=True, cache=True)
def _process_buckets_parallel(mem_x, mem_y, mem_t, counts, offs,
                              R, tau,
                              histogram, event_counter):
    """
    Parallel over buckets; within each bucket, preserve event order (time)
    and accumulate surfaces from all prior events in that bucket.
    """
    n_buckets = counts.shape[0]
    size = 2 * R + 1
    inv_tau = 1.0 / tau

    for b in prange(n_buckets):
        L = counts[b]
        if L == 0:
            continue

        start = offs[b]
        c = b // 2               # cell id
        p = b - c * 2            # polarity 0/1

        # local histogram for this (cell, polarity)
        hloc = np.zeros((size, size), dtype=np.float32)

        # process events in time order; for each event i, use prior events [0..i-1]
        for i in range(L):
            xi = mem_x[start + i]
            yi = mem_y[start + i]
            ti = mem_t[start + i]

            x0 = xi - R
            y0 = yi - R

            # Accumulate contributions from all prior events in this bucket
            for j in range(i+1):
                mx = mem_x[start + j]
                my = mem_y[start + j]

                dx = mx - x0
                dy = my - y0
                if 0 <= dx < size and 0 <= dy < size:
                    dt = (ti - mem_t[start + j]) * 1.0e-6  # µs → s
                    w = np.exp(-dt * inv_tau)
                    hloc[dy, dx] += np.float32(w)

        # Write back (no overlap across threads)
        histogram[c, p, :, :] += hloc
        event_counter[c, p] += L


@njit(cache=True)
def _count_per_bucket(xs, ys, ps, cell_map, n_cells):
    # buckets are (cell * 2 + polarity)
    counts = np.zeros(n_cells * 2, dtype=np.int64)
    for i in range(xs.shape[0]):
        c = cell_map[ys[i], xs[i]]
        b = c * 2 + int(ps[i])
        counts[b] += 1
    return counts

@njit(cache=True)
def _exclusive_prefix_sum(counts):
    n = counts.shape[0]
    offs = np.empty(n + 1, dtype=np.int64)
    s = 0
    for i in range(n):
        offs[i] = s
        s += counts[i]
    offs[n] = s
    return offs

# -------------------- Utilities --------------------

@njit(cache=True)
def compute_local_memory_time_surface(event_i: np.ndarray,
                                      filtered_memory: np.ndarray,
                                      R: int,
                                      tau: float) -> np.ndarray:
    """
    Original per-event function kept for compatibility.
    Not used in the fast path.
    """
    x_i = int(event_i[0])
    y_i = int(event_i[1])
    t_i = event_i[2]

    xs = filtered_memory[:, 0].astype(np.int32)
    ys = filtered_memory[:, 1].astype(np.int32)
    ts = filtered_memory[:, 2]

    x0 = x_i - R; y0 = y_i - R
    size = 2*R + 1

    delta_t = (t_i - ts) / 1.0e6  # -> seconds
    values = np.exp(-delta_t / tau).astype(np.float32)

    # As in your original: relies on R >= K-1 so indices are in-range
    idx = (ys - y0) * size + (xs - x0)
    out_flat = np.bincount(idx, weights=values, minlength=size*size)
    return out_flat.reshape(size, size).astype(np.float32)


def get_pixel_cell_partition_matrix(width: int, height: int, K: int, device=None) -> np.ndarray:
    """
    Row-major (y,x) → cell id:
    cell = (y // K) * (width // K) + (x // K)
    """
    cells_x = width // K
    x = np.arange(width, dtype=np.int32)
    y = np.arange(height, dtype=np.int32)
    col = x // K
    row = y // K
    matrix = row[:, None] * cells_x + col[None, :]
    return matrix.astype(np.int32)


def main():
    import time
    dataset_dir = '/fs/nexus-scratch/tuxunlu/git/SportsActionUnderstanding/procs'
    dataset = HatsFast(dataset_dir,
                   width=720, height=720,
                   tau=0.5,
                   R=80,
                   K=80,
                   cache_root="./cache/",
                   use_cache=True,
                   purpose="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dataloader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
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
