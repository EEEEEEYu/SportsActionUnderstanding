import torch.utils.data as data
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from data.utils.preprocessor import Preprocessor
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

CLS_NAME_TO_INT = {f"{i:06d}": i for i in range(16)}


def normalize_array(arr: np.ndarray, normalize: str):
    """
    Normalize a numpy array in-place.
    Args:
        arr: np.ndarray
        normalize: 'standardization' | 'normalization' | 'None'
    Returns:
        np.ndarray
    """
    arr = arr.astype(np.float32)
    eps = 1e-8
    if normalize == 'standardization':
        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean) / (std + eps)
    elif normalize == 'normalization':
        min_val = arr.min()
        max_val = arr.max()
        arr = (arr - min_val) / (max_val - min_val + eps)
    elif normalize == 'None':
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    return arr


def _last_event_time_per_pixel(target_H: int,
                               target_W: int,
                               x_ds: np.ndarray,
                               y_ds: np.ndarray,
                               t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized computation of last event *time* and *index* per pixel.

    Returns:
        last_time_map_flat: (target_H*target_W,) float64 of last timestamps; pixels with no events set to a very negative value.
        last_idx_map_flat:  (target_H*target_W,) int64 of last event indices in the original arrays; -1 for no events.
    """
    total = target_H * target_W
    if t.size == 0:
        # no events in chunk
        last_time_map_flat = np.full(total, -1e20, dtype=np.float64)
        last_idx_map_flat = np.full(total, -1, dtype=np.int64)
        return last_time_map_flat, last_idx_map_flat

    # Flattened pixel indices in downsampled grid
    pix_idx = (y_ds.astype(np.int64) * target_W + x_ds.astype(np.int64))

    # Stable lexsort by (pix_idx, time)
    order = np.lexsort((t, pix_idx))
    pix_sorted = pix_idx[order]
    t_sorted = t[order]

    # Find ends of each group (last occurrence is the last event at that pixel)
    if pix_sorted.size == 0:
        last_time_map_flat = np.full(total, -1e20, dtype=np.float64)
        last_idx_map_flat = np.full(total, -1, dtype=np.int64)
        return last_time_map_flat, last_idx_map_flat

    boundaries = np.flatnonzero(np.diff(pix_sorted))
    group_end_pos = np.concatenate([
        boundaries, np.array([pix_sorted.size - 1], dtype=np.int64)
    ])

    last_pix = pix_sorted[group_end_pos]
    last_time = t_sorted[group_end_pos]
    last_orig_idx = order[group_end_pos]

    # Fill maps
    last_time_map_flat = np.full(total, -1e20, dtype=np.float64)
    last_idx_map_flat = np.full(total, -1, dtype=np.int64)
    last_time_map_flat[last_pix] = last_time
    last_idx_map_flat[last_pix] = last_orig_idx

    return last_time_map_flat, last_idx_map_flat


def create_time_surface(source_H: int,
                        source_W: int,
                        target_H: int,
                        target_W: int,
                        events: tuple[np.ndarray, np.ndarray],
                        tau: float,
                        t_query: float | None = None,
                        output_index: bool = False,
                        normalize: str = 'None'):
    """
    Compute a Time Surface (TS) for a chunk of events at query time t_query.

    Time Surface (per polarity or all events):
        TS(x, y; t_query) = exp(- (t_query - T_last(x, y)) / tau)
    where T_last(x, y) is the timestamp of the most recent event at pixel (x, y)
    before (or at) t_query. Pixels with no prior events have value 0.

    Args:
        source_H, source_W: original sensor resolution.
        target_H, target_W: desired output resolution (<= source size); coordinates are downsampled by integer factors.
        events: (t, xy) where t is (N,), xy is (N, 2) with columns (x, y). Assumes ascending t.
        tau: decay constant in the same time units as 't'.
        t_query: time at which the surface is evaluated. If None, uses last timestamp in t.
        output_index: if True, also returns the last event index per pixel (int) with -1 for no event.
        normalize: optional normalization on the final map ('None' recommended since TS is already in [0,1]).

    Returns:
        ts: np.ndarray of shape (target_H, target_W), dtype float32, in [0, 1].
        last_idx_map (optional): np.ndarray of shape (target_H, target_W) of int64 indices; only if output_index=True.
    """
    # Unpack
    t = events[0]
    xy = events[1]

    if t.size == 0:
        ts = np.zeros((target_H, target_W), dtype=np.float32)
        if output_index:
            return ts, -np.ones((target_H, target_W), dtype=np.int64)
        return ts, None

    # Downsample mapping factors
    fx = max(1, source_W // target_W)
    fy = max(1, source_H // target_H)

    x_ds = (xy[:, 0] // fx).astype(np.int64)
    y_ds = (xy[:, 1] // fy).astype(np.int64)

    # Clamp safety (events exactly at boundary)
    x_ds = np.clip(x_ds, 0, target_W - 1)
    y_ds = np.clip(y_ds, 0, target_H - 1)

    if t_query is None:
        t_query = float(t[-1])

    last_time_flat, last_idx_flat = _last_event_time_per_pixel(
        target_H, target_W, x_ds, y_ds, t.astype(np.float64)
    )

    # Exponential decay; pixels with no events have last_time=-1e20 -> exp(-huge) ~ 0
    delta = (t_query - last_time_flat) / 1.0e6 / float(tau)
    # Numerical safety
    ts_flat = np.exp(-delta)

    ts = ts_flat.reshape(target_H, target_W).astype(np.float32)
    ts = normalize_array(ts, normalize)

    if output_index:
        return ts, last_idx_flat.reshape(target_H, target_W)
    return ts, None


class EventTimeSurface(data.Dataset):
    """
    Dataset that converts event sequences into Time Surface tensors.

    Output per sample: (L, C, H, W) where:
        L: number of temporal chunks produced by the Preprocessor
        C: 1 if use_polarity=False, else 2 (pos, neg)
        H, W: target (possibly downsampled) spatial resolution
    """

    def __init__(
        self,
        dataset_dir: str,
        height: int,
        width: int,
        use_polarity: bool,
        use_cache: bool,
        cache_root: str,
        purpose: str,
        downsample_ratio: float,
        accumulation_interval_ms: float,
        tau: float,
        normalize: str = 'None',
    ):
        self.dataset_dir = dataset_dir
        self.preprocessor = Preprocessor(dataset_dir=dataset_dir)
        self.height = height
        self.width = width
        self.use_polarity = use_polarity
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.purpose = purpose
        self.downsample_ratio = downsample_ratio
        self.accumulation_interval_ms = accumulation_interval_ms
        self.tau = tau
        self.normalize = normalize

        # Target size (integer downsample)
        self.target_H = int(self.height // self.downsample_ratio)
        self.target_W = int(self.width // self.downsample_ratio)
        self.target_H = max(1, self.target_H)
        self.target_W = max(1, self.target_W)

    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        valid_len = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, valid_len, torch.stack(labels)

    def __len__(self):
        return len(self.preprocessor)

    def __getitem__(self, idx):
        events_t_list, events_xy_list, events_p_list, class_name, seq_folder = self.preprocessor[idx]
        print("Processing sequence:", seq_folder)

        # Build cache path
        rel_seq_path = os.path.relpath(seq_folder)
        cache_dir_path = os.path.join(self.cache_root, rel_seq_path)
        cache_name = f"timesurface_tau{self.tau}_ds{self.downsample_ratio}_dt{self.accumulation_interval_ms}ms.pt"
        cached_path = os.path.join(cache_dir_path, cache_name)

        if self.use_cache and os.path.exists(cached_path):
            data = torch.load(cached_path)
            return data, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)

        ts_all = []  # list of (C, H, W)

        counter = 0
        for events_t, events_xy, events_p in tqdm(
            zip(events_t_list, events_xy_list, events_p_list),
            total=len(events_t_list),
            desc="TimeSurface",
        ):
            # Evaluate surface at last timestamp in this chunk
            if not self.use_polarity:
                ts, _ = create_time_surface(
                    source_H=self.height,
                    source_W=self.width,
                    target_H=self.target_H,
                    target_W=self.target_W,
                    events=(events_t, events_xy),
                    tau=self.tau,
                    t_query=None,  # last timestamp
                    output_index=False,
                    normalize=self.normalize,
                )
                ts_stack = ts[None, ...]  # (1, H, W)
            else:
                pos_idx = np.where(events_p == 1)[0]
                neg_idx = np.where(events_p == 0)[0]

                ts_pos, _ = create_time_surface(
                    source_H=self.height,
                    source_W=self.width,
                    target_H=self.target_H,
                    target_W=self.target_W,
                    events=(events_t[pos_idx], events_xy[pos_idx]),
                    tau=self.tau,
                    t_query=None,
                    output_index=False,
                    normalize='None',  # normalize after stacking if desired
                )
                ts_neg, _ = create_time_surface(
                    source_H=self.height,
                    source_W=self.width,
                    target_H=self.target_H,
                    target_W=self.target_W,
                    events=(events_t[neg_idx], events_xy[neg_idx]),
                    tau=self.tau,
                    t_query=None,
                    output_index=False,
                    normalize='None',
                )

                ts_pos = normalize_array(ts_pos, self.normalize)

                # # save ts_pos as an image
                # print(ts_pos.shape)
                # print(ts_pos)
                # plt.imshow(ts_pos, cmap='hot')
                # plt.axis("off")
                # plt.savefig(os.path.join("./", f"{class_name}_{counter}_ts_pos.png"), bbox_inches="tight", pad_inches=0)
                # plt.close()
                # counter += 1

                ts_stack = np.stack([ts_pos, ts_neg], axis=0)  # (2, H, W)
                ts_stack = normalize_array(ts_stack, self.normalize)

            ts_all.append(ts_stack)

        ts_all = np.stack(ts_all, axis=0)  # (L, C, H, W)
        ts_all_t = torch.from_numpy(ts_all).float()

        if self.use_cache:
            os.makedirs(cache_dir_path, exist_ok=True)
            torch.save(ts_all_t, cached_path)

        return ts_all_t, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)
    
def main():
    import matplotlib.pyplot as plt
    import time

    dataset_dir = '/fs/nexus-scratch/tuxunlu/git/SportsActionUnderstanding/procs'

    dataset = EventTimeSurface(dataset_dir=dataset_dir,
                                width=720, height=720,
                                tau=0.5,
                                downsample_ratio=1,
                                accumulation_interval_ms=100,
                                use_cache=True,
                                cache_root='./cache/',
                                use_polarity=True,
                                purpose='train',
                                )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_iter = iter(dataloader)

    for i in range(len(dataloader)):
    # for i in range(1):
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
