import torch.utils.data as data
import numpy as np
import torch
import os

from .utils.preprocessor import Preprocessor
from torch.nn.utils.rnn import pad_sequence

CLS_NAME_TO_INT = {f"{i:06d}": i for i in range(16)}

def normalize_array(voxel_grid, normalize):
    voxel_grid = voxel_grid.astype(np.float32)  # prevent integer division
    eps = 1e-8  # small constant to avoid divide-by-zero

    if normalize == 'standardization':
        mean = voxel_grid.mean()
        std = voxel_grid.std()
        voxel_grid = (voxel_grid - mean) / (std + eps)
    elif normalize == 'normalization':
        min_val = voxel_grid.min()
        max_val = voxel_grid.max()
        voxel_grid = (voxel_grid - min_val) / (max_val - min_val + eps)
    elif normalize == 'None':
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")
    
    return voxel_grid

def create_voxel_grid_with_index(source_H, source_W, target_H, target_W, num_bins, events, output_index=False, normalize='standardization'):
    """
    Voxelize a chunk of events (t, x, y, p) into a 3D grid of shape:
      (num_bins, target_H, target_W)
    by counting events per time bin, and also return a corresponding nested list
    that holds the original event indices for each voxel.
    
    This version supports downsampling: if the desired grid size (target_H, target_W)
    is lower than the original sensor resolution, the event coordinates are
    mapped accordingly.
    
    Optimizations:
      - All binning is performed in a vectorized manner.
      - Event grouping by voxel is done via sorting and np.unique,
        avoiding per-event loops over the whole stream.
    
    Parameters:
      target_H       : int, desired output height (can be less than sensor height)
      target_W       : int, desired output width  (can be less than sensor width)
      num_bins: int, number of temporal bins
      events  : tuple with:
                  events[0]: 1D array of timestamps (assumed sorted),
                  events[1]: 2D array of (x, y) coordinates,
                  events[2]: 1D array of polarities (ignored here)
    
    Returns:
      voxel_grid   : np.ndarray of shape (num_bins, target_H, target_W) with event counts.
      voxel_indices: nested list [num_bins][target_H][target_W], where each element is a list
                      of event indices (from the original array) that fell into that voxel.
    """
    # Unpack events
    t = events[0]
    x = events[1][:, 0]
    y = events[1][:, 1]
    # p = events[2]  # Polarity is ignored in voxelization

    # Compute downsampling factors (assumes original dims are divisible by desired dims)
    factor_x = source_W // target_W
    factor_y = source_H // target_H

    # Map event coordinates to downsampled grid (ensure integer type)
    x_ds = (x // factor_x).astype(np.int64)
    y_ds = (y // factor_y).astype(np.int64)

    # Determine time bin for each event.
    t_min = t[0]
    t_max = t[-1]
    if t_min == t_max:
        # If all events have the same timestamp, assign bin 0.
        bin_idx = np.zeros_like(t, dtype=np.int64)
    else:
        time_edges = np.linspace(t_min, t_max, num_bins + 1)
        # np.digitize returns indices in 1...num_bins+1, so subtract 1.
        bin_idx = np.digitize(t, time_edges, right=False) - 1
        # Ensure that events with t == t_max are assigned to the last bin.
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    # Compute overall voxel index for each event:
    # overall_idx = bin_idx * (target_H*target_W) + (y_ds * target_W + x_ds)
    overall_idx = bin_idx * (target_H * target_W) + y_ds * target_W + x_ds

    # Use np.bincount to compute voxel counts over all voxels at once.
    total_voxels = num_bins * target_H * target_W
    counts = np.bincount(overall_idx, minlength=total_voxels)
    voxel_grid = counts.reshape(num_bins, target_H, target_W).astype(np.float32)

    voxel_grid = normalize_array(voxel_grid, normalize)

    if not output_index:
        return voxel_grid, None

    # Group event indices by voxel.
    # First, sort events by overall_idx.
    sorted_order = np.argsort(overall_idx)
    sorted_voxels = overall_idx[sorted_order]
    # Get unique voxel indices, their start positions, and counts.
    unique_voxels, start_indices, group_counts = np.unique(
        sorted_voxels, return_index=True, return_counts=True)

    # Prepare the nested list structure: [num_bins][target_H][target_W]
    voxel_indices = [[[[] for _ in range(target_W)] for _ in range(target_H)] for _ in range(num_bins)]
    
    # For each unique voxel, determine its time bin and spatial location,
    # then assign the list of corresponding event indices.
    for start, cnt in zip(start_indices, group_counts):
        voxel_val = sorted_voxels[start]
        b = voxel_val // (target_H * target_W)
        rem = voxel_val % (target_H * target_W)
        r = rem // target_W
        c = rem % target_W
        # Retrieve original event indices for this voxel.
        voxel_indices[b][r][c] = sorted_order[start:start + cnt].tolist()

    return voxel_grid, voxel_indices


class EventVoxel(data.Dataset):
    def __init__(
            self, 
            dataset_dir: str,
            height: int,
            width: int,
            num_bins: int,
            use_polarity: bool,
            use_cache: bool,
            cache_root: str,
            purpose: str,
            downsample_ratio: float
        ):
        self.dataset_dir = dataset_dir
        self.preprocessor = Preprocessor(dataset_dir=dataset_dir)
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.use_polarity = use_polarity
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.purpose = purpose
        self.downsample_ratio = downsample_ratio

    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        valid_len = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, valid_len, torch.stack(labels)

    def __len__(self):
        return len(self.preprocessor)

    def __getitem__(self, idx):
        events_t_list, events_xy_list, events_p_list, class_name = self.preprocessor[idx]

        if self.use_cache:
            rel_seq_path = os.path.relpath(self.dataset_dir)
            cache_dir_path = os.path.join(self.cache_root, rel_seq_path)
            cached_voxel_path = os.path.join(cache_dir_path, f"voxel_{self.num_bins}_{('up' if self.use_polarity else 'np')}.pt")

            if os.path.exists(cached_voxel_path):
                data = torch.load(cached_voxel_path)
                return data, CLS_NAME_TO_INT[class_name]

        else:
            cached_voxel_path = None

        voxel_grid_all_np = []

        for events_t, events_xy, events_p in zip(events_t_list, events_xy_list, events_p_list):
            if not self.use_polarity:
                voxel_grid_np, _ = create_voxel_grid_with_index(
                    source_H=self.height,
                    source_W=self.width,
                    target_H=self.height,
                    target_W=self.width,
                    num_bins=self.num_bins,
                    events=(events_t, events_xy),
                    output_index=False,
                    normalize='standardization'
                )
            else:
                positive_idx = np.where(events_p == 1)[0]
                negative_idx = np.where(events_p == 0)[0]

                voxel_grid_pos, _ = create_voxel_grid_with_index(
                    source_H=self.height,
                    source_W=self.width,
                    target_H=int(self.height // self.downsample_ratio),
                    target_W=int(self.width // self.downsample_ratio),
                    num_bins=self.num_bins,
                    events=(events_t[positive_idx], events_xy[positive_idx]),
                    output_index=False,
                    normalize='None' # Normalize only after concat
                )

                voxel_grid_neg, _ = create_voxel_grid_with_index(
                    source_H=self.height,
                    source_W=self.width,
                    target_H=int(self.height // self.downsample_ratio),
                    target_W=int(self.width // self.downsample_ratio),
                    num_bins=self.num_bins,
                    events=(events_t[negative_idx], events_xy[negative_idx]),
                    output_index=False,
                    normalize='None' # Normalize only after concat
                )

                voxel_grid_np = np.concatenate([voxel_grid_pos, voxel_grid_neg], axis=0)
                voxel_grid_np = normalize_array(voxel_grid=voxel_grid_np, normalize='standardization')

            voxel_grid_all_np.append(voxel_grid_np)

        voxel_grid_all_np = np.stack(voxel_grid_all_np, axis=0)
        voxel_grid_all_torch = torch.from_numpy(voxel_grid_all_np).float()

        # (L, C, H, W)
        return voxel_grid_all_torch, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)