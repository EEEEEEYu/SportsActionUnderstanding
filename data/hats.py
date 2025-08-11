import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from crop_dataset.crop_dataset import BSLoader

class Hats(Dataset):
    def __init__(self, dataset_dir, cam,
                 temp_window,
                 width, 
                 height,
                 tau,
                 R,
                 K,
                 calib_dir="crop_dataset/calibration_default",
                 split='train'):
        assert cam in ['event', 'rgb'], "Camera type must be either 'event' or 'rgb'."
        assert split in ['train', 'test'], "Split must be either 'train' or 'test'."

        self.dataset_dir = dataset_dir
        self.calib_dir = calib_dir
        self.split = split
        self.cam = cam
        self.temp_window = temp_window
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
            temp_window=self.temp_window,
            width=self.width,
            height=self.height,
            tau=self.tau,
            R=self.R,
            K=self.K
        )

        self.CLS_NAME_TO_INT = {
            'cheek': 0,
            'chin': 1,
            'ear': 2,
            'eyes': 3,
            'forehead': 4,
            'head': 5,
            'mouth': 6,
            'nose': 7
        }

        self.cls_seq_dirs = []
        # iterate over the folders in os.path.join(self.dataset_dir, self.split)
        for _dir in os.listdir(os.path.join(self.dataset_dir, self.split)):
            # iterate over the sequences
            for seq_dir in os.listdir(os.path.join(self.dataset_dir, self.split, _dir)):
                seq_path = os.path.join(self.dataset_dir, self.split, _dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                cls_name = os.path.basename(seq_path).split('_')[0]
                self.cls_seq_dirs.append((cls_name, seq_path))
        
    def __len__(self):
        return len(self.cls_seq_dirs)
    
    def __getitem__(self, idx):
        cls_name, seq_path = self.cls_seq_dirs[idx]
        loader = BSLoader(
            seq_path=seq_path,
            calib_path=self.calib_dir,
            use_metadata=False,
            to_tensor=True,
            crop=False
        )

        if self.cam == 'event':
            hats_list = []
            events_xytp = loader.event_frames # num_frames list of [1, num_events, 4] tensors with (x, y, t, p)

            for frame in events_xytp:
                evt_tensor = frame.squeeze(0).float()
                
                if not evt_tensor.numel():
                    hats_list.append(torch.zeros(self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1))
                    continue
                evt_tensor[:, 2] /= 1e6  # convert to seconds
                
                self.hats.process_all(evt_tensor)
                hats_list.append(self.hats.histograms)
                self.hats.reset()

            hats_list = torch.stack(hats_list, dim=0)  # [num_frames, n_cells, n_polarities, (2R+1), (2R+1)]
            num_frames, n_cells, n_polarities, H, W = hats_list.shape
            hats_list = hats_list.view(num_frames, n_cells * n_polarities, H * W)

            print(f"Computed HATS for sequence {seq_path}, shape: {hats_list.shape}")

            return torch.tensor(self.CLS_NAME_TO_INT[cls_name]).long(), hats_list
        
        elif self.cam == 'rgb':
            return torch.tensor(self.CLS_NAME_TO_INT[cls_name]).long(), loader.flir_frames

class HATS_representation:
    def __init__(self, temp_window, width, height, tau, R, K):

        self.temp_window = temp_window
        self.tau = tau
        self.R = R
        self.K = K

        self.cell_width = (width // K)
        self.cell_height = (height // K)
        self.n_cells = self.cell_width * self.cell_height
        self.n_polarities = 2
        self.index = {-1: 0, 1: 1}

        self.get_cell = get_pixel_cell_partition_matrix(width, height, K)

        self.reset()

    def reset(self):
        self.histogram = torch.zeros(self.n_cells, self.n_polarities, 2*self.R+1, 2*self.R+1, dtype=torch.float32)
        self.event_counter = torch.zeros(self.n_cells, self.n_polarities, dtype=torch.int32)
        self.cell_memory = np.empty([self.n_cells, self.n_polarities], dtype=torch.Tensor)

    def process(self, event):
        cell = self.get_cell[int(event[1]), int(event[0])]
        polarity_index = self.index[int(event[3])]

        if self.cell_memory[cell, polarity_index] is None:
            self.cell_memory[cell, polarity_index] = event.unsqueeze(0)
        else:
            self.cell_memory[cell, polarity_index] = torch.vstack((self.cell_memory[cell, polarity_index], event))

        self.cell_memory[cell, polarity_index] = filter_memory(
            self.cell_memory[cell, polarity_index],
            event[0],
            event[1],
            event[2],
            self.temp_window,
            self.R
        )

        time_surface = compute_local_memory_time_surface(
            event, self.cell_memory[cell, polarity_index], self.R, self.tau
        )

        self.histogram[cell, polarity_index, :, :] += time_surface

        self.event_counter[cell, polarity_index] += 1

    def process_all(self, events):
        events = events
        for event in events:
            self.process(event)

        self.histograms = normalize(self.histogram, self.event_counter)

def filter_memory(memory: torch.Tensor,
                  event_t: float,
                  event_x: int,
                  event_y: int,
                  temp_window: float,
                  R: int) -> torch.Tensor:
    """
    Same signature, but fully vectorized.
    memory: [M,3] = (t_j, x_j, y_j), sorted by time ascending
    Returns the filtered [K,3] tensor.
    """
    # 1) temporal cutoff via searchsorted
    times = memory[:, 0]                         # shape [M]
    cutoff = event_t - temp_window
    idx    = torch.searchsorted(times, cutoff)   # first index with time >= cutoff
    recent = memory[idx:]                        # [M’ ,3]

    # 2) spatial mask
    xs, ys = recent[:,0], recent[:,1]
    mask = (
        (xs >= event_x - R) & (xs <= event_x + R) &
        (ys >= event_y - R) & (ys <= event_y + R)
    )
    return recent[mask]


# def filter_memory(memory, event_x, event_y, event_t, temp_window, R):
#     """
#     Filter events in memory based on a temporal window and a spatial neighbourhood.
#     Find all events between [event_ts - temp_window, event_ts) and inside a (2R+1)x(2R+1) square centered at (event_i.x, event_i.y).
    
#     Args:
#         memory (torch.Tensor): Tensor of shape (N, 3) with events (t, x, y).
#         event_ts (torch.Tensor): Tensor of shape (N,) with event timestamps.
#         temp_window (float): Temporal window in seconds.
#         R (int): Spatial radius for filtering.

#     Returns:
#         torch.Tensor: Filtered memory tensor.
#     """
#     limit_t = event_t - temp_window

#     # Standard binary‐search to find first index with t >= limit_t
#     left, right = 0, len(memory)
#     while left < right:
#         mid = (left + right) // 2
#         if memory[mid][2] < limit_t:
#             left = mid + 1
#         else:
#             right = mid

#     # `left` is now the first index where t >= limit_t
#     recent = memory[left:]

#     # # Apply spatial filtering
#     # filtered = [
#     #     ev for ev in filtered
#     #     if (ev['x'] >= event_x - R and ev['x'] <= event_x + R and
#     #         ev['y'] >= event_y - R and ev['y'] <= event_y + R)
#     # ]

#     # # 1) Temporal cutoff via binary search
#     # limit_t = event_t - temp_window
#     # times   = memory[:, 2]                                                  # sorted timestamps
#     # idx     = torch.searchsorted(times, limit_t, right=False)
#     # recent  = memory[idx:]                                                  # all t >= limit_t

#     # 2) Spatial mask in one Boolean tensor
#     xs, ys = recent[:, 0], recent[:, 1]
#     mask   = (
#         (xs >= event_x - R) & (xs <= event_x + R) &
#         (ys >= event_y - R) & (ys <= event_y + R)
#     )

#     return recent[mask]


def normalize(histograms: torch.Tensor, event_counter: torch.Tensor) -> torch.Tensor:
    """
    Args:
        histograms: Tensor of shape [n_cells, n_polarities, H, W]
        event_counter: Tensor of shape [n_cells, n_polarities]
    Returns:
        result: same shape as `histograms`, each cell/channel divided by its count
    """
    # Unsqueeze count to [n_cells, n_polarities, 1, 1] and clamp to avoid zero‐division
    counts = event_counter.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)
    return histograms / counts


def compute_local_memory_time_surface(event_i: torch.Tensor,
                                      filtered_memory: torch.Tensor,
                                      R: int,
                                      tau: float) -> torch.Tensor:
    """
    Fully batched time‐surface.
    event_i: [4] = (t_i,x_i,y_i,p_i), filtered_memory: [K,3] = (t_j,x_j,y_j)
    Returns: [(2R+1),(2R+1)] surface tensor.
    """
    t_i, x_i, y_i = event_i[2], event_i[0], event_i[1]

    ts = filtered_memory[:, 2]
    xs = filtered_memory[:, 0].long()
    ys = filtered_memory[:, 1].long()

    delta_t = t_i - ts                            # [K]
    values  = torch.exp(-delta_t / tau)           # [K]

    # compute shifts
    shifted_x = (xs - (x_i - R)).clamp(0, 2*R).long()     # [K]
    shifted_y = (ys - (y_i - R)).clamp(0, 2*R).long()     # [K]

    size = 2*R+1
    idx_flat = shifted_y * size + shifted_x       # [K]
    surface_flat = torch.zeros(size*size, dtype=values.dtype, device=values.device)
    surface_flat = surface_flat.index_add(0, idx_flat, values)

    return surface_flat.view(size, size)

# def compute_local_memory_time_surface(event_i, filtered_memory, R, tau):
    # time_surface = torch.zeros(2*R+1, 2*R+1, dtype=torch.float32)

    # t_i = event_i[2]

    # for event_j in filtered_memory:
    #     delta_t = t_i - event_j[2]

    #     event_value = torch.exp(-delta_t / tau)

    #     shifted_x = int(event_j[0] - (event_i[0] - R))
    #     shifted_y = int(event_j[1] - (event_i[1] - R))

    #     time_surface[shifted_y, shifted_x] += event_value

    # return time_surface
    t_i = event_i[0]
    x_i, y_i = event_i[1], event_i[2]
    
    # Assume filtered_memory[:,0]=x_j, [:,1]=y_j, [:,2]=t_j
    xs = filtered_memory[:, 0]
    ys = filtered_memory[:, 1]
    ts = filtered_memory[:, 2]

    # 1) Compute temporal weights for all events at once
    delta_t = t_i - ts                        # shape: (N,)
    values = torch.exp(-delta_t / tau)        # shape: (N,)

    # 2) Compute shifted indices into the (2R+1)x(2R+1) grid
    #    relative to center (x_i, y_i)
    shifted_x = (xs - (x_i - R)).long()       # shape: (N,)
    shifted_y = (ys - (y_i - R)).long()       # shape: (N,)

    # 3) Mask out-of-bounds events
    size = 2 * R + 1
    valid = (
        (shifted_x >= 0) & (shifted_x < size) &
        (shifted_y >= 0) & (shifted_y < size)
    )
    shifted_x = shifted_x[valid]
    shifted_y = shifted_y[valid]
    values    = values[valid]

    # 4) Flatten 2D indices and accumulate via index_add
    idx_flat = shifted_y * size + shifted_x   # shape: (M,)
    surface_flat = torch.zeros(size*size, dtype=values.dtype, device=values.device)
    surface_flat = surface_flat.index_add(0, idx_flat, values)

    # 5) Reshape back to 2D
    return surface_flat.view(size, size)

def get_pixel_cell_partition_matrix(width: int, height: int, K: int, device=None) -> torch.Tensor:
    """
    Returns a (height x width) matrix where each entry is the cell index
    for that pixel, dividing the image into K×K-pixel cells.

    Args:
        width (int):  Image width in pixels (must be divisible by K).
        height (int): Image height in pixels (must be divisible by K).
        K (int):      Cell size (each cell is K×K pixels).
        device:       Optional torch device (e.g. 'cuda' or 'cpu').

    Returns:
        torch.IntTensor of shape (height, width) with values in [0, n_cells).
    """
    # Number of cells horizontally = width // K
    cell_width = width // K

    # 1) Create 1D coordinate arrays
    x = torch.arange(width, device=device, dtype=torch.int32)  # shape: (W,)
    y = torch.arange(height, device=device, dtype=torch.int32) # shape: (H,)

    # 2) Compute cell-row index for each x, and cell-col index for each y
    #    (integer division by K)
    cell_col_idx = x.div(K, rounding_mode='floor')   # shape: (W,)
    cell_row_idx = y.div(K, rounding_mode='floor')   # shape: (H,)

    # 3) Broadcast into 2D grid and compute linear cell index
    #    According to original formula: index = pixel_row * cell_width + pixel_col
    #    where pixel_row = cell_col_idx, pixel_col = cell_row_idx
    matrix = cell_col_idx.unsqueeze(0) * cell_width    \
           + cell_row_idx.unsqueeze(1)                # shape: (H, W)

    return matrix.to(torch.int32)


def main():
    import matplotlib.pyplot as plt
    import time

    dataset_dir = '/fs/nexus-projects/DVS_Actions/SimonSays'

    dataset = Hats(dataset_dir, split='train', cam='event',
                               temp_window=100,
                               width=680, height=720,
                               tau=0.5,
                               R=20,
                               K=40)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    dataloader_iter = iter(dataloader)               # get explicit iterator

    counter = 0
    while counter < 3:
        # time the actual batch fetch
        start_time = time.perf_counter()
        cls_name, frames = next(dataloader_iter)     # this now includes data loading
        proc_time = time.perf_counter() - start_time
        print(f"Class: {cls_name}, Frames: {frames.shape}")
        print(f"Process time: {proc_time:.4f} s")

        counter += 1
    
    # indices = [0]  # Select a few samples to visualize
    # for index in indices:
    #     events, label = dataset[index]  
    #     # `events` is a structured array with fields x, y, p (polarity), t

    #     # Initialize HATS
    #     hats = HATS(
    #         temp_window=100,
    #         width=35, height=35,
    #         tau=0.5,
    #         R=7,
    #         K=7
    #     )

    #     # Prepare events as a list of [t, x, y, p]
    #     evt_list = [
    #         [float(ev['t'])/1e6, int(ev['x']), int(ev['y']), int(ev['p'])]
    #         for ev in events
    #     ]
    #     evt_tensor = torch.tensor(evt_list, dtype=torch.float32)

    #     print(torch.max(evt_tensor, dim=0))
    #     for i in range(10):
    #         print(evt_tensor[i])

    #     # Compute HATS descriptors
    #     hats.process_all(evt_tensor)

    #     fig=plt.figure(figsize=(20,16))
    #     for i in range(5*5):
    #         fig.add_subplot(5, 5, i+1)
    #         plt.imshow(hats.histograms[i, 0], cmap='hot')
    #     plt.savefig(f'digit_{label}_polarity_0.png')

    #     fig=plt.figure(figsize=(20,16))
    #     for i in range(5*5):
    #         fig.add_subplot(5, 5, i+1)
    #         plt.imshow(hats.histograms[i, 1], cmap='hot')
    #     plt.savefig(f'digit_{label}_polarity_1.png')

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)