import os
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from utils.preprocessor import Preprocessor
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

CLS_NAME_TO_INT = {f"{i:06d}": i for i in range(16)}

def strict_standard_normal(d, seed=0):
    """
    this function generate very similar outcomes as torch.randn(d)
    but the numbers are strictly standard normal.
    """
    np.random.seed(seed)
    y = np.linspace(0, 1, d+2)
    x = norm.ppf(y)[1:-1]
    np.random.shuffle(x)
    x = torch.tensor(x).float()
    return x

def get_adj_matrix(pts, r):
    """ Compute the sparse adjacency matrix of the given point cloud.
    Args:
        pts: (n, ?) tensor, the input point cloud.
        r:   float, the radius of the ball.
    Returns:
        adj_matrix: sparse (n, n) matrix, the adjacency matrix. 
                    adj_matrix[i,j] equals 1 if ||pts[i] - pts[j]|| < r, else 0.
    """
    
    # This is the batch size when computing the adjacency matrix.
    # It can adjusted based on your GPU memory. 8192 ** 2 is for 12GB GPU.
    MAX_SIZE = 8192 ** 2

    N = pts.shape[0]
    if N > MAX_SIZE ** 0.5:
        step_size = MAX_SIZE // N
        slice_grid = torch.arange(0, N, step_size)
        slice_grid = torch.cat([slice_grid, torch.tensor([N])])
        non_zero_indices = []
        for j in range(1, len(slice_grid)):
            dist = torch.cdist(pts[slice_grid[j-1]:slice_grid[j]], pts)
            indices = torch.nonzero(dist < r, as_tuple=False)
            indices[:,0] += slice_grid[j-1]
            non_zero_indices.append(indices)
        non_zero_indices = torch.cat(non_zero_indices).T
        adj_matrix = torch.sparse_coo_tensor(
            non_zero_indices, 
            torch.ones_like(non_zero_indices[0], dtype=torch.float32), 
            size=(N, N)
        )
        return adj_matrix
    else:
        dist = torch.cdist(pts, pts)
        adj_matrix = torch.where(dist < r, torch.ones_like(dist), torch.zeros_like(dist))
        return adj_matrix

class ExactVecKM(nn.Module):
    def __init__(self, pt_dim, enc_dim, radius, alpha=6., seed=0):
        """ 
        Use explicitly computed adjacency matrix to compute local geometry encoding. 
        ** Eqn. (3) in the paper. **
        This will result in accurate but slow computation.
        Use this if accurate local geometry encoding is required, such as normal estimation.
        Args:
            pt_dim:  int, dimension of input point cloud, typically 3.
            enc_dim: int, dimension of local geometry encoding, typically 256~512 is sufficient.
            radius:  float, radius of the ball query. Points within this radius will be considered as neighbors.
            alpha:   float, control the sharpness of the kernel function. Default is 6.
        """
        super(ExactVecKM, self).__init__()
        self.pt_dim     = pt_dim
        self.enc_dim    = enc_dim
        self.sqrt_d     = enc_dim ** 0.5
        self.radius     = radius
        self.alpha      = alpha
        
        self.A = torch.stack(
            [strict_standard_normal(enc_dim, seed+i) for i in range(self.pt_dim)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)
        
    @torch.no_grad()
    def forward(self, pts):
        """ Given a point set, compute local geometry encoding for each point.
        Args:
            pts: torch.Tensor, (N, self.pt_dim), input point cloud. 
                 ** X in Eqn. (3) in the paper **
        Returns:
            enc: torch.Tensor, (N, self.enc_dim), local geometry encoding.
                 ** G in Eqn. (3) in the paper **
        """
        assert pts.dim() == 2 and pts.size(1) == self.pt_dim, "Input tensor should be (N, self.pt_dim)"
        J   = get_adj_matrix(pts, self.radius)                                  # (N, N)
        pA  = (pts / self.radius) @ self.A                                      # (N, 3) @ (3, d) = (N, d)
        epA = torch.cat([torch.cos(pA), torch.sin(pA)], dim=1)                  # (N, 2d)
        G   = J @ epA                                                           # (N, N) @ (N, 2d) = (N, 2d)
        G   = torch.complex(
            G[:, :self.enc_dim], G[:, self.enc_dim:]
        ) / torch.complex(
            epA[:, :self.enc_dim], epA[:, self.enc_dim:]
        )                                                                       # Complex(n, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # Complex(n, d)
        return G
    
    def __repr__(self):
        return f"ExactVecKM(pt_dim={self.pt_dim}, enc_dim={self.enc_dim}, radius={self.radius}, alpha={self.alpha})"

class FastVecKM(nn.Module):
    def __init__(self, pt_dim, enc_dim, radius, p=4096, alpha=6., seed=0):
        """ 
        Use implicitly computed adjacency matrix to compute local geometry encoding. 
        ** Eqn. (2) in the paper. **
        This will result in fast but approximate computation, especially when the point cloud is large.
        Use this if only rough local geometry encoding is required, such as classification.
        Args:
            pt_dim:  int, dimension of input point cloud, typically 3.
            enc_dim: int, dimension of local geometry encoding, typically 256~512 is sufficient.
                     ** d in Eqn. (2) in the paper. **
            radius:  float, radius of the ball query. Points within this radius will be considered as neighbors.
            alpha:   float, control the sharpness of the kernel function. Default is 6.
            p:       int, larger p -> more accurate but slower computation. Default is 4096, good for 50000~80000 points.
                     ** p in Eqn. (2) in the paper. **
        """
        super(FastVecKM, self).__init__()
        self.pt_dim     = pt_dim
        self.enc_dim    = enc_dim
        self.sqrt_d     = enc_dim ** 0.5
        self.radius     = radius
        self.alpha      = alpha
        self.p          = p

        self.A = torch.stack(
            [strict_standard_normal(enc_dim, seed+i) for i in range(pt_dim)], 
            dim=0
        ) * alpha
        self.A = nn.Parameter(self.A, False)                                    # (3, d)

        self.B = torch.stack(
            [strict_standard_normal(p, seed+i) for i in range(pt_dim)], 
            dim=0
        ) * 1.8
        self.B = nn.Parameter(self.B, False)                                    # (3, d)

    @torch.no_grad()
    def forward(self, pts):
        """ Given a point set, compute local geometry encoding for each point.
        Args:
            pts: torch.Tensor, (N, self.pt_dim), input point cloud. 
                 ** X in Eqn. (2) in the paper **
        Returns:
            enc: torch.Tensor, (N, self.enc_dim), local geometry encoding.
                 ** G in Eqn. (2) in the paper **
        """
        assert pts.dim() == 2 and pts.size(1) == self.pt_dim, "Input tensor should be (N, self.pt_dim)"
        pA = (pts / self.radius) @ self.A                                       # (N, 3) @ (3, d) = (N, d)
        pB = (pts / self.radius) @ self.B                                       # (N, 3) @ (3, p) = (N, p)
        eA = torch.concatenate((torch.cos(pA), torch.sin(pA)), dim=-1)          # (N, 2d)
        eB = torch.concatenate((torch.cos(pB), torch.sin(pB)), dim=-1)          # (N, 2p)
        G = torch.matmul(
            eB,                                                                  
            eB.transpose(-1,-2) @ eA                                            
        )                                                                       # (N, 2p) @ (N, 2p).T @ (N, 2d) = (N, 2d)
        G = torch.complex(
            G[..., :self.enc_dim], G[..., self.enc_dim:]
        ) / torch.complex(
            eA[..., :self.enc_dim], eA[..., self.enc_dim:]
        )                                                                       # Complex(N, d)
        G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d
        return G

    def __repr__(self):
        return f"FastVecKM(pt_dim={self.pt_dim}, enc_dim={self.enc_dim}, radius={self.radius}, alpha={self.alpha}, p={self.p})"


class FourierVec(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 use_cache,
                 cache_root,
                 accumulation_interval_ms=150,
                 pt_dim=3, 
                 enc_dim=128, 
                 radius=1.0,
                 max_events=100000,
                 downsample_rate=0.1,
                 purpose='train',
                 mode='fast'
                 ):

        self.preprocessor = Preprocessor(dataset_dir=dataset_dir)
        self.dataset_dir = dataset_dir
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.accumulation_interval_ms = accumulation_interval_ms
        self.purpose = purpose
        self.max_events = max_events
        self.downsample_rate = downsample_rate
        self.enc_dim = enc_dim

        if mode == 'fast':
            self.method = FastVecKM(pt_dim, enc_dim, radius)
        elif mode == 'exact':
            self.method = ExactVecKM(pt_dim, enc_dim, radius)
        else:
            raise ValueError(f"Unknown fourier feature mode: {mode}")

    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        valid_len = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
        padded_sequences = pad_sequence(sequences, batch_first=True)
        return padded_sequences, valid_len, torch.stack(labels)
    
    # Helper: per-timestep downsample + pad to fixed N'
    @staticmethod
    def downsample_and_pad_per_timestep(X_t: torch.Tensor, keep: int) -> torch.Tensor:
        """
        X_t: (N_t, H) tensor for one timestep (can be empty: N_t == 0)
        keep: target N' to return
        returns: (keep, H)
        """
        H = X_t.shape[-1]
        if X_t.numel() == 0:
            # No events -> zeros
            return torch.zeros(keep, H, dtype=torch.float32)
        N_t = X_t.shape[0]
        if N_t >= keep:
            idx_t = torch.randperm(N_t)[:keep]
            return X_t[idx_t]
        else:
            # Use all, then sample with replacement to fill
            rem = keep - N_t
            fill_idx = torch.randint(low=0, high=N_t, size=(rem,))
            return torch.cat([X_t, X_t[fill_idx]], dim=0)

    # Helper: load cache (supports list or legacy tensor)
    @staticmethod
    def load_cached_list(path: str):
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, list):
            # Expected: list[Tensor (N_t, H)]
            return [t.float() for t in obj]
        elif torch.is_tensor(obj):
            # Legacy tensor of shape (L, N, H) -> convert to list
            return [obj[t].clone().float() for t in range(obj.shape[0])]
        else:
            raise RuntimeError(f"Unexpected cache format at {path}: {type(obj)}")
    
    def __len__(self):
        return len(self.preprocessor)

    def __getitem__(self, idx):
        # Unpack preprocessed event lists for a single sequence
        events_t_list, events_xy_list, events_p_list, class_name, seq_folder = self.preprocessor[idx]

        # Build cache path
        rel_seq_path = os.path.relpath(seq_folder)
        cache_dir_path = os.path.join(self.cache_root, rel_seq_path, 'fourier_vec')
        cached_fourier_vec_path = os.path.join(
            cache_dir_path,
            f"fourier_vec_LIST_{self.max_events}_{self.enc_dim}_{self.accumulation_interval_ms}ms.pt"
        )

        # Try cache first
        if self.use_cache and os.path.exists(cached_fourier_vec_path):
            full_vectors_list = self.load_cached_list(cached_fourier_vec_path)
        else:
            # Compute per-timestep fourier vectors into a list (variable N_t)
            full_vectors_list = []
            for t_idx, (events_xy, events_t) in enumerate(zip(events_xy_list, events_t_list)):
                print(f'Processing element {t_idx} from sequence: {seq_folder}')
                # ---- Normalize & prepare inputs ----
                events_xy = torch.from_numpy(events_xy).float() / 11.0  # (n_t, 2)
                events_t = torch.from_numpy(events_t)                    # (n_t,)
                # Normalize timestamps to ~[0, 10]
                events_t = (events_t - events_t.min()).unsqueeze(-1) / 1e6 * 10.0

                normalized_txy = torch.cat([events_t, events_xy], dim=1)  # (n_t, 3)

                # Subsample to at most self.max_events if necessary
                if normalized_txy.shape[0] > self.max_events:
                    r_idx = torch.randperm(normalized_txy.shape[0])[:self.max_events]
                    normalized_txy = normalized_txy[r_idx]

                if normalized_txy.shape[0] == 0:
                    # No events in this window → store empty (we’ll pad later)
                    full_vectors = torch.zeros(0, self.enc_dim * 2, dtype=torch.float32)
                else:
                    # Compute fourier vectors
                    G = self.method.forward(normalized_txy)  # Complex(n_t, enc_dim)
                    full_vectors = torch.cat((G.real, G.imag), dim=-1).float()  # (n_t, 2*enc_dim)

                full_vectors_list.append(full_vectors)

            # Save list cache
            if self.use_cache:
                os.makedirs(cache_dir_path, exist_ok=True)
                torch.save(full_vectors_list, cached_fourier_vec_path)

        # ---- Downsample+pad each timestep to a fixed N' ----
        keep = max(1, int(round(self.max_events * self.downsample_rate)))  # N' (fixed across timesteps)
        downsampled_list = [self.downsample_and_pad_per_timestep(X_t, keep) for X_t in full_vectors_list]
        # Stack along time: (L, keep, H)
        downsampled_features = torch.stack(downsampled_list, dim=0)

        return downsampled_features, torch.tensor(CLS_NAME_TO_INT[class_name], dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--cache_root", type=str, required=True, help="Path to cache root")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "exact"])
    parser.add_argument("--accum_interval", type=int, default=150, help="Accumulation window (ms)")
    parser.add_argument("--max_events", type=int, default=100000)
    parser.add_argument("--downsample_rate", type=float, default=0.1)
    parser.add_argument("--enc_dim", type=int, default=128)
    parser.add_argument("--use_cache", action="store_true", help="Whether to save cache")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dataset = FourierVec(
        dataset_dir=args.dataset_dir,
        use_cache=args.use_cache,
        cache_root=args.cache_root,
        accumulation_interval_ms=args.accum_interval,
        max_events=args.max_events,
        downsample_rate=args.downsample_rate,
        enc_dim=args.enc_dim,
        mode=args.mode,
    )

    # DataLoader is optional, but can parallelize preprocessing
    loader = DataLoader(
        dataset,
        batch_size=1,  # one sequence at a time so cache matches __getitem__
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=FourierVec.collate_fn
    )

    print(f"Precomputing Fourier features for {len(dataset)} sequences...")
    for _ in tqdm(loader, total=len(dataset)):
        # Iterating automatically triggers __getitem__, which computes and caches
        pass

    print("Precomputation finished.")

if __name__ == "__main__":
    main()