import dataset
import os
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


from crop_dataset.crop_dataset import BSLoader


import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

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
                 cam,
                 pt_dim=3, 
                 enc_dim=256, 
                 radius=1.0,
                 max_events=80000,
                 downsample_rate=0.1,
                 calib_dir="crop_dataset/calibration_default",
                 split='train',
                 mode='fast'
                 ):
        assert cam in ['event', 'rgb'], "Camera type must be either 'event' or 'rgb'."
        assert split in ['train', 'test'], "Split must be either 'train' or 'test'."

        self.dataset_dir = dataset_dir
        self.calib_dir = calib_dir
        self.split = split
        self.cam = cam
        self.max_events = max_events
        self.downsample_rate = downsample_rate
        self.enc_dim = enc_dim

        if mode == 'fast':
            self.method = FastVecKM(pt_dim, enc_dim, radius)
        elif mode == 'exact':
            self.method = ExactVecKM(pt_dim, enc_dim, radius)
        else:
            raise ValueError(f"Unknown fourier feature mode: {mode}")

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

    
    def __len__(self):
        return len(self.cls_seq_dirs)

    def __getitem__(self, idx):
        cls_name, seq_path = self.cls_seq_dirs[idx]
        loader = BSLoader(
            seq_path=seq_path,
            calib_path=self.calib_dir,
            use_metadata=False,
            to_tensor=True,
            crop=False,
            disable_tqdm=True,
        )

        # (B, L, XXXX)
        if self.cam == 'event':
            all_events = loader.event_frames # num_frames list of [1, num_events, 4] tensors with (x, y, t, p)

            # Define base paths
            project_root = "/fs/nexus-projects/DVS_Actions/SimonSays"
            scratch_root = "/fs/nexus-scratch/haowenyu/SimonSays"

            # Compute relative path under 'SimonSays/...'
            rel_seq_path = os.path.relpath(seq_path, project_root)

            # Construct scratch-based cache path
            scratch_seq_path = os.path.join(scratch_root, rel_seq_path)
            cached_fourier_vec_path = os.path.join(scratch_seq_path, f'fourier_vec_{self.max_events}_{self.enc_dim}.pt')

            # Ensure directory exists
            os.makedirs(os.path.dirname(cached_fourier_vec_path), exist_ok=True)

            # If computed, then just load the saved tensor
            if os.path.exists(cached_fourier_vec_path):
                data = torch.load(cached_fourier_vec_path)
                target_len = int(self.downsample_rate * self.max_events)
                downsample_idx = torch.randperm(data.shape[0])[:target_len]
                data = data[downsample_idx]
                print(f'Found cached fourier feature of shape {data.shape}, loading from {cached_fourier_vec_path}')
                return torch.tensor(self.CLS_NAME_TO_INT[cls_name]).long(), data

            all_features = []
            for idx, events_xytp in enumerate(all_events):
                print(f'Processing element {idx} from sequence: {seq_path}')
                if events_xytp.shape[0] == 0:
                    all_features.append(torch.zeros((1, int(self.max_events * self.downsample_rate), self.enc_dim * 2)))
                    continue
                events_xy = events_xytp.squeeze()[:, :2]
                events_t = events_xytp.squeeze()[:, 2]
                # First normalization, correct built-in offset

                # Second normalization, make it numerical stable for fourier feature compute
                events_xy /= 11
                events_t = (events_t - events_t.min()).unsqueeze(-1) / 1e6 * 10

                normalized_txy = torch.cat([events_t, events_xy], dim=1)

                if normalized_txy.shape[0] > self.max_events:
                    r_idx = torch.randperm(normalized_txy.shape[0])[:self.max_events]
                    normalized_txy = normalized_txy[r_idx]

                # (max_events, enc_dim)
                G = self.method.forward(normalized_txy)

                # (max_events, enc_dim * 2)
                G = torch.cat((G.real, G.imag), dim=-1)

                # (max_events * downsample_rate, enc_dim * 2)
                target_len = int(self.downsample_rate * self.max_events)
                downsample_idx = torch.randperm(G.shape[0])[:target_len]
                G = G[downsample_idx]
                
                # Pad if needed
                if G.shape[0] < target_len:
                    pad_len = target_len - G.shape[0]
                    pad_tensor = torch.zeros((pad_len, G.shape[1]), device=G.device, dtype=G.dtype)
                    G = torch.cat([G, pad_tensor], dim=0)

                all_features.append(G.unsqueeze(0))

            # Always override the cached tensor
            data = torch.cat(all_features, dim=0)
            print(f'Saving fourier vector to {cached_fourier_vec_path}')
            torch.save(data, cached_fourier_vec_path)

            return torch.tensor(self.CLS_NAME_TO_INT[cls_name]).long(), data
        elif self.cam == 'rgb':
            return cls_name, loader.flir_frames
