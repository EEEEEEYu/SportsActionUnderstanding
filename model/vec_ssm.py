import torch
import torch.nn as nn
from .ssm_utils.s5.s5_model import S5Block

class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)

class VecSsm(nn.Module):
    def __init__(self, num_classes, input_dim, ssm_dim, state_dim, max_events=50000, downsample_rate=0.1, dropout_ratio=0.5):
        super().__init__()

        self.feature_transform = nn.Sequential(
            ResidualBlock(input_dim),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            ResidualBlock(input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.Linear(input_dim // 2, 1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.bridge = nn.Sequential(
            nn.Linear(int(max_events * downsample_rate), ssm_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LayerNorm(ssm_dim * 2),
            nn.Linear(ssm_dim * 2, ssm_dim)
        )

        self.s5block = S5Block(
            ssm_dim, 
            state_dim, 
            bidir=False, 
            block_count=1,
            ff_dropout=0.35,
            attn_dropout=0.35,
        )

        self.classification_head = nn.Sequential(
            nn.Linear(ssm_dim, ssm_dim // 2),
            nn.ReLU(),
            nn.Linear(ssm_dim // 2, ssm_dim //2),
            nn.ReLU(),
            nn.Linear(ssm_dim // 2, num_classes)
        )

        self.state = None

    # Call this after whole sequence is finished
    def reset_state(self, batch_size):
        self.state = self.s5block.s5.initial_state(batch_size)

    # Expected input shape: B L N H
    # Expected output shape: B C
    def forward(self, x):
        assert len(x.shape) == 4, f"Number of dimensions is not 4! Input shape: {x.shape}"
        B, L, N, H = x.shape

        # Apply feature transform to all frames
        x = x.view(B * L, N, H)
        x = self.feature_transform(x)         # (B*L, N, 1)
        x = x.squeeze(-1)                     # (B*L, N)
        x = self.bridge(x)                    # (B*L, ssm_dim)
        x = x.view(B, L, -1)                  # (B, L, ssm_dim)

        if self.state is None:
            self.state = self.s5block.s5.initial_state(batch_size=B)
            self.state = self.state.to(x.device)

        # Pass entire sequence through S5 State Space Model
        out, self.state = self.s5block(x, self.state)   # (B, L, ssm_dim)

        out = self.classification_head(out)             # (B, L, num_classes)

        return out

def main():
    B, L, N, H = 2, 5, 5000, 256
    num_classes = 10
    input_dim = H
    ssm_dim = 128
    state_dim = 64

    model = VecSsm(num_classes, input_dim, ssm_dim, state_dim)

    dummy_input = torch.randn(B, L, N, H)  # (B, L, N, H)
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")  # Expect (B, L, num_classes)

if __name__ == '__main__':
    main()