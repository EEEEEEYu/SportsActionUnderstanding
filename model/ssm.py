import torch
import torch.nn as nn
from .ssm_utils.s5.s5_model import S5Block


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv(x)
        return self.relu(out + identity)



class ConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, common_blocks=2, downsample_blocks=4):
        super().__init__()
        layers = []
        current_channels = in_channels

        for _ in range(downsample_blocks):
            layers.append(
                ConvResidualBlock(current_channels, current_channels * 2, stride=2)
            )
            current_channels *= 2

        for _ in range(common_blocks):
            layers.append(
                ConvResidualBlock(current_channels, hidden_channels, stride=1)
            )
            current_channels = hidden_channels

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):  # (B*L, C, H, W)
        return self.encoder(x)  # (B*L, D, H', W')



class Ssm(nn.Module):
    def __init__(self, num_classes, in_channels, ssm_dim, state_dim, common_blocks=2, downsample_blocks=4):
        super().__init__()

        self.encoder = ConvEncoder(in_channels, hidden_channels=ssm_dim, common_blocks=common_blocks, downsample_blocks=downsample_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global spatial pooling

        self.bridge = nn.Sequential(
            nn.LayerNorm(ssm_dim),
            nn.Linear(ssm_dim, ssm_dim)
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
            nn.Linear(ssm_dim // 2, num_classes)
        )

        self.state = None

    def reset_state(self, batch_size):
        self.state = self.s5block.s5.initial_state(batch_size)

    def forward(self, x):
        # x: (B, L, C, H, W)
        B, L, C, H, W = x.shape
        x = x.view(B * L, C, H, W)                  # (B*L, C, H, W)
        x = self.encoder(x)                         # (B*L, ssm_dim, H', W')
        x = self.pool(x).squeeze(-1).squeeze(-1)    # (B*L, ssm_dim)
        x = self.bridge(x)                          # (B*L, ssm_dim)
        x = x.view(B, L, -1)                        # (B, L, ssm_dim)

        if self.state is None:
            self.reset_state(B)

        out, self.state = self.s5block(x, self.state)   # (B, L, ssm_dim)
        out = self.classification_head(out)             # (B, L, num_classes)

        return out
    
"""
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

class VectorSsm(nn.Module):
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
    def reset_state(self):
        self.state = None

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
"""
    

"""class Ssm(nn.Module):
    def __init__(
            self,
            num_classes, 
            ssm_dim, 
            state_dim,
            input_dim,
            in_channels=3, 
            max_events=50000, 
            downsample_rate=0.1,
            downsample_factor=4,
            dropout_ratio=0.5,
            ssm_type='vector',
        ):

        super().__init__()

        if ssm_type == 'vector':
            self.net = VectorSsm(
                num_classes=num_classes,
                input_dim=input_dim,
                ssm_dim=ssm_dim,
                state_dim=state_dim,
                max_events=max_events,
                downsample_rate=downsample_rate,
                dropout_ratio=dropout_ratio
            )
        elif ssm_type == 'conv':
            self.net = ConvSsm(
                num_classes=num_classes,
                in_channels=in_channels,
                ssm_dim=ssm_dim,
                state_dim=state_dim,
                downsample_factor=downsample_factor
            )
        else:
            raise ValueError(f"Invalid ssm type: {ssm_type}")
        
    def forward(self, x):
        return self.net(x)
    
    def reset_state(self):
        self.net.reset_state()"""

def main():
    B, L, C, H, W = 2, 5, 10, 360, 360
    num_classes = 10
    ssm_dim = 128
    state_dim = 64

    model = Ssm(num_classes, in_channels=C, ssm_dim=ssm_dim, state_dim=state_dim, common_blocks=2, downsample_blocks=4)
    model = model.cuda()

    dummy_input = torch.randn(B, L, C, H, W).cuda()
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")  # Expect (B, L, num_classes)

    """B, L, N, H = 2, 5, 5000, 256
    num_classes = 10
    input_dim = H
    ssm_dim = 128
    state_dim = 64

    model = VectorSsm(num_classes, input_dim, ssm_dim, state_dim)

    dummy_input = torch.randn(B, L, N, H)  # (B, L, N, H)
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")  # Expect (B, L, num_classes)"""


if __name__ == "__main__":
    main()
