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


if __name__ == "__main__":
    main()
