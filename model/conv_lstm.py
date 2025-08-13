import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_channels, kernel_size, norm_type='layer', bias=True):
        super().__init__()
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type

        self.conv = nn.Conv2d(input_dim + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=bias)

        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(hidden_channels)
        elif norm_type == 'layer':
            self.norm = lambda x: nn.LayerNorm(x.shape[1:]).to(x.device)
        else:
            self.norm = None

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        # Apply normalization if configured
        if self.norm:
            h_cur = self.norm(h_cur)

        return h_cur, c_cur

    def init_state(self, batch_size, spatial_size, device):
        H, W = spatial_size
        return (torch.zeros(batch_size, self.hidden_channels, H, W, device=device),
                torch.zeros(batch_size, self.hidden_channels, H, W, device=device))

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class ConvLstm(nn.Module):
    def __init__(
            self, 
            height, 
            width, 
            num_classes, 
            in_channels, 
            embedding_dim, 
            hidden_channels, 
            lstm_kernel=(3, 3), 
            num_lstm_layers=1, 
            norm_type='instance'
        ):
        super().__init__()
        self.height = height
        self.width = width

        self.pool1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.enc1 = EncoderBlock(32, 32)
        self.pool2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.enc2 = EncoderBlock(32, hidden_channels)
        

        # Stacked ConvLSTM at bottleneck
        self.lstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_channels, hidden_channels, kernel_size=lstm_kernel, norm_type=norm_type)
            for _ in range(num_lstm_layers)
        ])

        # Decoder: convs + downsample + linear projection
        self.dec_conv1 = nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_channels + 32, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )  # 1024 * H//32 × W//32
        self.to_embedding = nn.Linear(hidden_channels * 16, embedding_dim)

        self.final_fc = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim //2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim //2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        device = x.device
        skip_feats1 = []
        skip_feats2 = []

        # ======== Encoder (per-frame) ==========
        for t in range(T):
            x_t = x[:, t]  # [B, C, H, W]
            f1 = self.enc1(self.pool1(x_t))     # [B, 32, H//2, W//2]
            f2 = self.enc2(self.pool2(f1))      # [B, hidden_channels, H//4, W//4]
            skip_feats1.append(f1)
            skip_feats2.append(f2)

        feats = torch.stack(skip_feats2, dim=1)  # [B, T, C, H//4, W//4]

        # ======== ConvLSTM Stack ==========
        H_, W_ = feats.shape[-2:]
        states = [layer.init_state(B, (H_, W_), device) for layer in self.lstm_layers]
        output_seq = []

        for t in range(T):
            x_t = feats[:, t]
            for i, layer in enumerate(self.lstm_layers):
                h, c = layer(x_t, states[i])
                states[i] = (h, c)
                x_t = h
            output_seq.append(h.unsqueeze(1))

        h_lstm = torch.cat(output_seq, dim=1)  # [B, T, C, H//4, W//4]

       # ======== Decoder: skip + convs + linear projection ==========
        output = []
        for t in range(T):
            h = h_lstm[:, t]             # [B, C, H//4, W//4]

            # Add feature 2
            skip = skip_feats2[t]        # [B, C, H//4, W//4]
            f = torch.cat([h, skip], dim=1)  # [B, 2C, H//4, W//4]
            f = F.relu(self.dec_conv1(f))
            f = F.relu(self.dec_conv2(f))

            # Add feature 1
            f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)  # [B, C, H//4, W//4]
            skip1 = skip_feats1[t]  # [B, 32, H//4, W//4]
            f = torch.cat([f, skip1], dim=1)  # [B, C + 32, H//4, W//4]

            f = self.downsample(f)       # [B, C, H//4, W//4]
            f = f.flatten(1)
            f = self.to_embedding(f)

            pred_logits = self.final_fc(f)

            output.append(pred_logits)

        return torch.stack(output, dim=1)

def main():
    B, T, C, H, W = 1, 13, 10, 640, 480
    model = Convlstm(
        height=640,
        width=480,
        num_classes=10,
        in_channels=C,
        embedding_dim=128,
        hidden_channels=64,
        lstm_kernel=(3, 3),
        num_lstm_layers=2,
        norm_type='instance'  # or 'layer'
    )
    x = torch.randn(B, T, C, H, W)
    y = model(x)
    print("Output shape:", y.shape)  # [B, T, num_classes]

if __name__ == '__main__':
    main()