import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim, 
            kernel_size, 
            padding=padding, 
            bias=bias
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)   # [B, in+hidden, H, W]
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

    def init_state(self, batch_size, spatial_size, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c


class ConvLstm(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=64, kernel_size=3, num_layers=1, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # H/2
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, 3, stride=2, padding=1),   # H/4
            nn.ReLU()
        )
        self.lstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        return: [B, T, num_classes]
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Encode all frames first
        feats = []
        for t in range(T):
            f = self.encoder(x[:, t])   # [B, hidden, H/4, W/4]
            feats.append(f)
        feats = torch.stack(feats, dim=1)  # [B, T, hidden, H/4, W/4]

        # Init states for each LSTM layer
        H_, W_ = feats.shape[-2:]
        states = [cell.init_state(B, (H_, W_), device) for cell in self.lstm_layers]

        outputs = []
        for t in range(T):
            x_t = feats[:, t]
            for i, cell in enumerate(self.lstm_layers):
                h, c = cell(x_t, states[i])
                states[i] = (h, c)
                x_t = h   # input for next layer

            # Per-frame classification
            pooled = F.adaptive_avg_pool2d(x_t, (1, 1)).flatten(1)  # [B, hidden]
            logits = self.fc(pooled)  # [B, num_classes]
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, num_classes]

def main():
    B, T, C, H, W = 2, 13, 10, 360, 360
    model = ConvLstm(
        in_channels=C,
        num_classes=10,
        hidden_dim=64,
        kernel_size=3,
        num_layers=2,
        embedding_dim=128
    )
    x = torch.randn(B, T, C, H, W)
    y = model(x)
    print("Output shape:", y.shape)  # [B, T, num_classes]

if __name__ == "__main__":
    main()
