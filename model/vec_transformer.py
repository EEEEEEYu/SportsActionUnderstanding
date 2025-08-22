import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T]


def _make_padding_mask(valid_len, T, device):
    """
    valid_len: LongTensor [B], each value in [1..T]
    returns: Bool mask [B, T], True where we want to MASK (i.e., padding positions)
    """
    B = valid_len.shape[0]
    ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    # mask True for pad (positions >= valid_len)
    return ar >= valid_len.unsqueeze(1)


def _make_causal_mask(T, device):
    """
    returns: Bool mask [T, T], True where attention is disallowed (future positions)
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class VecTransformer(nn.Module):
    """
    Encoder-only Transformer that:
      - takes input x: (B, T, N, H)
      - reduces H -> 1 (shared linear)
      - reduces N -> d_model (shared linear)
      - applies positional encoding
      - runs TransformerEncoder with optional causal & padding masks
      - returns per-frame logits: (B, T, num_classes)
    """
    def __init__(self,
                 num_vectors,        # N
                 vector_dim,         # H
                 num_classes,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 max_len=1000):
        super().__init__()
        self.reducer_H = nn.Sequential(   # (..., H) -> (..., 1)
            nn.Linear(vector_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.relu = nn.ReLU()
        self.reducer_N = nn.Sequential(   # (..., N) -> (..., d_model)
            nn.Linear(num_vectors, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x, valid_len=None, causal=False):
        """
        x: (B, T, N, H)
        valid_len: LongTensor [B], number of valid (non-pad) steps per sequence (optional)
        causal: bool, if True use lower-triangular (no-lookahead) mask
        returns: (B, T, num_classes)
        """
        B, T, N, H = x.shape
        device = x.device

        # Reduce H -> 1, then squeeze: (B, T, N)
        x = self.reducer_H(x).squeeze(-1)
        # Reduce N -> d_model: (B, T, d_model)
        x = self.reducer_N(x)
        # Positional encoding
        x = self.pos_encoder(x)

        # Masks
        attn_mask = _make_causal_mask(T, device) if causal else None                   # [T, T] or None
        key_padding_mask = _make_padding_mask(valid_len, T, device) if valid_len is not None else None  # [B, T] or None

        # Transformer encoder
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)
        logits = self.fc_out(x)  # (B, T, num_classes)
        return logits


# ------------------------- Dummy test -------------------------
def _demo():
    torch.manual_seed(0)
    device = "cpu"
    B, T_max, N, H = 3, 12, 16, 64
    num_classes = 7

    # variable lengths per sample (<= T_max)
    valid_len = torch.tensor([12, 7, 10], dtype=torch.long, device=device)

    # build a batch with padding in time
    x = torch.randn(B, T_max, N, H, device=device)
    # (optional) zero out padding region for clarity (not required; masks handle it)
    for b in range(B):
        if valid_len[b] < T_max:
            x[b, valid_len[b]:].zero_()

    model = VecTransformer(
        num_vectors=N, vector_dim=H, num_classes=num_classes,
        d_model=128, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.1, max_len=1024
    ).to(device)

    # Training-style forward: causal + padding masks
    logits_train = model(x, valid_len=valid_len, causal=True)   # (B, T, C)
    print("Training forward logits:", logits_train.shape)

    # Inference-style forward: full context (no causal), still respect padding
    logits_infer = model(x, valid_len=valid_len, causal=False)  # (B, T, C)
    print("Inference forward logits:", logits_infer.shape)

    # Extract classification for current time (last valid step per sequence)
    idx = (valid_len - 1).clamp(min=0)
    current_logits = logits_infer[torch.arange(B), idx]  # (B, C)
    print("Current logits:", current_logits.shape)

if __name__ == "__main__":
    _demo()
