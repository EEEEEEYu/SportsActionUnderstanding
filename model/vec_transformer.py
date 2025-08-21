import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class EventTransformerClassifier(nn.Module):
    def __init__(self, 
                 num_vectors,        # N
                 vector_dim,         # H
                 num_classes, 
                 d_model=128, 
                 nhead=8, 
                 num_encoder_layers=4, 
                 dim_feedforward=256, 
                 dropout=0.1,
                 max_len=500):
        super().__init__()

        # Step 1. Reduce last dimension H -> 1 (shared)
        self.reducer_H = nn.Linear(vector_dim, 1)

        # Step 2. Reduce N -> d_model
        self.reducer_N = nn.Linear(num_vectors, d_model)

        # Step 3. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Step 4. Transformer Encoder (encoder-only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )

        # Step 5. Classification head (use mean pooling of sequence)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, T, N, H)
        """
        B, T, N, H = x.shape

        # Reduce H -> 1
        x = self.reducer_H(x)        # (B, T, N, 1)
        x = x.squeeze(-1)            # (B, T, N)

        # Reduce N -> d_model
        x = self.reducer_N(x)        # (B, T, d_model)

        # Positional encoding
        x = self.pos_encoder(x)      # (B, T, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # Sequence pooling (mean across T)
        x = x.mean(dim=1)            # (B, d_model)

        # Classification
        logits = self.fc_out(x)      # (B, num_classes)
        return logits

def main():
    B, T, N, H = 2, 20, 16, 64   # batch=2, seq_len=20, 16 vectors of dim 64
    num_classes = 10

    x = torch.randn(B, T, N, H)

    model = EventTransformerClassifier(
        num_vectors=N,
        vector_dim=H,
        num_classes=num_classes,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_len=100
    )

    logits = model(x)
    print(logits.shape)  # (2, 10)

if __name__ == '__main__':
    main()