import torch
import torch.nn as nn

class VecLstm(nn.Module):
    def __init__(self, num_vectors, vector_dim, hidden_dim, num_classes,
                 num_layers=1, bidirectional=False, dropout=0.25, reduction="linear"):
        super().__init__()
        self.num_vectors = num_vectors
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Reduce H → 1
        if reduction == "linear":
            self.reducer = nn.Sequential( # (H → 1)
                nn.Linear(vector_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif reduction == "mean":
            self.reducer = None                       # use mean pooling
        else:
            raise ValueError("reduction must be 'linear' or 'mean'")
        
        # LSTM input dim = N (number of vectors after reduction)
        self.lstm = nn.LSTM(
            input_size=num_vectors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)
    
    def forward(self, x):
        """
        x: (B, T, N, H)
        return: (B, T, num_classes)  # per-frame classification
        """
        B, T, N, H = x.shape
        
        # Step 1. Reduce H → 1
        if self.reducer is not None:  # linear projection
            x = self.reducer(x)       # (B, T, N, 1)
            x = x.squeeze(-1)         # (B, T, N)
        else:  # mean pooling
            x = x.mean(dim=-1)        # (B, T, N)
        
        # Step 2. LSTM
        out, _ = self.lstm(x)  # out: (B, T, hidden_dim * num_directions)
        
        # Step 3. Classifier applied at every timestep
        logits = self.fc(out)  # (B, T, num_classes)
        
        return logits

def main():
    B, T, N, H = 4, 10, 5000, 128   # smaller N for testing
    num_classes = 5

    x = torch.randn(B, T, N, H)

    model = VecLstm(
        num_vectors=N,
        vector_dim=H,
        hidden_dim=128,
        num_classes=num_classes,
        bidirectional=True,
        reduction="linear"   # or "mean"
    )

    logits = model(x)
    print(logits.shape)  # (B, T, num_classes) → (4, 10, 5)

if __name__ == '__main__':
    main()
