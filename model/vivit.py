import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

def _tokens_per_frame(cfg: VivitConfig) -> int:
    # number of spatial tokens per frame after tubelet embedding
    ph, pw = cfg.tubelet_size[1], cfg.tubelet_size[2]
    assert cfg.image_size % ph == 0 and cfg.image_size % pw == 0, "image_size must be divisible by tubelet_size (spatial)."
    return (cfg.image_size // ph) * (cfg.image_size // pw)

class Vivit(nn.Module):
    """
    Wraps Hugging Face VivitModel (raw backbone) to output per-frame logits:
      input:  x  -> (B, T, C, H, W)
      output: logits -> (B, T, num_classes)

    Notes:
    - Uses tubelet_size=(1, ph, pw) so temporal stride = 1 (one token group per frame).
    - Pools spatial tokens per frame (mean) to obtain a T-length sequence of features.
    - Variable-length padding handled via `valid_len` (masking logits at padded steps).
    - Causal=True does prefix evaluation (T forward passes) to simulate causal attention,
      since VivitModel forward does not accept an attention mask.  # (see HF docs)
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 224,
        tubelet_size=(1, 18, 18),
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        self.config = VivitConfig(
            image_size=image_size,
            num_frames=8,  # just a hint; ViViT can handle variable T at runtime if divisible by tubelet_t
            tubelet_size=list(tubelet_size),
            num_channels=in_channels,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=mlp_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attn_dropout,
        )
        # Important: temporal tubelet size = 1 to keep one step per frame
        assert self.config.tubelet_size[0] == 1, "Set tubelet_size[0] (temporal) to 1 for true per-frame outputs."

        self.backbone = VivitModel(self.config)  # raw hidden states (no head) :contentReference[oaicite:3]{index=3}
        self.tokens_per_frame = _tokens_per_frame(self.config)
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, num_classes),
        )

    @torch.no_grad()
    def _shape_check(self, x):
        # x: (B, T, C, H, W)
        _, _, C, H, W = x.shape
        if C != self.config.num_channels:
            raise ValueError(f"num_channels={C} mismatch config.num_channels={self.config.num_channels}")
        if not (H == W == self.config.image_size):
            raise ValueError(f"Expected square frames {self.config.image_size}x{self.config.image_size}, got {H}x{W}.")

    def _encode_once(self, x):
        # x: (B, T, C, H, W)
        outputs = self.backbone(pixel_values=x)  # <- no interpolate_pos_encoding
        hs = outputs.last_hidden_state  # (B, 1 + T*P, D)
        hs = hs[:, 1:, :]               # drop CLS
        P = self.tokens_per_frame
        B = hs.size(0)
        if hs.shape[1] % P != 0:
            raise RuntimeError(f"Token count {hs.shape[1]} not divisible by tokens_per_frame={P}. "
                            f"Check tubelet_size & image_size.")
        T_eff = hs.shape[1] // P
        hs = hs.view(B, T_eff, P, -1).mean(dim=2)  # (B, T, D)
        return self.head(hs)                       # (B, T, num_classes)

    def forward(self, x, valid_len: torch.LongTensor | None = None, causal: bool = False):
        """
        x: (B, T, C, H, W)
        valid_len: (B,) number of valid frames
        causal: if True, emulate causal behavior by zeroing future frames per prefix step
        """
        self._shape_check(x)
        B, T, _, _, _ = x.shape
        device = x.device

        if not causal:
            # One pass with full context
            logits = self._encode_once(x)  # (B, T, C)
        else:
            # Prefix evaluation with fixed T: zero future frames for each t
            zeros_like = torch.zeros_like(x)
            outs = []
            for t in range(T):
                x_masked = zeros_like.clone()
                # copy only the observed prefix [0..t]
                x_masked[:, :t + 1].copy_(x[:, :t + 1])
                # IMPORTANT: do NOT use interpolate_pos_encoding here
                logits_full = self._encode_once(x_masked)   # (B, T, C)
                outs.append(logits_full[:, t:t+1])          # take the current step only
            logits = torch.cat(outs, dim=1)                  # (B, T, C)

        # Optional: mask logits at padded timesteps (useful for metrics)
        if valid_len is not None:
            idx = torch.arange(T, device=device).unsqueeze(0)      # (1, T)
            pad_mask = idx >= valid_len.unsqueeze(1)               # (B, T)
            logits = logits.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return logits



# -------------------- quick demo --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, H, W = 2, 8, 3, 360, 360
    num_classes = 5

    x = torch.randn(B, T, C, H, W)
    valid_len = torch.tensor([8, 5], dtype=torch.long)  # second sample padded after step 4

    model = Vivit(
        num_classes=num_classes,
        in_channels=C,
        image_size=H,
        tubelet_size=(1, 18, 18),  # temporal=1 → per-frame outputs
        hidden_size=384,           # lighter than default
        num_layers=4,
        num_heads=8,
        mlp_dim=512,
        dropout=0.0,
        attn_dropout=0.0,
    )

    # full-context (non-causal) pass
    logits_full = model(x, valid_len=valid_len, causal=False)
    print("Full-context logits:", tuple(logits_full.shape))  # (B, T, num_classes)

    # causal (prefix-eval) pass
    logits_causal = model(x, valid_len=valid_len, causal=True)
    print("Causal logits:", tuple(logits_causal.shape))      # (B, T, num_classes)
