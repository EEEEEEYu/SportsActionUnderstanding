import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

class FrameWiseViViT(nn.Module):
    def __init__(self, num_classes, pretrained_model_name="google/vivit-base"):
        super().__init__()
        # Load self‑attention backbone (no final pooling)
        config = VivitConfig.from_pretrained(pretrained_model_name)
        self.backbone = VivitModel.from_pretrained(pretrained_model_name, config=config)
        hidden = config.hidden_size
        self.cls_head = nn.Linear(hidden, num_classes)

    def forward(self, video_frames: torch.Tensor):
        # video_frames: (B, L, C, H, W)
        B, L, C, H, W = video_frames.shape
        # treat each frame as "single-frame video" of length=1
        video_frames = video_frames.view(B * L, C, H, W)
        # reshape to (B*L, num_frames=1, channels, H, W)
        inputs = video_frames.unsqueeze(1)
        outputs = self.backbone(pixel_values=inputs)  
        # outputs.last_hidden_state shape: (B*L, num_tokens, hidden_size)
        # assume CLS token at first position
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        logits = self.cls_head(cls_tokens)  # (B*L, num_classes)
        return logits.view(B, L, -1)


def test_framewise_vivit():
    B, L, C, H, W = 2, 5, 3, 224, 224
    num_classes = 10
    model = FrameWiseViViT(num_classes=num_classes)
    model.eval()

    dummy = torch.randn(B, L, C, H, W)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (B, L, num_classes), f"got {out.shape}"
    print("Frame‑wise ViViT output shape:", out.shape)

if __name__ == "__main__":
    test_framewise_vivit()
