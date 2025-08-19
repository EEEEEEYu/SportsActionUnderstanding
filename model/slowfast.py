import torch
import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast

class SlowFastSequenceModel(nn.Module):
    def __init__(self, num_classes=10, alpha=4, model_depth=50):
        super().__init__()
        self.alpha = alpha

        # Create model with dummy classes
        backbone = create_slowfast(
            input_channels=(3, 3),
            model_depth=model_depth,
            model_num_class=1,   # dummy
            dropout_rate=0.0
        )

        # Remove the entire head (which does pooling + proj)
        # Keep only stem + resnet pathway blocks + fusion blocks
        self.feature_extractor = nn.Sequential(*backbone.blocks[:-1])

        # Find output feature dim from last conv block
        dummy_fast = torch.randn(1, 3, 32, 224, 224)
        dummy_slow = dummy_fast[:, :, ::alpha, :, :]
        with torch.no_grad():
            feats = self.feature_extractor([dummy_slow, dummy_fast])
        c_out = feats.shape[1]

        # New classifier (per time step)
        self.classifier = nn.Linear(c_out, num_classes)

    def forward(self, x):
        # Input: (B, C, T, H, W)
        fast = x
        slow = x[:, :, ::self.alpha, :, :]
        inputs = [slow, fast]

        # Extract features: (B, C_feat, T_out, H_out, W_out)
        feats = self.feature_extractor(inputs)

        # Collapse spatial dims -> (B, C_feat, T_out)
        feats = feats.mean(dim=[3, 4])

        # Rearrange to (B, T_out, C_feat)
        feats = feats.transpose(1, 2)

        # Classify -> (B, T_out, num_classes)
        out = self.classifier(feats)
        return out

def main():
    B, C, T, H, W = 2, 3, 32, 224, 224
    num_classes = 10

    model = SlowFastSequenceModel(num_classes=num_classes, alpha=4)
    x = torch.randn(B, C, T, H, W)
    out = model(x)

    print("Output shape:", out.shape)  # (B, L, num_classes)

if __name__ == "__main__":
    main()