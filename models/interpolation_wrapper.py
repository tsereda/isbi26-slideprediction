import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolationWrapper(nn.Module):
    """
    Predicts the middle slice as the average of the previous and next slices (channels 0-3 and 4-7).
    Ignores any neural network; just averages the two input slices.
    """
    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        assert in_channels == 8, "Expected 8 input channels (4 modalities × 2 slices)"
        assert out_channels == 4, "Expected 4 output channels (4 modalities)"
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # x: [B, 8, H, W] (channels 0-3: prev, 4-7: next)
        prev = x[:, 0:4, :, :]
        next = x[:, 4:8, :, :]
        avg = 0.5 * (prev + next)
        return avg
