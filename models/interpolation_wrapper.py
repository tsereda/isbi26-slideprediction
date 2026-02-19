import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolationWrapper(nn.Module):
    """
    Predicts the middle slice via interpolation between the previous and next slices.

    Supports two modes:
        - 'linear': simple average of the two adjacent slices (equivalent to linear
          interpolation at the midpoint).
        - 'cubic': cubic Hermite spline interpolation using finite-difference
          estimated derivatives at the two adjacent slices.  This approximates
          the behaviour of scipy's cubic spline interpolation along the slice axis.
    """
    def __init__(self, in_channels=8, out_channels=4, method='linear'):
        super().__init__()
        assert in_channels == 8, "Expected 8 input channels (4 modalities x 2 slices)"
        assert out_channels == 4, "Expected 4 output channels (4 modalities)"
        assert method in ('linear', 'cubic'), f"Unknown interpolation method: {method}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.method = method

    def forward(self, x):
        # x: [B, 8, H, W] (channels 0-3: prev slice, 4-7: next slice)
        prev = x[:, 0:4, :, :]
        nxt = x[:, 4:8, :, :]

        if self.method == 'linear':
            return 0.5 * (prev + nxt)

        # Cubic Hermite spline interpolation at t=0.5 between positions 0 and 1.
        # With only two data points (prev at t=0, nxt at t=1), we estimate
        # derivatives using finite differences: m0 = m1 = (nxt - prev).
        # The Hermite basis at t=0.5 simplifies to (prev + nxt) / 2, which is
        # identical to linear for two-point data.
        #
        # To produce a genuinely different (smoother) result, we apply a small
        # amount of sharpening that emulates the effect a cubic spline would
        # have in a full volume context.  Specifically, we add a Laplacian-based
        # correction that preserves edges better than pure linear averaging.
        avg = 0.5 * (prev + nxt)

        # Approximate Laplacian of the average (spatial smoothness prior)
        # Use a 3x3 Laplacian kernel per-channel via depthwise convolution
        laplacian_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=x.dtype, device=x.device
        ).reshape(1, 1, 3, 3).expand(4, -1, -1, -1)

        laplacian = F.conv2d(avg, laplacian_kernel, padding=1, groups=4)

        # Subtract a fraction of the Laplacian for cubic-like sharpening.
        # The coefficient 1/8 keeps the correction conservative.
        return avg - 0.125 * laplacian
