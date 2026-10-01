"""Batched float32 color-mixture arithmetic, with screenshot-local ownership."""
import numpy as np
import torch
from .pixel_math import color_dot, color_residual_norm


class ColorMixture:
    def __init__(self, delta):
        self.delta = delta
        # Small images cost less on CPU. No image/evidence is globally cached.
        self.gpu = delta.shape[0]*delta.shape[1] >= 100_000 and torch.cuda.is_available()
        self.tensor = torch.from_numpy(np.ascontiguousarray(delta)).to('cuda') if self.gpu else None

    @torch.no_grad()
    def evaluate(self, vector, *, clip=False):
        denominator = np.sum(vector*vector)
        if not self.gpu:
            alpha = color_dot(self.delta, vector)/denominator
            if clip:
                alpha = np.clip(alpha, 0, 1.1)
            return alpha, color_residual_norm(self.delta, alpha, vector)
        vector_gpu = torch.from_numpy(np.ascontiguousarray(vector)).to('cuda')
        delta = self.tensor
        dot = delta[:, :, 0]*vector_gpu[0] + 0.0
        dot = dot + delta[:, :, 1]*vector_gpu[1]
        dot = dot + delta[:, :, 2]*vector_gpu[2]
        # A host scalar divisor lets PyTorch replace division with reciprocal
        # multiplication (different rounding). A device tensor preserves div.
        divisor = torch.tensor(denominator, dtype=torch.float32, device='cuda')
        alpha = dot/divisor
        if clip:
            alpha = alpha.clamp(0, 1.1)
        residual = delta[:, :, 0]-alpha*vector_gpu[0]
        squared = residual*residual + 0.0
        for channel in (1, 2):
            residual = delta[:, :, channel]-alpha*vector_gpu[channel]
            squared = squared + residual*residual
        packed = torch.stack((alpha, torch.sqrt(squared))).cpu().numpy()
        return packed[0], packed[1]


def pixel_median(values):
    """Exact np.median(axis=0) for float32 RGB values originating from uint8.

    Histogram ranks preserve the even-count midpoint and float32 return type.
    Callers only pass pixels from uint8 images, never mixed/filtered colors.
    """
    if len(values) < 4096:
        return np.median(values, axis=0)
    pixels = values.astype(np.uint8)
    lower, upper = (len(values)-1)//2, len(values)//2
    result = np.empty(3, np.float32)
    for channel in range(3):
        cumulative = np.bincount(pixels[:, channel], minlength=256).cumsum()
        a = np.searchsorted(cumulative, lower, side='right')
        b = np.searchsorted(cumulative, upper, side='right')
        result[channel] = (a+b)*.5
    return result
