"""Three-channel float32 operations with the original reduction order.

Image color arithmetic has a fixed three-element last axis. Explicit channel
operations avoid allocating and reducing large HxWx3 intermediates. No fused
arithmetic or reduced precision is used; the initial +0 matches sum's identity.
"""
import numpy as np


def color_dot(delta, vector):
    total = delta[:, :, 0]*vector[0]
    np.add(np.float32(0), total, out=total)
    for channel in (1, 2):
        np.add(total, delta[:, :, channel]*vector[channel], out=total)
    return total


def color_residual_norm(delta, alpha, vector):
    value = delta[:, :, 0]-alpha*vector[0]
    total = value*value
    np.add(np.float32(0), total, out=total)
    for channel in (1, 2):
        value = delta[:, :, channel]-alpha*vector[channel]
        np.multiply(value, value, out=value)
        np.add(total, value, out=total)
    np.sqrt(total, out=total)
    return total
