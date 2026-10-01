"""CUDA sparse scoring; final geometric decisions remain on CPU.

No CNN. Reproduce this OpenCV build's continuous three-channel distance
sampling and quantized two-channel semantic sampling separately. Floating
reductions/exp can still differ; verified mode compares the complete CPU run.
"""
from __future__ import annotations
import numpy as np
import torch

from .sparse_pose_refiner import _fields, score_poses as cpu_score_poses


def available():
    return torch.cuda.is_available()


def _template_fields(template):
    value = getattr(template, '_cuda_sparse_fields', None)
    if value is None:
        distances, semantics = _fields(template)
        fields = np.concatenate((distances, semantics), axis=2).transpose(2, 0, 1).copy()
        value = torch.from_numpy(fields).to('cuda')
        template._cuda_sparse_fields = value
    return value


def _evidence(evidence):
    value = getattr(evidence, '_cuda_sparse_evidence', None)
    if value is None:
        ne, ns, nb = len(evidence.edges), len(evidence.semantic), len(evidence.background)
        points = np.concatenate((evidence.edges, evidence.semantic, evidence.background, evidence.background))
        channels = np.concatenate((evidence.directions, evidence.labels + 3,
                                   np.full(nb, 3), np.full(nb, 4)))
        value = (torch.from_numpy(points - evidence.center).to('cuda'),
                 torch.from_numpy(channels.astype(np.int64)).to('cuda'),
                 torch.from_numpy(evidence.edge_weights).to('cuda'), ne, ns, nb)
        evidence._cuda_sparse_evidence = value
    return value


def sample_channels(fields, relative_points, channels, poses):
    """One selected channel per point, correct constant border on each tap."""
    _, height, width = fields.shape
    coords = relative_points[None] * poses[:, None, 2:3]
    coords = coords + poses[:, None, :2]
    fixed = torch.round(coords * 32).to(torch.int64)
    quantized = fixed.to(torch.float32) * (1.0 / 32)
    coords = torch.where((channels < 3)[None, :, None], coords, quantized)
    floor = torch.floor(coords)
    base = floor.to(torch.int64)
    frac = coords - floor
    x, y = base[..., 0], base[..., 1]
    fx, fy = frac[..., 0], frac[..., 1]
    border = torch.where(channels < 3, 40.0, 0.0)
    flat = fields.reshape(-1)
    def tap(dx, dy):
        xx, yy = x + dx, y + dy
        valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        indices = channels[None] * (height * width) + yy.clamp(0, height-1) * width + xx.clamp(0, width-1)
        return torch.where(valid, flat[indices], border)
    a = tap(0, 0) * ((1-fx) * (1-fy))
    b = tap(1, 0) * (fx * (1-fy))
    c = tap(0, 1) * ((1-fx) * fy)
    d = tap(1, 1) * (fx * fy)
    return ((a + b) + c) + d


@torch.no_grad()
def score_poses(template, evidence, poses, *, include_hit=True):
    # Small proposal lists are cheaper on CPU, and retain exact ranking there.
    if include_hit or len(poses) < 54 or len(evidence.edges) < 20 or len(evidence.semantic) < 20:
        return cpu_score_poses(template, evidence, poses, include_hit=include_hit)
    fields = _template_fields(template)
    points, channels, weights, ne, ns, nb = _evidence(evidence)
    poses_cuda = torch.from_numpy(np.ascontiguousarray(poses)).to('cuda')
    values = sample_channels(fields, points, channels, poses_cuda)
    distance = values[:, :ne] / poses_cuda[:, 2:3].clamp_min(.1)
    edge = (torch.exp(-distance / 4.0) * weights).sum(dim=1) / weights.sum()
    semantic = values[:, ne:ne+ns].mean(dim=1)
    conflict = (torch.maximum(values[:, ne+ns:ne+ns+nb], values[:, ne+ns+nb:]).mean(dim=1)
                if nb else torch.zeros_like(semantic))
    score = .50 * semantic + .45 * edge - .05 * conflict
    # One device-to-host transfer and synchronization, not one per metric.
    result = torch.stack((score, semantic, conflict)).cpu().numpy()
    evidence._cuda_score_calls = getattr(evidence, '_cuda_score_calls', 0) + 1
    return result[0], result[1], None, result[2]


def guarded_argmax(template, evidence, trials, scores):
    """Recheck almost tied neighborhoods on CPU; not a proof of global parity."""
    best = scores.argmax(axis=1)
    if scores.shape[1] < 2:
        return best
    ordered = np.partition(scores, -2, axis=1)
    close = np.flatnonzero(ordered[:, -1] - ordered[:, -2] <= 2e-6)
    for row in close:
        # Clipping at the search boundary creates identical poses. Inspect ALL
        # near-best candidates, not just top two: a distinct third pose matters.
        contenders = trials[row][scores[row] >= scores[row].max() - 2e-6]
        if np.all(contenders == contenders[0]):
            evidence._cuda_duplicate_ties = getattr(evidence, '_cuda_duplicate_ties', 0) + 1
            continue
        unique, inverse = np.unique(trials[row], axis=0, return_inverse=True)
        exact = cpu_score_poses(template, evidence, unique, include_hit=False)[0]
        # Expand scores to the original order so equal scores still select the
        # original first candidate, regardless of np.unique's sorted order.
        best[row] = exact[inverse].argmax()
        evidence._cuda_tie_rechecks = getattr(evidence, '_cuda_tie_rechecks', 0) + 1
    return best
