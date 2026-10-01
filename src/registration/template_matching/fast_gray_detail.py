"""Grayscale detail features for image matching."""
import cv2
import numpy as np


def features(bgr):
    color = bgr.astype(np.float32) / 255.0
    gray = color[:, :, 0]*0.114 + color[:, :, 1]*0.587 + color[:, :, 2]*0.299
    highpass = gray - cv2.blur(gray, (7, 7), borderType=cv2.BORDER_REPLICATE)
    energy = cv2.blur(highpass*highpass, (7, 7), borderType=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(np.tanh(highpass/np.sqrt(energy+0.0001)))


class FastGrayDetail:
    def __init__(self, map_bgr, ds, *, map_threshold=3, map_mask_radius=10):
        if not np.isfinite(ds) or ds <= 0:
            raise ValueError('ds must be positive')
        self.ds = float(ds)
        height, width = map_bgr.shape[:2]
        size = (max(1, int(width*self.ds)), max(1, int(height*self.ds)))
        color = cv2.resize(map_bgr, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        _, mask = cv2.threshold(gray, int(map_threshold), 255, cv2.THRESH_BINARY)
        radius = int(map_mask_radius)
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
            mask = cv2.dilate(mask, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if not cv2.countNonZero(mask):
            raise ValueError('map mask is empty')
        detail = features(color)
        detail = np.ascontiguousarray(np.where(mask > 0, detail, 0), dtype=np.float32)
        x, y, width, height = cv2.boundingRect(mask)
        self.origin = (x, y)
        self.feature = np.ascontiguousarray(detail[y:y+height, x:x+width])
        self.mask = np.ascontiguousarray(mask[y:y+height, x:x+width])
        self.nbytes = int(self.feature.nbytes+self.mask.nbytes)

    def match_template(self, template, mask, offset=(0, 0), *,
                       reject_outside=True, outside_allowance=0, trace=None):
        def rejected(reason, **details):
            if trace is not None:
                trace.update(status='rejected', reason=reason, **details)
            return None

        outside_allowance = int(outside_allowance)
        if outside_allowance < 0:
            raise ValueError('outside_allowance must be non-negative')
        if trace is not None:
            trace.update(model_feature_shape=list(self.feature.shape), model_origin=list(self.origin),
                         template_shape=list(template.shape), reject_outside=bool(reject_outside),
                         outside_allowance=outside_allowance)
        if template.shape != mask.shape:
            if trace is not None:
                trace.update(status='exception', reason='shape_mismatch')
            raise ValueError('mask and template must have the same shape')
        height, width = template.shape
        if min(height, width) < 2 or height > self.feature.shape[0] or width > self.feature.shape[1]:
            return rejected('template_size_invalid')
        weights = np.ascontiguousarray(mask > 0, dtype=np.float32)
        count = float(weights.sum())
        if trace is not None:
            trace['game_mask_pixels'] = int(count)
        if count <= 0:
            return rejected('game_mask_empty')
        template = np.ascontiguousarray(np.where(weights > 0, template, 0), dtype=np.float32)
        template_energy = float(np.sum(template.astype(np.float64)**2))
        if trace is not None:
            trace['template_energy'] = template_energy
        if not np.isfinite(template_energy) or template_energy < 1e-9:
            return rejected('template_energy_empty')
        squared = np.square(self.feature)
        window_energy = cv2.matchTemplate(squared, weights, cv2.TM_CCORR)
        del squared
        valid = window_energy > max(1e-8, 1e-10*count)
        if not valid.any():
            return rejected('map_window_energy_empty')
        scores = cv2.matchTemplate(self.feature, template, cv2.TM_CCORR)
        np.divide(scores, np.sqrt(np.maximum(window_energy, 1e-20)*template_energy), out=scores)
        valid &= np.isfinite(scores)
        if reject_outside:
            outside_map = np.ascontiguousarray(self.mask == 0, dtype=np.float32)
            outside = cv2.matchTemplate(outside_map, weights, cv2.TM_CCORR)
            valid &= np.isfinite(outside) & (outside < float(outside_allowance)+0.5)
        if not valid.any():
            details = {}
            if reject_outside:
                details['minimum_outside_pixels'] = float(np.min(outside))
                details['outside_allowance'] = outside_allowance
            return rejected('all_positions_outside_map_mask', **details)
        scores[~valid] = -np.inf
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        map_x = (x+self.origin[0]-offset[0])/self.ds
        map_y = (y+self.origin[1]-offset[1])/self.ds
        location = (int(round(map_x)), int(round(map_y)))
        if trace is not None:
            trace.update(status='matched', reason=None, score=float(score), location=list(location))
            if reject_outside:
                trace['selected_outside_pixels'] = int(round(float(outside[y, x])))
        return float(score), location
