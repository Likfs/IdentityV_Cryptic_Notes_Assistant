"""Crop uniform borders while preserving exclusive slice coordinates."""
import numpy as np

THRESHOLD = 10.5


def scan(image, direction):
    height, width = image.shape[:2]
    if direction == 'left':
        for x in range(width):
            if np.std(image[:, x]) > THRESHOLD:
                return x
    elif direction == 'right':
        for x in range(width - 1, -1, -1):
            if np.std(image[:, x]) > THRESHOLD:
                return x
    elif direction == 'top':
        for y in range(height):
            if np.std(image[y, :]) > THRESHOLD:
                return y
    elif direction == 'bottom':
        for y in range(height - 1, -1, -1):
            if np.std(image[y, :]) > THRESHOLD:
                return y
    return None


def crop(game):
    height, width = game.shape[:2]
    left = scan(game, 'left') or 0
    right = scan(game, 'right') or width - 1
    top = scan(game, 'top') or 0
    bottom = scan(game, 'bottom') or height - 1
    return {'image': game[top:bottom, left:right], 'left': left, 'top': top}


def crop_binary(game):
    """Same exclusive boundaries as crop(), specialized for 0/255 matcher pixels."""
    height, width = game.shape
    columns = np.count_nonzero(game, axis=0) / height
    rows = np.count_nonzero(game, axis=1) / width
    xs = np.flatnonzero(255 * np.sqrt(columns * (1-columns)) > THRESHOLD)
    ys = np.flatnonzero(255 * np.sqrt(rows * (1-rows)) > THRESHOLD)
    left = (int(xs[0]) if xs.size else None) or 0
    right = (int(xs[-1]) if xs.size else None) or width-1
    top = (int(ys[0]) if ys.size else None) or 0
    bottom = (int(ys[-1]) if ys.size else None) or height-1
    return {'image': game[top:bottom, left:right], 'left': left, 'top': top}
