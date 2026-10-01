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
