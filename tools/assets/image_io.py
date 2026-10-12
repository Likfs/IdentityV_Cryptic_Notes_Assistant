"""Unicode-safe image I/O shared by asset preview tools."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    image = _decode_image(Path(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    if not imwrite_unicode(path, image):
        raise ValueError(f"无法编码图片：{path}")


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    path = Path(path)
    if not path.exists():
        return None
    return _decode_image(path, flags)


def _decode_image(path, flags):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.png', image)
    if ok:
        path.write_bytes(encoded.tobytes())
    return ok
