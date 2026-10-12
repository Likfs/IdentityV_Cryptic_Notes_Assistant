"""Composite an original-image overlay without changing transparent pixels."""
import cv2
import numpy as np


def composite_route_layer(screenshot_bgr: np.ndarray, layer_bgra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把已生成的 BGRA 路线层合成到截图，供正式投影预览使用。"""
    if screenshot_bgr.shape[:2] != layer_bgra.shape[:2] or layer_bgra.shape[2] != 4:
        raise ValueError("截图和路线层尺寸不一致")
    warped_alpha_u8 = layer_bgra[:, :, 3]
    # Alpha-zero pixels have exactly the original color. Avoid full-screen
    # float intermediates when the route occupies only part of the viewport.
    bounds = cv2.boundingRect(warped_alpha_u8)
    x, y, width, height = bounds
    output = screenshot_bgr.copy()
    if width == 0 or height == 0:
        return output, warped_alpha_u8
    region = np.s_[y:y+height, x:x+width]
    warped_color = layer_bgra[region][:, :, :3]
    warped_alpha = warped_alpha_u8[region].astype(np.float32) / 255.0
    warped_premultiplied = warped_color.astype(np.float32) / 255.0 * warped_alpha[:, :, None]
    base = screenshot_bgr[region].astype(np.float32) / 255.0
    composite = base * (1.0 - warped_alpha[:, :, None]) + warped_premultiplied
    output[region] = np.clip(composite * 255.0, 0, 255).astype(np.uint8)
    return output, warped_alpha_u8
