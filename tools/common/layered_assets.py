"""Offline layered asset loading and projection for calibration previews."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.paths import PROJECT_ROOT
from src.projection.route_overlay import composite_route_layer


def resolve_calibration_path(value: str | Path) -> Path:
    """Resolve project-relative paths stored in portable calibration files."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class RouteOverlayAsset:
    layout: str
    floor: str
    variant: str
    rgba: np.ndarray
    asset_to_clean: np.ndarray
    calibration_path: Path
    asset_path: Path
    structure_mask: np.ndarray | None = None
    asset_kind: str = "route"


def _read_unchanged(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (2, 3):
        return np.vstack((matrix, np.array([0.0, 0.0, 1.0])))
    raise ValueError(f"矩阵必须为 2x3 或 3x3，实际为 {matrix.shape}")


def load_route_overlay_asset(
    calibration_path: Path,
    floor: str,
) -> RouteOverlayAsset:
    calibration_path = calibration_path.resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if floor not in payload["floors"]:
        raise KeyError(f"标定文件没有楼层 {floor}")
    floor_data = payload["floors"][floor]
    if floor_data["status"] != "PASS" or not floor_data.get("asset"):
        raise RuntimeError(f"{payload['layout']} / {floor} 尚无可用路线资产，状态为 {floor_data['status']}")
    asset_path = resolve_calibration_path(floor_data["asset"])
    rgba = _read_unchanged(asset_path)
    kind = payload.get("asset_kind", "route")
    if kind == "clean" and rgba.ndim == 3 and rgba.shape[2] in (3, 4):
        # Reference assets are reused without duplicating or altering source PNGs.
        visible = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY) > 48
        alpha = np.where(visible, rgba[:, :, 3] if rgba.shape[2] == 4 else 255, 0).astype(np.uint8)
        rgba = np.dstack((rgba[:, :, :3], alpha))
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise RuntimeError(f"路线资产必须是 BGRA 图片：{asset_path}")
    structure = None
    if floor_data.get("structure_mask"):
        structure = _read_unchanged(resolve_calibration_path(floor_data["structure_mask"]))
        if structure.ndim != 2 or structure.shape != rgba.shape[:2]:
            raise ValueError("路线结构校验图必须为与路线资产同尺寸的灰度图")
    return RouteOverlayAsset(
        layout=payload["layout"],
        floor=floor,
        variant=payload["variant_id"],
        rgba=rgba,
        asset_to_clean=homogeneous(np.asarray(floor_data["asset_to_clean_homography"])),
        calibration_path=calibration_path,
        asset_path=asset_path,
        structure_mask=structure,
        asset_kind=kind,
    )


def project_route_overlay(
    screenshot_bgr: np.ndarray,
    asset: RouteOverlayAsset,
    clean_to_screen: np.ndarray,
    opacity: float = 0.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把原路线图资产投影到截图，返回合成图、屏幕 alpha 和资产到屏幕矩阵。"""
    if screenshot_bgr.ndim != 3 or screenshot_bgr.shape[2] != 3:
        raise ValueError("游戏截图必须为 BGR 三通道图片")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity 必须位于 0 到 1")

    layer, asset_to_screen = warp_route_overlay(
        asset,
        clean_to_screen,
        (screenshot_bgr.shape[1], screenshot_bgr.shape[0]),
        opacity,
    )
    composite, warped_alpha_u8 = composite_route_layer(screenshot_bgr, layer)
    return composite, warped_alpha_u8, asset_to_screen


def warp_route_overlay(
    asset: RouteOverlayAsset,
    clean_to_screen: np.ndarray,
    screen_size: tuple[int, int],
    opacity: float = 0.96,
) -> tuple[np.ndarray, np.ndarray]:
    """生成可直接显示的 BGRA 路线层，不包含游戏截图。"""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity 必须位于 0 到 1")
    asset_to_screen = homogeneous(clean_to_screen) @ asset.asset_to_clean
    # 颜色用 Lanczos 保住原路线的细线和文字，alpha 用线性插值。OpenCV 在 CPU
    # 上分别处理三通道 uint8 与单通道 uint8，比四通道浮点透视变换更快。
    warped_color = cv2.warpPerspective(
        asset.rgba[:, :, :3],
        asset_to_screen,
        screen_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha_u8 = cv2.warpPerspective(
        asset.rgba[:, :, 3],
        asset_to_screen,
        screen_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = np.clip(warped_alpha_u8.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    return np.dstack((warped_color, alpha)), asset_to_screen
