"""Run the production multiscale_structure workflow with isolated Demo caches and outputs."""
from pathlib import Path
import json

from src.app_catalog import MapCatalog
from src.projection_workflow import process_screenshot, save_workflow_result
from src.projection.route_overlay import homogeneous
from tools.common.paths import TOOL_CACHE_ROOT


class DemoProjectionService:
    def __init__(self, backend="auto", cache_root: Path | None = None):
        if backend not in ("cpu", "auto"):
            raise ValueError("Demo 配准后端必须为 cpu 或 auto")
        self.backend = backend
        self.catalog = MapCatalog()
        self.profile = self.catalog.profile("a_bringer_of_doom", "nightmare")
        self.cache_root = (cache_root or TOOL_CACHE_ROOT / "route_projection_demo") / backend
        self.prepared_cache = {}
        self.asset_cache = {}

    def project(self, screenshot, layout, floor, opacity, output, *, capture_ms=0., capture_roi=None):
        if layout not in self.catalog.layouts(self.profile):
            raise ValueError(f"未知布局：{layout}")
        if floor not in ("b1", "1st", "2nd"):
            raise ValueError(f"未知楼层：{floor}")
        result = process_screenshot(
            screenshot, catalog=self.catalog, profile=self.profile, model_bundle=None,
            floor=floor, layout_override=layout, route_versions={}, opacity=opacity,
            prepared_cache=self.prepared_cache, asset_cache=self.asset_cache,
            registration_backend=self.backend, template_cache_root=self.cache_root,
        )
        matrix = None
        if result.accepted and result.registration and result.registration.clean_to_screen is not None:
            asset = self.asset_cache[(self.profile.key, layout, result.variant_id, floor)]
            matrix = homogeneous(result.registration.clean_to_screen) @ asset.asset_to_clean
        directory = save_workflow_result(result, Path(output))
        info = (result.registration.validation.get("registration_backend", {}) if result.registration else {})
        payload = dict(result.diagnostics(), layout=layout, opacity=opacity,
                       capture_ms=capture_ms, capture_roi=capture_roi,
                       projection_ms=result.timings_ms.get("projection"),
                       backend_info=info, registration_backend=self.backend,
                       diagnostic_directory=str(directory),
                       asset_to_screen_homography=matrix.tolist() if matrix is not None else None)
        (directory / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result, payload
