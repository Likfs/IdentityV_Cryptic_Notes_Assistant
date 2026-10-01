"""Shared offline map-to-route calibration, independent of experiment inputs."""
import cv2
import numpy as np

def affine_to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 3):
        return matrix
    return np.vstack((matrix, np.array([0.0, 0.0, 1.0])))


def estimate_template_to_route(
    template: np.ndarray,
    route: np.ndarray,
    search_crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict]:
    """一次性标定干净底图到路线图。

    先用多尺度模板匹配提供粗位置，再比较相似、仿射和单应候选。
    完整地图范围、尺度、形变条件数和重投影误差共同排除重复走廊的退化解。
    """
    crop_x, crop_y, crop_w, crop_h = search_crop
    search = route[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    clean_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    content_mask = np.where(clean_gray > 48, 255, 0).astype(np.uint8)
    content_mask = cv2.erode(content_mask, np.ones((3, 3), np.uint8))

    def search_scales(scales: np.ndarray) -> tuple[float, float, tuple[int, int]]:
        best = (-1.0, 0.0, (0, 0))
        for scale in scales:
            size = (round(template.shape[1] * float(scale)), round(template.shape[0] * float(scale)))
            if size[0] > search.shape[1] or size[1] > search.shape[0]:
                continue
            scaled = cv2.resize(clean_gray, size, interpolation=cv2.INTER_AREA)
            scaled_mask = cv2.resize(content_mask, size, interpolation=cv2.INTER_NEAREST)
            response = cv2.matchTemplate(
                search_gray,
                scaled,
                cv2.TM_CCORR_NORMED,
                mask=scaled_mask,
            )
            _, score, _, location = cv2.minMaxLoc(response)
            if score > best[0]:
                best = (float(score), float(scale), location)
        if best[0] < 0:
            raise RuntimeError("路线图搜索区域小于缩放后的干净底图。")
        return best

    maximum_scale = min(search.shape[1] / template.shape[1], search.shape[0] / template.shape[0]) * 0.985
    minimum_scale = max(0.16, maximum_scale * 0.42)
    coarse_step = max(0.003, (maximum_scale - minimum_scale) / 64.0)
    coarse_scales = np.arange(minimum_scale, maximum_scale + coarse_step * 0.5, coarse_step, dtype=np.float32)
    coarse = search_scales(coarse_scales)
    fine_step = max(0.0002, coarse_step / 16.0)
    fine = search_scales(
        np.arange(coarse[1] - coarse_step, coarse[1] + coarse_step + fine_step * 0.5, fine_step, dtype=np.float32)
    )
    score, scale, location = fine
    coarse_matrix = np.array(
        [
            [scale, 0.0, location[0] + crop_x],
            [0.0, scale, location[1] + crop_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 在整层粗定位附近做特征精配准。几何门限会排除重复走廊产生的远距离假匹配。
    detector = cv2.SIFT_create(nfeatures=15000, contrastThreshold=0.006, edgeThreshold=16)
    clean_keypoints, clean_descriptors = detector.detectAndCompute(clean_gray, content_mask)
    search_mask = np.where(search_gray > 45, 255, 0).astype(np.uint8)
    route_keypoints, route_descriptors = detector.detectAndCompute(search_gray, search_mask)
    refinement = {"attempted": False, "accepted": False}
    if clean_descriptors is not None and route_descriptors is not None:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(clean_descriptors, route_descriptors, k=2)
        # 路线颜色和房间文字会让严格 Lowe 比例检验丢失一些正确匹配。这里逐级放宽，
        # 再用受约束的仿射模型和完整地图几何过滤，而不是只让单应矩阵自由拟合。
        match_sets = []
        for ratio in (0.78, 0.84, 0.90, 0.94):
            good = [first for first, second in pairs if first.distance < ratio * second.distance]
            if len(good) >= 8:
                match_sets.append((ratio, good))
        if match_sets:
            base_ratio, base_good = match_sets[0]
            source = np.float32([clean_keypoints[match.queryIdx].pt for match in base_good])
            target = np.float32([route_keypoints[match.trainIdx].pt for match in base_good])
            target[:, 0] += crop_x
            target[:, 1] += crop_y
            predicted = cv2.perspectiveTransform(source.reshape(1, -1, 2), coarse_matrix).reshape(-1, 2)
            gated = np.linalg.norm(predicted - target, axis=1) <= 75.0
            refinement = {
                "attempted": True,
                "ratio_matches": len(base_good),
                "geometrically_gated_matches": int(gated.sum()),
                "accepted": False,
                "candidates": [],
            }

            candidates = []

            def consider_candidate(
                label: str,
                candidate_source: np.ndarray,
                candidate_target: np.ndarray,
                model: str,
            ) -> None:
                if len(candidate_source) < 8:
                    return
                if model == "similarity":
                    affine, candidate_mask = cv2.estimateAffinePartial2D(
                        candidate_source,
                        candidate_target,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0,
                        maxIters=30000,
                        confidence=0.999,
                        refineIters=25,
                    )
                    homography = affine_to_homogeneous(affine) if affine is not None else None
                elif model == "affine":
                    affine, candidate_mask = cv2.estimateAffine2D(
                        candidate_source,
                        candidate_target,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0,
                        maxIters=30000,
                        confidence=0.999,
                        refineIters=25,
                    )
                    homography = affine_to_homogeneous(affine) if affine is not None else None
                else:
                    homography, candidate_mask = cv2.findHomography(
                        candidate_source,
                        candidate_target,
                        cv2.RANSAC,
                        6.0,
                        maxIters=30000,
                        confidence=0.999,
                    )
                if homography is None or candidate_mask is None:
                    return
                inliers = candidate_mask.reshape(-1).astype(bool)
                count = int(inliers.sum())
                transformed = cv2.perspectiveTransform(
                    candidate_source.reshape(1, -1, 2),
                    homography,
                ).reshape(-1, 2)
                errors = np.linalg.norm(transformed - candidate_target, axis=1)
                corners = np.float32(
                    [[[0, 0], [template.shape[1] - 1, 0], [template.shape[1] - 1, template.shape[0] - 1], [0, template.shape[0] - 1]]]
                )
                quadrilateral = cv2.perspectiveTransform(corners, homography)[0]
                center = quadrilateral.mean(axis=0)
                area = abs(float(cv2.contourArea(quadrilateral.astype(np.float32))))
                effective_scale = float(np.sqrt(area / max(1.0, template.shape[0] * template.shape[1])))
                # 拒绝把重复的小房间压成一点的退化解，也拒绝极端剪切。
                local_linear = homography[:2, :2]
                singular_values = np.linalg.svd(local_linear, compute_uv=False)
                condition = float(singular_values[0] / max(singular_values[-1], 1e-9))
                plausible = (
                    count >= 12
                    and float(np.median(errors[inliers])) <= 8.0
                    and crop_x - crop_w * 0.15 <= center[0] <= crop_x + crop_w * 1.15
                    and crop_y - crop_h * 0.20 <= center[1] <= crop_y + crop_h * 1.20
                    and 0.10 <= effective_scale <= maximum_scale * 1.35
                    and condition <= 1.65
                    and cv2.isContourConvex(quadrilateral.astype(np.float32))
                )
                info = {
                    "label": label,
                    "matches": len(candidate_source),
                    "inliers": count,
                    "median_error_px": float(np.median(errors[inliers])) if count else 999.0,
                    "p95_error_px": float(np.percentile(errors[inliers], 95)) if count else 999.0,
                    "effective_scale": effective_scale,
                    "linear_condition": condition,
                    "plausible": bool(plausible),
                }
                refinement["candidates"].append(info)
                if plausible:
                    candidates.append((count, -info["median_error_px"], homography.astype(np.float64), info))

            for ratio, good in match_sets:
                source = np.float32([clean_keypoints[match.queryIdx].pt for match in good])
                target = np.float32([route_keypoints[match.trainIdx].pt for match in good])
                target[:, 0] += crop_x
                target[:, 1] += crop_y
                for model in ("similarity", "affine", "homography"):
                    consider_candidate(f"ratio_{ratio:.2f}_{model}", source, target, model)

                predicted = cv2.perspectiveTransform(source.reshape(1, -1, 2), coarse_matrix).reshape(-1, 2)
                gated = np.linalg.norm(predicted - target, axis=1) <= 90.0
                if int(gated.sum()) >= 12:
                    for model in ("similarity", "affine", "homography"):
                        consider_candidate(
                            f"ratio_{ratio:.2f}_coarse_gated_{model}",
                            source[gated],
                            target[gated],
                            model,
                        )
            if candidates:
                # 内点数优先；同等支持度时选择重投影误差较小的解。
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                _, _, matrix, selected = candidates[0]
                refinement.update(
                    {
                        "accepted": True,
                        "selected_candidate": selected["label"],
                        "inliers": selected["inliers"],
                        "median_inlier_error_px": selected["median_error_px"],
                        "p95_inlier_error_px": selected["p95_error_px"],
                        "homography": matrix.tolist(),
                    }
                )
    if not refinement.get("accepted", False):
        matrix = coarse_matrix
    diagnostics = {
        "method": "masked_multiscale_template_matching",
        "score": score,
        "scale": scale,
        "translation": [int(location[0] + crop_x), int(location[1] + crop_y)],
        "search_crop": list(search_crop),
        "coarse_scale_range": [float(minimum_scale), float(maximum_scale)],
        "feature_refinement": refinement,
    }
    return matrix, diagnostics


