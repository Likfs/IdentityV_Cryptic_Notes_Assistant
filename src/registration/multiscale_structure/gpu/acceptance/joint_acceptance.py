"""Bounded single-category exception; all other multiscale_structure evidence remains required."""
from src.registration.multiscale_structure.gpu.search_core.bounded_policy import finite

POLICY = 'single_category_geometry_support_v1'


def review_single_category(review, raster):
    checks = review.get('checks', {})
    quality = checks.get('selected_pose_quality', {})
    failed = [name for name, check in checks.items() if not check.get('passed')]
    limits = {}

    def bounded(name, value, threshold, lower=True):
        limits[name] = {'value': value, 'threshold': threshold,
                        'comparison': '>=' if lower else '<=',
                        'passed': bool(finite(value) and (value >= threshold if lower else value <= threshold))}

    semantic = raster.get('semantic_precision')
    bounded('raster_semantic_precision', semantic, .68)
    limits['single_minor_category_failure'] = {'passed': bool(
        failed == ['selected_pose_quality'] and finite(semantic) and semantic < .72),
        'failed_checks': failed, 'normal_semantic_threshold': .72}
    bounded('raster_structure_precision', raster.get('structure_precision'), .95)
    bounded('trusted_semantic_precision', quality.get('semantic'), .95)
    bounded('interior_semantic_precision', quality.get('interior'), .95)
    bounded('wall_support_fraction', quality.get('support'), .95)
    orientations = quality.get('orientations', [])
    bounded('minimum_axis_support', min(orientations) if len(orientations) == 2 else None, 3)
    bounded('wall_median_screen_px', quality.get('median_screen_px'), 1.5, False)
    bounded('visible_p95_screen_px', checks.get('visible_perturbation', {}).get('p95_screen_px'), 4., False)
    full = checks.get('full_viewport_bound', {})
    bounded('viewport_p95_px', full.get('viewport_p95_px'), 12., False)
    bounded('holdout_p95_px', full.get('holdout_p95_px'), 16., False)
    # Require every named gate, including zero unresolved spatial competitors.
    required = ('two_axis_support', 'partial_fit_redundancy', 'visible_perturbation',
                'full_viewport_bound', 'background_and_extent', 'original_hard_gates',
                'all_competitors_resolved')
    limits['other_gates'] = {'passed': all(checks.get(k, {}).get('passed') is True for k in required)}
    return {'policy': POLICY, 'accepted': all(c['passed'] for c in limits.values()),
            'checks': limits, 'reasons': [k for k, c in limits.items() if not c['passed']]}
