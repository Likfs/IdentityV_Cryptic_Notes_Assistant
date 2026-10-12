import pytest


def test_screen_factor_matches_original_convention():
    from src.registration.template_matching.screen import factor_for_size
    assert factor_for_size(2560,1440)==1
    assert factor_for_size(1920,1080)==pytest.approx(4/3)
    assert factor_for_size(3840,2160)==pytest.approx(2/3)
    assert factor_for_size(0,0)==1
