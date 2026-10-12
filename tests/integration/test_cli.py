from __future__ import annotations
import pytest
from src.app_catalog import MapCatalog, load_settings


@pytest.mark.parametrize('backend', ['pure_structure', 'icon_structure'])
def test_offline_projection_requires_a_confirmed_layout(monkeypatch, capsys, tmp_path, backend):
    from src import __main__ as main
    catalog = MapCatalog()
    settings = load_settings(catalog, tmp_path / "settings.json")
    settings['layout_recognition_backend'] = backend
    monkeypatch.setattr(main, 'load_settings', lambda _: settings)
    monkeypatch.setattr(main.sys, 'argv', ['src', '--project-once', 'unused.png'])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2 and '--layout' in capsys.readouterr().err
