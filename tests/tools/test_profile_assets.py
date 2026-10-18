from tools.assets.imports import prepare_profile_assets as importer


def test_number_audit_preserves_gap_and_rejects_duplicate_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, 'ROOT', tmp_path)
    clean, sources = tmp_path/'clean', tmp_path/'sources'
    sources.mkdir()
    for number in (23, 25):
        layout = f'layout_{number:02d}'
        directory = clean/layout
        directory.mkdir(parents=True)
        for floor in ('2nd', '1st'):
            (directory/f'{layout}_{floor}.png').touch()
    for number in (23, 24, 25):
        (sources/f'{number}路线.jpg').touch()
    rows = importer.audit(clean, sources, ('2nd', '1st'))
    assert [r['number'] for r in rows] == [23, 24, 25]
    assert rows[0]['ready'] and rows[2]['ready']
    assert rows[1]['missing_floors'] == ['2nd', '1st'] and not rows[1]['ready']
    (sources/'25另一份.jpg').touch()
    assert not importer.audit(clean, sources, ('2nd', '1st'))[2]['ready']
