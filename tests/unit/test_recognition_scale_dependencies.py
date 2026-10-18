import hashlib
import json

import pytest

from src.structure_layout.scales import read_scale_overrides, check_scale_overrides


@pytest.mark.parametrize('changed', ['source', 'audit', 'missing_derivation', 'outside'])
def test_derived_scale_rejects_changed_or_missing_evidence(tmp_path, changed):
    clean = tmp_path/'clean'
    image = clean/'layout_01/layout_01_1st.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'original input')
    audit = tmp_path/'sources/audit.json'
    audit.parent.mkdir()
    audit.write_text('{}')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    derivation = dict(path='sources/audit.json', sha256=digest(audit), dependencies=[
        dict(path='clean/layout_01/layout_01_1st.png', sha256=digest(image))])
    record = dict(source_sha256=digest(image), scales=[3.18], evidence=dict(
        method='shared_geometry_scale_transfer', derivation_sha256=digest(audit)))
    data = dict(schema_version=1, derivation=derivation, targets={'layout_01/layout_01_1st.png':record})
    config = tmp_path/'recognition_scales.json'
    config.write_text(json.dumps(data))
    assert read_scale_overrides(clean) == data['targets']
    if changed == 'source':
        image.write_bytes(b'changed geometry')
    elif changed == 'audit':
        audit.write_text('{"changed":true}')
    elif changed == 'outside':
        derivation['path'] = '../outside.json'
    else:
        del data['derivation']
    config.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        read_scale_overrides(clean)
    assert check_scale_overrides(clean)['errors']


@pytest.mark.parametrize('changed', ['capture', 'audit', 'scale', 'missing_evidence', 'minimum'])
def test_second_zoom_requires_unchanged_capture_geometry_and_minimum_scale(tmp_path, changed):
    clean=tmp_path/'clean';image=clean/'layout_01/layout_01_1st.png'
    image.parent.mkdir(parents=True);image.write_bytes(b'source')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    base=tmp_path/'base.json';base.write_text('{}')
    shot=tmp_path/'capture.png';shot.write_bytes(b'capture')
    audit=tmp_path/'zoom.json'
    key='layout_01/layout_01_1st.png'
    audit.write_text(json.dumps(dict(base_derivation_sha256=digest(base),factor_to_minimum=.4,
        minimum_scales={key:3.2})))
    data=dict(schema_version=1,derivation=dict(path='base.json',sha256=digest(base),dependencies=[]),
        zoom_derivation=dict(path='zoom.json',sha256=digest(audit),dependencies=[dict(path='capture.png',sha256=digest(shot))]),
        targets={key:dict(source_sha256=digest(image),scales=[3.2,1.28],scale_modes={'minimum':3.2,'50_percent':1.28})})
    path=tmp_path/'recognition_scales.json';path.write_text(json.dumps(data))
    assert read_scale_overrides(clean)[key]['scales']==[3.2,1.28]
    if changed=='capture':shot.write_bytes(b'new capture')
    elif changed=='audit':audit.write_text('{}')
    elif changed=='missing_evidence':del data['zoom_derivation']
    else:
        index=0 if changed=='minimum' else 1
        name='minimum' if changed=='minimum' else '50_percent'
        data['targets'][key]['scales'][index]+=0.1
        data['targets'][key]['scale_modes'][name]+=0.1
    path.write_text(json.dumps(data))
    assert check_scale_overrides(clean)['errors']
