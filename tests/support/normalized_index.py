"""Small, evidence-bound profile for recognition normalization lifecycle tests."""
import hashlib
import json
import cv2
import numpy as np


def make_profile(root):
    clean = root/'clean';targets = {}
    for floor, minimum in [('1st',3.178), ('2nd',3.174)]:
        name = f'layout_05/layout_05_{floor}.png'
        path = clean/name;path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((221,257,3),100,np.uint8)
        cv2.rectangle(image,(30,35),(210,180),(175,175,175),3)
        cv2.imencode('.png',image)[1].tofile(path)
        targets[name] = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scales=[minimum, round(minimum*.4,6)], scale_modes={'minimum':minimum,'50_percent':round(minimum*.4,6)})
    audit = root/'base.json';audit.write_text('{}')
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    zoom = root/'zoom.json';zoom.write_text(json.dumps(dict(base_derivation_sha256=digest(audit),
        factor_to_minimum=.4,minimum_scales={n:r['scales'][0] for n,r in targets.items()})))
    config = root/'recognition_scales.json'
    config.write_text(json.dumps(dict(schema_version=1,targets=targets,
        derivation=dict(path='base.json',sha256=digest(audit),dependencies=[]),
        zoom_derivation=dict(path='zoom.json',sha256=digest(zoom),dependencies=[]))))
    return clean
