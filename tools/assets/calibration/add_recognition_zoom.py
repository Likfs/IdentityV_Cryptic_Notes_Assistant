"""Measure a second fixed recognition zoom from paired captures; projection stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.app_catalog import MapCatalog
from src.maintenance_paths import profile_root
from src.structure_layout.scales import read_scale_overrides
from tools.common.route_calibration import fit_fixed_scale


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def measure(large, minimum):
    sift = cv2.SIFT_create(nfeatures=2500)
    ka, da = sift.detectAndCompute(cv2.cvtColor(large, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(minimum, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        raise ValueError('截图特征不足')
    pairs = [m for m,n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < .7*n.distance]
    if len(pairs) < 12:
        raise ValueError('对应点不足')
    a = np.float32([ka[m.queryIdx].pt for m in pairs])
    b = np.float32([kb[m.trainIdx].pt for m in pairs])
    matrix, mask = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000)
    if matrix is None:
        raise ValueError('无法建立截图对应关系')
    a,b = a[mask.ravel().astype(bool)], b[mask.ravel().astype(bool)]
    rotation = float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
    factor, details = fit_fixed_scale(a,b)
    if (len(a)<12 or min(np.ptp(b,axis=0))<100 or abs(rotation)>.15
            or details['residual_p95_original_pixels']>2 or not .1<factor<.9):
        raise ValueError('截图对应点覆盖、方向或残差不符合要求')
    return dict(factor=factor, rotation=rotation, points_zoomed=a.tolist(),
                points_minimum=b.tolist(), **details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--minimum',type=Path,nargs='+',required=True)
    parser.add_argument('--zoomed',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    profile=MapCatalog().profile(*args.profile.split('/'))
    base=profile_root(profile.clean_root)
    config_path=base/'recognition_scales.json'
    before=sha(config_path)
    records=read_scale_overrides(profile.clean_root, offline=True)
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if not records or config.get('zoom_derivation'):
        raise ValueError('需要已核对的单档最小尺度；已有第二档时请在暂存副本重新标定')
    if any(len(r['scales'])!=1 for r in records.values()):
        raise ValueError('每张底图必须只有一个已核对的最小尺度')
    if min(len(set(args.minimum)),len(set(args.zoomed)))<2:
        raise ValueError('至少需要两张最小和两张 50% 截图')
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    captures=[]
    for kind,paths in [('minimum',args.minimum),('50_percent',args.zoomed)]:
        for i,path in enumerate(paths):
            data=path.read_bytes()
            image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'无法读取截图：{path}')
            captures.append(dict(kind=kind, path=f'sources/recognition_zoom50/{kind}_{i+1:02}.png',
                sha256=hashlib.sha256(data).hexdigest(), original_path=str(path), image=image, data=data))
    links=[]; rejected=[]
    for large in [c for c in captures if c['kind']=='50_percent']:
        for small in [c for c in captures if c['kind']=='minimum']:
            try:
                if large['image'].shape!=small['image'].shape:
                    raise ValueError('截图尺寸不同，须先核对 ROI')
                fit=measure(large['image'],small['image'])
                links.append(dict(zoomed=large['path'],minimum=small['path'],**fit))
            except ValueError as exc:
                rejected.append(dict(zoomed=large['path'],minimum=small['path'],reason=str(exc)))
    if min(len({r[k] for r in links}) for k in ('zoomed','minimum'))<2:
        raise ValueError('至少需要两组独立截图通过几何核对')
    factors=np.array([r['factor'] for r in links])
    factor=float(np.median(factors))
    if np.ptp(factors)/factor>.005:
        raise ValueError('截图间的缩放比例不一致')
    audit=dict(schema_version=1,profile=profile.key,mode='50_percent',
        factor_to_minimum=factor,method='paired_capture_fixed_geometry',
        base_derivation_sha256=config['derivation']['sha256'],
        all_floors_ingame_verified=False,layout_labels_verified=False,
        tool_sha256=sha(Path(__file__)),links=links,rejected_links=rejected,
        captures=[{k:v for k,v in c.items() if k not in ('image','data')} for c in captures],
        minimum_scales={name:r['scales'][0] for name,r in records.items()})
    audit_rel=Path('sources/recognition_zoom50/calibration.json')
    write(args.output/audit_rel,audit)
    deps=[]
    for c in captures:
        dest=args.output/c['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(c['data'])
        deps.append(dict(path=c['path'],sha256=c['sha256']))
    config['zoom_derivation']=dict(path=audit_rel.as_posix(),sha256=sha(args.output/audit_rel),dependencies=deps)
    for name,r in config['targets'].items():
        minimum=r['scales'][0]
        second=round(minimum*factor,6)
        r['scales']=[minimum,second]
        r['scale_modes']={'minimum':minimum,'50_percent':second}
    write(args.output/'recognition_scales.json',config)
    if args.publish:
        if sha(config_path)!=before:
            raise ValueError('标定期间正式识别配置已变化')
        read_scale_overrides(profile.clean_root, offline=True)
        for rel in [audit_rel,*[Path(c['path']) for c in captures]]:
            dest=base/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                raise ValueError(f'已有同名证据，未覆盖：{dest}')
        (args.output/'recognition_scales_before_publish.json').write_bytes(config_path.read_bytes())
        for rel in [audit_rel,*[Path(c['path']) for c in captures]]:
            (base/rel).write_bytes((args.output/rel).read_bytes())
        write(config_path,config)
    print(json.dumps(dict(factor=factor,relative_zoom=1/factor,pairs=len(links),
        rejected=len(rejected),targets=len(records),published=args.publish)))


if __name__=='__main__':
    main()
