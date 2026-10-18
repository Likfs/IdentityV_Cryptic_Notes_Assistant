"""Import saved scales after checking source-image content fingerprints."""
from pathlib import Path
import argparse
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True,help='包含 scale_config.json、config.json 和 images 的导入目录')
    parser.add_argument('--output', type=Path, default=ROOT/'workspace/asset_build/maintenance/configs/template_matching.json', help='审核后再发布到正式配置')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scales=json.loads((args.source/'scale_config.json').read_text(encoding='utf-8'))['maps']
    options=json.loads((args.source/'config.json').read_text(encoding='utf-8'))
    images={}
    for path in (args.source/'images').rglob('*'):
        if path.is_file():images.setdefault(sha(path),[]).append(path)
    entries={}
    for calibration in sorted((ROOT/'maintenance/maps').glob('*/*/routes/route_zhanshi_fast/*/calibration.json')):
        data=json.loads(calibration.read_text(encoding='utf-8'));source=ROOT/data['route']
        matches=images.get(sha(source),[])
        if len(matches)!=1:raise ValueError(f'{data["layout"]}: source image missing or ambiguous')
        source_key=matches[0].relative_to(args.source).as_posix()
        saved=scales.get(source_key,{}).get('scale')
        if saved is None or float(saved)<=0:raise ValueError(f'{source_key}: saved scale missing')
        for floor,item in data['floors'].items():
            if item['status']!='PASS':raise ValueError('Calibration not PASS')
            entries[item['template']]=dict(layout=data['layout'],floor=floor,source_reference=source_key,
                source_image=data['route'],source_sha256=sha(source),saved_scale=float(saved),
                clean_sha256=sha(ROOT/item['template']),calibration=calibration.relative_to(ROOT).as_posix(),
                calibration_sha256=sha(calibration),clean_to_original=item['clean_to_route_homography'])
    manifest=dict(schema_version=1,scale_source_sha256=sha(args.source/'scale_config.json'),
        perf_mode=int(options.get('perf_mode',3)),capture=options.get('capture',{}),entries=entries)
    args.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Imported {len(entries)} floors from {len(set(e["source_reference"] for e in entries.values()))} byte-identical originals')

if __name__=='__main__':main()
