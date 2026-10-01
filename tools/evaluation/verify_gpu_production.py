"""Small production GPU/CPU integration replay; requires working CUDA."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.common.paths import TOOL_CACHE_ROOT
from tools.evaluation.benchmark_first_registration import output_path
from src.app_catalog import MapCatalog
from src.projection_workflow import process_screenshot, switch_result_version
from src.registration.multiscale_structure.runtime import initialize, prewarm_template
from src.registration_core.utils import imread_unicode


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()
                if k not in ('timings_ms', 'elapsed_ms', 'registration_backend')}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args();args.output=args.output or output_path("gpu_verification", "auto");cv2.setNumThreads(4)
    status=initialize()
    if status['effective_backend']!='cuda':
        raise RuntimeError(f'Cannot validate GPU execution: {status}')
    catalog=MapCatalog();profile=catalog.profile('a_bringer_of_doom','nightmare')
    records=[dict(path='tests/fixtures/registration/game2.png',layout='layout_05',floor='2nd')]
    for name in ('layout_16_b1_semantic_positive','layout_13_1st_narrow','layout_15_b1_narrow'):
        path=ROOT/f'tests/fixtures/registration/multiscale_structure/{name}.json'
        record=json.loads(path.read_text(encoding='utf8'))
        records.append(dict(path=path.with_suffix('.png').relative_to(ROOT).as_posix(),
                            layout=record['layout'],floor=record['floor']))
    rows=[]
    for row in records:
        image=imread_unicode(ROOT/row['path']);prepared={};assets={}
        def run(backend):
            return process_screenshot(image,catalog=catalog, template_cache_root=TOOL_CACHE_ROOT / 'route_projection_templates',profile=profile,model_bundle=None,
                floor=row['floor'],layout_override=row['layout'],route_versions={},opacity=.72,
                prepared_cache=prepared,asset_cache=assets,registration_backend=backend)
        cpu=run('cpu');source=next(iter(prepared.values()))
        assert not hasattr(source,'_gpu_structure_prepared')
        prewarm_template(source)
        gpu=run('auto');cpu_after=run('cpu')
        assert source._gpu_structure_prepared.processed is not source.processed
        assert not hasattr(source.processed,'_cuda_sparse_fields')
        assert gpu.registration.validation['registration_backend']['effective_backend']=='cuda'
        assert normalized(cpu.registration.diagnostics())==normalized(gpu.registration.diagnostics())==normalized(cpu_after.registration.diagnostics())
        versions={}
        if cpu.accepted:
            for variant in catalog.variants(profile,row['layout']):
                options=dict(catalog=catalog,profile=profile,variant_id=variant.variant_id,
                             opacity=.72,asset_cache=assets,prepared_cache=prepared)
                a=switch_result_version(cpu,**options);b=switch_result_version(gpu,**options)
                assert np.array_equal(a.layer,b.layer) and np.array_equal(a.projection,b.projection)
                versions[variant.variant_id]=hashlib.sha256(b.layer.tobytes()+b.projection.tobytes()).hexdigest()
        else:
            assert gpu.layer is None and gpu.projection is None
        rows.append({**row,'accepted':gpu.accepted,'identical':True,'projection_hashes':versions})
        print(row['layout'],row['floor'],'accepted='+str(gpu.accepted),'identical=True',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(device=status['device'],rows=rows),ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
