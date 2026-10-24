"""Fixed study bindings; import performs no numerical work."""
from pathlib import Path
import hashlib
import json
import os

HERE=Path(__file__).resolve().parent
H2=HERE.parent
P2=H2.parent
ROOT=P2.parents[3]
BANK=H2/'candidate_old_both_tail_trace_bounded_v6.py'
COMPONENT=H2/'candidate_thin_star_route_v1.py'
TRAIN=(57,68,45,59,3,12)
PARITY=(35,100)
RIDGE=1e-3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1<<20):digest.update(block)
    return digest.hexdigest()


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)


def once(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def verify_pin(entry):assert pin(entry['path'])==entry,entry['path']


def verify():
    for entry in read(HERE/'seal.json')['sources']:verify_pin(entry)
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        assert os.environ.get(variable)=='1',(variable,os.environ.get(variable))
    assert os.getpriority(os.PRIO_PROCESS,0)>=10


def guard(name):
    path=HERE/f'{name}_guard.json';value=read(path)
    assert value['completed'] and value['whole_group_retired'] and not value['remaining_rows']
    assert value['max_swap_growth_bytes']==0
    return pin(path)


def fixture_rows(mode):
    refs=read(HERE/'data_refs.json')
    if mode=='capture':return refs['train']
    assert mode=='parity'
    store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    verify_pin(refs['parity_store_manifest'])
    manifest=read(store/'manifest.json')['roles']['observer_fit']['row_manifests']
    return [dict(rank=rank,weights=dict(path=str(store/Path(manifest[rank]['path']).parent/'weights.npy')),
                 mlp_name=f'phase2-{rank}') for rank in PARITY]
