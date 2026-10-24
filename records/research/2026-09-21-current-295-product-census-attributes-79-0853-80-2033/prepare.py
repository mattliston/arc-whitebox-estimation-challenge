import ast
from common import *

assert sha(SOURCE)==PARENT_SHA and read(HERE/'build.json')['passed']
for p in HERE.glob('*.py'):ast.parse(p.read_text())
inputs=[SOURCE,D/'run_guarded.py',D/'official_best_331684.json',
        D/'lazy_cost_census_v1/census.py',D/'lazy_cost_census_v1/COST_REPORT.md',
        D/'lazy_mixed_fmm_reprice_v3/reprice.py',D/'lazy_mixed_fmm_reprice_v3/verify_paths.py',
        D/'lazy_mixed_fmm_reprice_v3/RESULT.md',D/'lazy_mixed_fmm_reprice_v3/BATCHING_SCOPE.md',
        A/'smirnov_executable_transpose_certificate_v1.json',A/'rectangular_catalog_cyclic_certificate_v1.json',
        A/'SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_V1_CLOSED.md',
        A/'DENSE_GROUPED_FMM_COST_PREFLIGHT_2026_09_07_V1.md']
controls=[]
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=read(store/'manifest.json')['roles']['observer_fit']['row_manifests']
for rank,name in ((35,'parent_triple'),(100,'parent_cold100')):
    rp=D/f'full_lower_multilayer_mean_observer_v1/{name}/result.json';data=read(rp);rec=data['records'][0]
    assert data['candidate_sha256']==PARENT_SHA and rec['rank']==rank and rec['includes_initialization']
    weights=store/Path(manifest[rank]['path']).parent/'weights.npy'
    assert sha(weights)==rec['weights_sha256'] and sha(rec['output'])==rec['output_sha256']
    inputs.extend([rp,Path(rec['output']),weights]);controls.append(dict(rank=rank,result=pin(rp),flops=rec['flops'],prediction_sha256=rec['output_sha256']))
write(HERE/'controls.json',dict(rows=controls))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.glob('*.py'))]+[pin(HERE/n) for n in ('PROTOCOL.md','build.json','controls.json')]+[pin(p) for p in inputs],targets_opened=False,parent_sha256=PARENT_SHA))
print('Sources and current cold controls sealed before both predictions.')
