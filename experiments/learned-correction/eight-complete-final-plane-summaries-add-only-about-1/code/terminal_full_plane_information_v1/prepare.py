import ast
from common import *
assert sha(PARENT)==PARENT_SHA
capture=read(CAPTURE/'capture/result.json');assert capture['completed'] and capture['repeat_all_arrays_exact'] and capture['extra_parent_operations']==0
assert capture['records'][0]['array_digests']==capture['records'][2]['array_digests']
controls=read(CAPTURE/'controls.json');rows=[]
for rec in capture['records'][:2]:
    rank=rec['rank'];control=controls[str(rank)]
    assert rec['all16_and_bill_exact'] and rec['cache_retired'] and control['includes_initialization']
    assert rec['weights']['sha256']==control['weights_sha256']
    for p in [rec['capture'],rec['weights']]:check(p)
    tp=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    pp=Path(control['output']);assert sha(pp)==control['output_sha256']
    rows.append(dict(rank=rank,capture=rec['capture'],array_digests=rec['array_digests'],weights=rec['weights'],
        parent_prediction=pin(pp),parent_cold_flops=control['flops'],truth=pin(tp)))
assert [r['parent_cold_flops'] for r in rows]==[413448696909,416264792412]
write(HERE/'inputs.json',dict(rows=rows,capture_result=pin(CAPTURE/'capture/result.json'),controls=pin(CAPTURE/'controls.json'),targets_not_decoded=True))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
external=[PARENT,D/'official_best_331684.json',D/'continuation_checkpoint_v74.json',D/'run_guarded.py',
    CAPTURE/'closeout.json',CAPTURE/'capture.py',CAPTURE/'capture/result.json',CAPTURE/'controls.json',CAPTURE/'PROTOCOL.md',CAPTURE/'RESULT.md',
    D/'twohop_terminal_v1/PROTOCOL.md',D/'twohop_terminal_v1/NEGATIVE_RESULT.md',
    D/'terminal_representation_v1/NEGATIVE_RESULT.md',D/'terminal_latent_covariance_information_v3/NEGATIVE_RESULT.md',
    D/'lazy_mean_source_resolution_v1/RESULT.md']
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in external],
    input_arrays_not_yet_decoded=True,targets_not_decoded=True))
print(dict(sealed=True,inputs=pin(HERE/'inputs.json')),flush=True)
