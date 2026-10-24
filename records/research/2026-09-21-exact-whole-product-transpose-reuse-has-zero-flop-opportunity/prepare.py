import ast
from common import *
assert sha(SOURCE)=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
cp=D/'continuation_checkpoint_v84.json';assert sha(cp)=='1807b796b1afd77ae9e3fd5b4409e21d00b791b531ea7598b1722004c6974620'
previous=read(cp)
for k in ('study','official_bank','workers','candidate','archive','rejection'):verify(previous[k])
rows=[];external=[SOURCE,cp,D/'run_guarded.py',D/'official_best_331684.json',OLD/'PROTOCOL.md',OLD/'RESULT.md',OLD/'census.py']
for rank in (35,100):
    old_result=OLD/f'rank{rank:04d}/result.json';old=read(old_result)
    assert old['completed'] and old['all16_prediction_bytes_exact'] and old['parent_bill_exact'] and old['parent_sha256']==sha(SOURCE)
    control=Path(old['control_path']);assert sha(control)==old['control_sha256'];ref=read(control)['records'][0]
    assert ref['rank']==rank and ref['includes_initialization'] and ref['flops']==old['complete_flops']
    wp=P2/f'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy'
    assert sha(wp)==ref['weights_sha256'];pp=Path(ref['output']);assert sha(pp)==ref['output_sha256']
    attr=OLD/f'rank{rank:04d}/attribution.json';assert sha(attr)==old['attribution_sha256']
    trace=read(attr);assert len(trace['outermost_mm_calls'])==295 and trace['total_flops']==ref['flops']
    rows.append(dict(rank=rank,weights=pin(wp),prediction=pin(pp),control=pin(control),old_result=pin(old_result),attribution=pin(attr),flops=ref['flops']))
    external += [wp,pp,control,old_result,attr]
external += [Path(previous[k]['path']) for k in ('study','workers','candidate','archive','rejection')]
write(HERE/'inputs.json',dict(rows=rows,source=pin(SOURCE),targets_not_read=True,pending_upload_unchanged=True))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)],
    before_new_calls=True,before_fingerprints=True,no_targets=True))
print(dict(sealed=True,ranks=[35,100],candidate_unchanged=True),flush=True)
