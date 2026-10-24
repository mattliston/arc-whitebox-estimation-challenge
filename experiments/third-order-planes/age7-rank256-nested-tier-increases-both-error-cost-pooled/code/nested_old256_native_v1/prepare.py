import ast
from common import *
assert sha(PARENT)==PARENT_SHA
parent=PARENT.read_bytes();fragment=(HERE/'fragment.py').read_bytes()
candidate=parent+b'\n\n'+fragment
assert candidate[:len(parent)]==parent and candidate[len(parent)+2:]==fragment
ast.parse(candidate.decode())
with (HERE/'candidate.py').open('xb') as f:f.write(candidate)
write(HERE/'build.json',dict(parent=pin(PARENT),fragment=pin(HERE/'fragment.py'),candidate=pin(HERE/'candidate.py'),
    whole_source_inverse_byte_exact=candidate[:-len(fragment)-2]==parent,unchanged_original_prefix=True))
rows=[]
for rank,control in [(35,'parent_triple'),(100,'parent_cold100')]:
    path=D/f'full_lower_multilayer_mean_observer_v1/{control}/result.json';r=read(path)['records'][0]
    wp=P2/f'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy'
    tp=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    assert sha(wp)==r['weights_sha256'] and sha(r['output'])==r['output_sha256']
    rows.append(dict(rank=rank,weights=pin(wp),truth=pin(tp),parent_result=pin(path),parent=r,prediction=pin(r['output'])))
write(HERE/'inputs.json',dict(rows=rows,arrays_not_decoded=True))
external=[PARENT,D/'run_guarded.py',D/'official_best_331684.json',D/'continuation_checkpoint_v73.json',
 A/'current_comp_nested_old224_v1/INDEPENDENT_40_PRICE_AUDIT_20260913_UTC.md',
 D/'rank256_objective_v1/NEGATIVE_RESULT.md',D/'full_lower_cost_frontier_v1/closeout.json']
pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in external]
write(HERE/'sources_sealed.json',dict(pins=pins,targets_not_decoded=True,
    public_sources=['https://github.com/504aldo/whest-p2-cumulant-k3','https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218']))
print(dict(candidate=pin(HERE/'candidate.py'),source_files=len(pins)),flush=True)
