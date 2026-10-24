import ast
from common import *
assert read(NATIVE/'score.json')['passed'] and read(NATIVE/'qualification.json')['passed']
assert sha(NATIVE/'closeout.json')=='13fca44e27fed298b5948fd6dcc8936f67d64c2f6867b0fbabd087293b81dada'
closeout=read(NATIVE/'closeout.json')
for p in closeout['artifacts']+closeout['external_pins']:verify(p)
assert not read(FIT/'result.json')['passed']
copies=[]
for name in ('native.py','reference.py','build.json','unit.json'):
    with (HERE/name).open('xb') as f:f.write((NATIVE/name).read_bytes())
    assert (HERE/name).read_bytes()==(NATIVE/name).read_bytes()
    copies.append(dict(old=pin(NATIVE/name),new=pin(HERE/name),byte_exact=True))
E=D/'full_lower_covariance_edge_native_v1';P=D/'full_lower_covariance_edge_packed_v1'
packing=read(P/'native_qualified.json');assert packing['passed'] and packing['cold_flop_delta']==packing['warm_flop_delta']==0
assert packing['candidate']==pin(PARENT) and packing['parent']==pin(E/'candidate.py')
direct={r['rank']:r for r in read(D/'covariance_head_spsa_native_v1/baselines_sealed.json')['rows']}
rows={};weights={};external=[]
for stage,roster in (('panel',PANEL),('confirmation',CONFIRM)):
    old={r['rank']:r for r in read(E/f'{stage}_predictions_sealed.json')['rows']};rows[stage]=[]
    for rank in roster:
        x=old[rank];c=x['call'];assert c['includes_initialization'] and c['rank']==rank and c['input_flops_separate']==0
        wp=P2/f'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy'
        wpin=pin(wp);assert wpin['sha256']==c['weights_sha256'];weights[str(rank)]=wpin
        if rank in direct:
            d=direct[rank]
            for k in ('output','receipt','source','truth'):verify(d[k])
            assert d['source']==pin(PARENT) and d['flops']==c['flops'] and d['output']['sha256']==c['output_sha256']
            prediction=d['output'];receipt=d['receipt'];source=d['source'];equivalence=False
        else:
            prediction=pin(c['output']);receipt=x['result'];source=pin(E/'candidate.py');equivalence=True
        assert prediction['sha256']==c['output_sha256'];verify(receipt);verify(x['truth'])
        parent=dict(prediction=prediction,receipt=receipt,source=source,packing_equivalence_used=equivalence,
            flops=c['flops'],multiplier=max(.1,c['flops']/2**41),includes_initialization=True,weights_sha256=c['weights_sha256'])
        rows[stage].append(dict(rank=rank,parent=parent,truth=x['truth'],weights=wpin))
        external += [wp,Path(prediction['path']),Path(receipt['path']),Path(x['truth']['path'])]
    external.append(E/f'{stage}_predictions_sealed.json')
assert not set(PANEL+CONFIRM)&{35,100,150,200,250,300,400,500}
write(HERE/'inputs.json',dict(**rows,weights=weights,candidate=pin(CANDIDATE),bank=pin(PARENT),packing_qualification=pin(P/'native_qualified.json'),targets_not_decoded=True))
write(HERE/'source_copies.json',dict(copies=copies,previous_native_closeout=pin(NATIVE/'closeout.json'),copied_native_harness=True,reused_unit_guard=pin(NATIVE/'unit_watch.json')))
external += [PARENT,CANDIDATE,D/'run_guarded.py',D/'official_best_331684.json',D/'continuation_checkpoint_v81.json',
    NATIVE/'closeout.json',NATIVE/'score.json',NATIVE/'qualification.json',NATIVE/'native.py',NATIVE/'reference.py',
    NATIVE/'unit.json',NATIVE/'unit_watch.json',NATIVE/'build.json',NATIVE/'parameters.npz',NATIVE/'bank_literal.json',
    NATIVE/'PROTOCOL.md',E/'candidate.py',P/'native_qualified.json',P/'packing_build.json',P/'packing_unit.json',
    D/'covariance_head_spsa_native_v1/baselines_sealed.json']
for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)],
    before_panel_calls=True,before_panel_targets=True,rosters_from_prior_native_protocol=True))
print(dict(sealed=True,panel=PANEL,confirmation=CONFIRM,candidate=pin(CANDIDATE)),flush=True)
