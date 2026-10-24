import datetime
from common import HERE,D,P2,N,E,PARENT,NATIVE,TRAIN,CHECK,PANEL,CONFIRM,read,write,pin,verify

old=read(N/'inputs.json');train=old['training_parent'];assert [r['rank'] for r in train]==TRAIN
for r in train:
    if r['rank'] in [36,38]:
        receipt=N/f"baseline_replay_audit_v1/base{r['rank']}/result.json";c=read(receipt)['records'][0]
        r['output']=pin(c['output']);r['receipt']=pin(receipt);r['source']=pin(PARENT)
    for k in ['output','receipt','source','truth']:verify(r[k])
panel=[];confirmation=[]
for stage,ranks,out in [('panel',PANEL,panel),('confirmation',CONFIRM,confirmation)]:
    existing={r['rank']:r for r in read(E/f'{stage}_predictions_sealed.json')['rows']}
    for rank in ranks:
        r=existing[rank];c=r['call'];assert c['includes_initialization']
        x=dict(rank=rank,output=pin(c['output']),flops=c['flops'],weights_sha256=c['weights_sha256'],truth=r['truth'],source=pin(E/'candidate.py'))
        verify(x['truth']);out.append(x)
check=[dict(rank=r,truth=pin(P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz')) for r in CHECK]
write(HERE/'inputs.json',dict(train=train,check=check,panel=panel,confirmation=confirmation,target_arrays_decoded=False))
files=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'inputs.json',PARENT,NATIVE,D/'run_guarded.py',
    D/'official_best_331684.json',N/'closeout_after_replay_v1.json',N/'unit.json',D/'full_lower_covariance_edge_packed_v1/closeout.json']
assert all(p.exists() for p in files)
pins=[pin(p) for p in sorted(files)]
write(HERE/'sources_sealed.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pins=pins,
    targets_not_decoded=True,numerical_optimization_not_started=True))
print(dict(source_pins=len(pins),train=TRAIN,check=CHECK,panel=PANEL,confirmation=CONFIRM),flush=True)
