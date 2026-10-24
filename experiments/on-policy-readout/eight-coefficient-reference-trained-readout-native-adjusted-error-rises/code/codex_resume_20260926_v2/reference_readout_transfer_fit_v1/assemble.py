from common import *
verify();base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
with np.load(BASIS/'basis.npz') as z:directions=z['directions'].copy()
data={};receipts=[];repeats=0
for name in NAMES:
    guard(name);r=read(HERE/name/'complete.json');assert not r['final_targets_read'] and not r['coefficients_fitted']
    assert tuple(x['mini'] for x in r['rows'])==TRAIN;data[name]={}
    if name=='zero':expected=np.zeros((1,32),dtype='float32')
    else:
        d=int(name[1]);amplitude=.125 if name.endswith('plus') else -.125
        expected=(amplitude*directions[:,d]).astype('float32')[None,:]
    for row in r['rows']:
        a=packet(row['packet']);assert a['readout'].tobytes()==expected.tobytes()
        assert row['first9_exact'] and row['switch_restored'] and not row['final_targets_read']
        if row['repeated_prior_case']:assert row['repeated_output_and_bill_exact'];repeats+=1
        data[name][row['mini']]=a['prediction']
    receipts.append(pin(HERE/name/'complete.json'))
assert repeats==10
B=np.stack([np.stack([(data[f'd{d}_plus'][mini][-1]-data[f'd{d}_minus'][mini][-1])/.25 for d in range(8)],axis=-1) for mini in TRAIN])
fast=np.stack([packet(base[mini]['packet'],['prediction'])['prediction'][-1] for mini in TRAIN])
zero=np.stack([data['zero'][mini][-1] for mini in TRAIN])
fastzero={r['mini']:r for r in read(BASIS/'zero/complete.json')['rows']}
overheads=[fastzero[mini]['flops']-base[mini]['flops'] for mini in (57,68)]
assert overheads[0]==overheads[1] and overheads[0]>0
costs=np.array([base[mini]['flops']+overheads[0] for mini in TRAIN],dtype='int64')
weights=costs.astype('float64')/float(np.mean(costs))
assert B.shape==(6,1024,8) and np.isfinite(B).all() and np.isfinite(fast).all() and np.isfinite(zero).all()
rows=[]
for i,mini in enumerate(TRAIN):
    b=packet(base[mini]['packet'],['prediction'])['prediction']
    for name in NAMES:assert data[name][mini][:9].tobytes()==b[:9].tobytes()
    for d in range(8):
        p=data[f'd{d}_plus'][mini][-1];m=data[f'd{d}_minus'][mini][-1];z=zero[i]
        rows.append(dict(mini=mini,direction=d,response_rms=float(np.sqrt(np.mean((.125*B[i,:,d])**2))),
            even_over_odd=float(np.linalg.norm(p+m-2*z)/max(np.linalg.norm(p-m),1e-30))))
path=HERE/'features.npz'
with path.open('xb') as f:np.savez(f,B=B,fast_parent=fast,reference_zero=zero,costs=costs,weights=weights,
    directions=directions,train=np.array(TRAIN,dtype='int64'))
once(HERE/'features_frozen.json',dict(packet=pin(path),train=list(TRAIN),shape=list(B.shape),all_directions_retained=True,
    helper_overhead=overheads[0],capture_receipts=receipts,repeat_controls=10,rows=rows,
    final_targets_read=False,coefficients_fitted=False))
print(json.dumps(dict(event='features_frozen',packet=pin(path),repeat_controls=10,rows=rows)),flush=True)
