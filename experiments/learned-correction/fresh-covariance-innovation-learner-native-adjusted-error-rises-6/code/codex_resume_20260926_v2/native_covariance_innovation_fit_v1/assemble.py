from common import *
verify();guard_ok('prepare');branches={}
for l in range(9,15):
    guard_ok(f'branch{l}')
    r=read(HERE/f'branch{l}/complete.json');assert r['source_inverse_exact'] and r['target_layer']==l
    branches[l]={x['mini']:x for x in r['rows']}
base,refs=tables();out=HERE/'labels';out.mkdir()
rows=read(PILOT/'labels/complete.json')['rows'].copy()
for r in rows:assert pin(r['packet']['path'])==r['packet']
for mini in MINIS:
    b=checked_npz(base[mini]['packet'],[f'{k}_{l}' for l in LAYERS for k in ('C','m')])
    t=checked_npz(refs[mini]['packet']);packet={};cells=[]
    for l in LAYERS:
        old=b[f'C_{l}'];m=b[f'm_{l}'];T=t[f'C_{l}']
        cf=old if l==8 else checked_npz(branches[l][mini]['packet'],['C'])['C']
        J=project(T-cf,m);E=project(T-old,m);D=project(cf-old,m)
        np.testing.assert_allclose(J,E-D,atol=2e-17,rtol=1e-11)
        h=m/np.linalg.norm(m);A=(T-cf+(T-cf).T)/2;np.fill_diagonal(A,0)
        lam=np.linalg.solve(np.diag(1-2*h*h)+np.outer(h,h),A@h)
        independent=A-np.outer(lam,h)-np.outer(h,lam);np.fill_diagonal(independent,0)
        np.testing.assert_allclose(J,independent,atol=5e-17,rtol=5e-10)
        packet[f'J_{l}_upper']=J[UPPER]
        cells.append(dict(layer=l,innovation_energy=float(np.sum(J*J)),cumulative_energy=float(np.sum(E*E)),
            innovation_to_cumulative=float(np.sum(J*J)/np.sum(E*E)),independent_max_error=float(np.max(abs(J-independent)))))
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**packet)
    rows.append(dict(mini=mini,packet=pin(p),cells=cells))
    print(json.dumps(dict(event='labels',mini=mini,ratios=[c['innovation_to_cumulative'] for c in cells])),flush=True)
assert [x['mini'] for x in rows]==list(TRAIN)
once(out/'complete.json',dict(rows=rows,pilot_result=pin(PILOT/'result.json'),
    branch_receipts=[pin(HERE/f'branch{l}/complete.json') for l in range(9,15)],
    final_targets_used=False,check_labels_used=False,all_labels_frozen_before_fit=True))
