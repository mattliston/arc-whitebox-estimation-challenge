"""Frozen four-direction finite-Jacobian LS; no MC truth use or model execution."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
plan=json.loads((H/'PLAN_v1.json').read_text());basis=json.loads((H/'basis_v1/BASIS_v1.json').read_text())
manifest=json.loads((H/'forwards_v1/MANIFEST_v1.json').read_text())
assert manifest['status']=='FIXED_TWENTY_COMPLETE_AND_RETIRED' and len(manifest['rows'])==20
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,d in plan['source_pins'].items():assert sha(p)==d,p
assert manifest['plan_sha256']==sha(H/'PLAN_v1.json')
assert manifest['basis_sha256']==sha(H/'basis_v1/BASIS_v1.json')
assert sha(basis['base'])==basis['base_sha256']
records={(r['rank'],r['arm']):r for r in manifest['rows']}
teachers={r['rank']:r for r in json.loads(Path(plan['gradient_manifest']).read_text())['rows']}
columns=[];targets=[]
for seed in plan['ranks']:
    def prediction(arm):
        r=records[seed,arm];assert sha(r['result'])==r['result_sha256']
        # Deliberately access prediction only, never true-MSE fields.
        x=np.asarray(json.loads(Path(r['result']).read_text())['final_prediction'],dtype='d')
        assert x.shape==(1024,) and np.isfinite(x).all()
        return x
    base=prediction('base')
    tr=teachers[seed];assert sha(tr['teacher'])==tr['teacher_sha256']
    t=json.loads(Path(tr['teacher']).read_text());assert sha(t['target'])==t['target_sha256']
    with np.load(t['target'],allow_pickle=False) as z:teacher=z['final_prediction'].copy()
    targets.append(teacher-base)
    columns.append(np.stack([prediction('d'+str(i))-base for i in range(4)],axis=1))
J=np.concatenate(columns);r=np.concatenate(targets)
assert J.shape==(4096,4) and r.shape==(4096,)
with np.load(basis['base'],allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
deltas=[]
for row in basis['rows']:
    assert sha(row['arrays'])==row['sha256']
    with np.load(row['arrays'],allow_pickle=False) as z:
        deltas.append({k:z[k].astype('d')-base[k].astype('d') for k in base})
parameter_basis=np.stack([np.concatenate([(delta[k]/basis['rms'][k]).ravel() for k in base]) for delta in deltas],axis=1)

def solve_and_serialize(A,b):
    u,s,vt=np.linalg.svd(A,full_matrices=False)
    keep=s>plan['svd_relative_cutoff']*s[0]
    assert s[0]>0 and keep.any()
    x=vt[keep].T@((u[:,keep].T@b)/s[keep])
    direction={k:sum(x[i]*deltas[i][k] for i in range(4)) for k in base}
    relative=max(float(np.sqrt(np.mean(v*v))/basis['rms'][k]) for k,v in direction.items())
    trust=min(1.,(plan['trust_max_relative_rms']-4*np.finfo(np.float32).eps)/max(relative,np.finfo(float).tiny))
    x*=trust
    arrays={k:np.ascontiguousarray(base[k].astype('d')+sum(x[i]*deltas[i][k] for i in range(4)),dtype='<f4') for k in base}
    changes={k:float(np.sqrt(np.mean((arrays[k].astype('d')-base[k])**2))/basis['rms'][k]) for k in base}
    assert max(changes.values())<=plan['trust_max_relative_rms']
    actual=np.concatenate([((arrays[k].astype('d')-base[k])/basis['rms'][k]).ravel() for k in base])
    serialized_x=np.linalg.lstsq(parameter_basis,actual,rcond=None)[0]
    outside=actual-parameter_basis@serialized_x
    offspan=float(np.linalg.norm(outside)/max(np.linalg.norm(actual),np.finfo(float).tiny))
    return serialized_x,arrays,dict(singular_values=s.tolist(),kept=keep.tolist(),
        retained_condition=float(s[0]/s[keep][-1]),nominal_coefficients=x.tolist(),
        coefficients_after_F32_serialization=serialized_x.tolist(),trust_scale=trust,
        relative_changes=changes,relative_parameter_offspan_rounding=offspan)

x,arrays,details=solve_and_serialize(J,r)
before=float(r@r);after=float((r-J@x)@(r-J@x));explained=1-after/before
loo=[]
for held in range(4):
    fit=np.ones(4096,dtype=bool);fit[held*1024:(held+1)*1024]=False
    xi,_,di=solve_and_serialize(J[fit],r[fit])
    rh=r[~fit];eh=rh-J[~fit]@xi
    loo.append(dict(rank=plan['ranks'][held],explained_fraction=1-float(eh@eh)/float(rh@rh),
                    retained_condition=di['retained_condition'],trust_scale=di['trust_scale']))
receipt=dict(status='LINEAR_GATE_PASS_HELD_POINT' if explained>=plan['minimum_explained_fraction'] else 'LINEAR_GATE_FAILED_NO_POINT',
             **details,leave_one_network_linear_predictions=loo,
             baseline_teacher_mse=before/4096,predicted_teacher_mse=after/4096,
             explained_fraction=explained,no_MC_truth_for_fit=True,no_CHECK_read=True,
             serialization_note='Gate uses actual F32 point projected onto measured span; off-span rounding is reported, not an observed output.',
             source_sha256=sha(__file__),manifest_sha256=sha(H/'forwards_v1/MANIFEST_v1.json'))
out=H/'solution_v1';out.mkdir()
if explained>=plan['minimum_explained_fraction']:
    np.savez(out/'late_arrays.npz',**arrays)
    receipt.update(point_sha256=sha(out/'late_arrays.npz'))
with (out/'SOLUTION_v1.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
