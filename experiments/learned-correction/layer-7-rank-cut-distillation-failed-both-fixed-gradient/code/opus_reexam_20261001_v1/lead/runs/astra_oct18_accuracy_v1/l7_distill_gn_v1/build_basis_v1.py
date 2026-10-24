"""Four fixed gradient directions; no forward, truth objective, or optimization."""
import ast,base64,hashlib,json,zlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
H=Path(__file__).resolve().parent
plan=json.loads((H/'PLAN_v1.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,d in plan['source_pins'].items():assert sha(p)==d,p
manifest=json.loads(Path(plan['gradient_manifest']).read_text())
assert manifest['status']=='ALL_FOUR_DISTILLATION_GRADIENTS_COMPLETE_AND_RETIRED'
assert len(manifest['rows'])==len(plan['ranks'])==4
tree=ast.parse(Path(plan['native_branch']).read_text())
literal=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='_O18_LATE_ENCODER_LITERAL' for t in n.targets)))
base={k:np.frombuffer(zlib.decompress(base64.b64decode(v[1])),dtype='<f4').reshape(v[0]).copy()
      for k,v in literal.items()}
assert len(base)==6
rms={k:float(np.sqrt(np.mean(v.astype('d')**2))) for k,v in base.items()}
assert min(rms.values())>0
out=H/'basis_v1';out.mkdir()
np.savez(out/'base_arrays.npz',**base)
rows=[]
for seed,row in zip(plan['ranks'],manifest['rows']):
    assert row['rank']==seed and sha(row['result'])==row['result_sha256']
    g=json.loads(Path(row['guard']).read_text())
    assert g['whole_group_retired'] and g['completed'] and g['exit_code']==0
    record=torch.load(row['result'],map_location='cpu',weights_only=False)
    assert record['scale']==1e8 and record['training_target_kind']=='frozen_full384_current_port_prediction'
    assert set(record['grads'])=={'late.'+k for k in base}
    direction={k:-record['grads']['late.'+k].numpy().astype('d')/1e8*rms[k]**2 for k in base}
    rate=max(float(np.sqrt(np.mean(v*v))/rms[k]) for k,v in direction.items())
    assert rate>0 and np.isfinite(rate)
    # Fixed float32 guard from the prior point builder; no outcome-based scale.
    scale=(plan['probe_max_relative_rms']-4*np.finfo(np.float32).eps)/rate
    arrays={k:np.ascontiguousarray(base[k].astype('d')+scale*direction[k],dtype='<f4') for k in base}
    change={k:float(np.sqrt(np.mean((arrays[k].astype('d')-base[k])**2))/rms[k]) for k in base}
    assert max(change.values())<=plan['probe_max_relative_rms']
    path=out/f'basis_{seed}.npz';np.savez(path,**arrays)
    rows.append(dict(direction_seed=seed,arrays=str(path),sha256=sha(path),normalization=scale,
                     relative_changes=change,gradient_result_sha256=row['result_sha256']))
with (out/'BASIS_v1.json').open('x') as f:
    json.dump(dict(status='FOUR_FIXED_GRADIENT_DIRECTIONS',base=str(out/'base_arrays.npz'),
        base_sha256=sha(out/'base_arrays.npz'),rms=rms,rows=rows,
        coordinates='One coordinate unit means the actual stored float32 basis perturbation.',
        no_model_run=True),f,indent=2)
print('FOUR_FIXED_BASIS_READY')
