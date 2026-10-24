from pathlib import Path
import json,hashlib,gc
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'covariance_replay_seal.json')['sources']:assert pin(e['path'])==e
names=('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus')
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
entries={name:{r['mini']:r for r in read(HERE/name/'complete.json')['rows']} for name in names}
def metrics(values):
    z,p,m,ph,mh=[np.asarray(v,dtype='float64') for v in values]
    rh=(p-m)/.25;rl=(ph-mh)/.125
    den=float(np.linalg.norm(rl));pair=float(np.linalg.norm(p-m))
    return dict(secant_relative=float(np.linalg.norm(rh-rl)/den) if den else 0.,
        response_rms=float(np.sqrt(np.mean((.125*rl)**2))),
        curvature=float(np.linalg.norm(p+m-2*z)/pair) if pair else 0.)
def transport(C,w,dtype):
    X=C.astype(dtype);W=w.astype(dtype);first=X@W;h=512
    result=np.empty((1024,1024),dtype=dtype)
    top=W.T[:h]@first;result[:h]=top;result[h:,:h]=top[:,h:].T
    result[h:,h:]=W.T[h:]@first[:,h:]
    return result.astype('float64')
rows=[];out=HERE/'covariance_replay';out.mkdir()
for mini in (57,68):
    wentry=base[mini]['weights'];assert pin(wentry['path'])==wentry
    w=np.load(wentry['path'],mmap_mode='r')[9].copy().astype('float32')
    inputs={};outputs={name:{} for name in names};checks=[];packet={}
    for name in names:
        e=entries[name][mini]['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:C=z['C8'].copy();native=z['Cpre9'].copy()
        inputs[name]=C;outputs[name]['native']=native
        for key,dtype in [('classical32','float32'),('classical64','float64')]:outputs[name][key]=transport(C,w,dtype)
        reference=outputs[name]['classical64']
        checks.append(dict(name=name,
            native_full_error_relative=float(np.linalg.norm(native-reference)/np.linalg.norm(reference)),
            native_diagonal_error_rms=float(np.sqrt(np.mean(np.diag(native-reference)**2))),
            classical32_full_error_relative=float(np.linalg.norm(outputs[name]['classical32']-reference)/np.linalg.norm(reference)),
            classical32_diagonal_error_rms=float(np.sqrt(np.mean(np.diag(outputs[name]['classical32']-reference)**2)))))
    fields={key:dict(full=metrics([outputs[name][key] for name in names]),diagonal=metrics([np.diag(outputs[name][key]) for name in names])) for key in outputs['zero']}
    independent=transport((inputs['d0_plus']-inputs['d0_minus'])/.25,w,'float64')
    actual=(outputs['d0_plus']['classical64']-outputs['d0_minus']['classical64'])/.25
    independent_gap=float(np.linalg.norm(actual-independent)/np.linalg.norm(independent))
    assert independent_gap<=1e-6,independent_gap
    native_response=(outputs['d0_plus']['native']-outputs['d0_minus']['native'])/.25
    response_errors=dict(full=float(np.linalg.norm(native_response-independent)/np.linalg.norm(independent)),
        diagonal=float(np.linalg.norm(np.diag(native_response-independent))/np.linalg.norm(np.diag(independent))))
    qualified=all(fields['classical32'][scope]['secant_relative']<=.05 and fields['classical32'][scope]['curvature']<=.10 for scope in ('full','diagonal'))
    for name in names:
        for key,val in outputs[name].items():packet[name+'_'+key+'_diag']=np.diag(val).copy()
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**packet)
    row=dict(mini=mini,fields=fields,checks=checks,independent_linear_check_relative=independent_gap,
        native_central_response_error=response_errors,classical32_locally_qualified=qualified,packet=pin(p))
    rows.append(row);print(json.dumps(row),flush=True)
    del inputs,outputs,packet;gc.collect()
once(HERE/'covariance_replay.json',dict(rows=rows,classical32_locally_qualified=all(r['classical32_locally_qualified'] for r in rows),
    final_targets_read=False,native_kernel_replaced=False,final_objective_qualified=False))
