"""Lossless mandatory optimizer-boundary state; no candidate selection."""
from fit_common import *


def arrays(torch,np,p,optimizer,pkeys):
    result={'p__'+k:v.detach().cpu().contiguous().numpy().copy() for k,v in p.items()}
    assert len(optimizer.state)==len(pkeys)
    for key in pkeys:
        state=optimizer.state[p[key]]
        assert set(state)=={'step','exp_avg','exp_avg_sq'}
        for name,value in state.items():
            result[name+'__'+key]=value.detach().cpu().contiguous().numpy().copy()
    result['rng__cpu']=torch.get_rng_state().numpy().copy()
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def descriptors(values):
    return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in values.items()}


def groups(optimizer):
    assert len(optimizer.param_groups)==1
    return json.loads(json.dumps([{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups],allow_nan=False))


def initial_rng(arm,torch,np):
    path=HERE/'initial_rng.npz'
    if arm=='reference':
        with path.open('xb') as f:np.savez(f,cpu=torch.get_rng_state().numpy().copy())
        write(HERE/'initial_rng.json',dict(payload=pin(path),arm=arm,no_seed_selection=True))
    elif arm=='split0':
        check(read(HERE/'initial_rng.json')['payload'])
        with np.load(path,allow_pickle=False) as z:value=z['cpu'].copy()
        assert value.dtype==np.uint8
        torch.set_rng_state(torch.from_numpy(value))


def save(folder,completed_updates,torch,np,p,optimizer,pkeys):
    values=arrays(torch,np,p,optimizer,pkeys)
    before=descriptors(values);settings=groups(optimizer)
    payload=folder/f'checkpoint_{completed_updates:02d}.npz'
    with payload.open('xb') as f:np.savez(f,**values)
    with np.load(payload,allow_pickle=False) as z:
        assert set(z.files)==set(values)
        for key in values:
            v=z[key]
            assert v.dtype==values[key].dtype and v.shape==values[key].shape and v.tobytes()==values[key].tobytes()
    assert descriptors(arrays(torch,np,p,optimizer,pkeys))==before and groups(optimizer)==settings
    path=folder/f'checkpoint_{completed_updates:02d}.json'
    write(path,dict(complete=True,completed_updates=completed_updates,payload=pin(payload),tensors=before,
        parameter_keys=list(p),trainable_keys=list(pkeys),optimizer_groups=settings,
        round_trip_exact=True,live_values_unchanged=True,mandatory_boundary_only=True,candidate_eligible=False))
    return pin(path)


def restore(path,completed_updates,torch,np,p,optimizer,pkeys):
    record=read(path);check(record['payload'])
    assert record['complete'] and record['completed_updates']==completed_updates
    assert record['parameter_keys']==list(p) and record['trainable_keys']==list(pkeys)
    assert record['round_trip_exact'] and record['live_values_unchanged'] and not record['candidate_eligible']
    assert not optimizer.state
    with np.load(record['payload']['path'],allow_pickle=False) as z:
        values={k:z[k].copy() for k in z.files}
    assert descriptors(values)==record['tensors']
    ownership=[id(v) for v in optimizer.param_groups[0]['params']]
    with torch.no_grad():
        for key,value in p.items():
            source=torch.from_numpy(values['p__'+key])
            assert source.dtype==value.dtype and source.shape==value.shape
            value.copy_(source)
    for key in pkeys:
        optimizer.state[p[key]]={name:torch.from_numpy(values[name+'__'+key].copy()) for name in ['step','exp_avg','exp_avg_sq']}
    settings=record['optimizer_groups'][0]
    assert set(settings)==set(optimizer.param_groups[0])-{'params'}
    for key,value in settings.items():optimizer.param_groups[0][key]=tuple(value) if key=='betas' else value
    torch.set_rng_state(torch.from_numpy(values['rng__cpu']))
    assert [id(v) for v in optimizer.param_groups[0]['params']]==ownership
    assert ownership==[id(p[k]) for k in pkeys]
    assert descriptors(arrays(torch,np,p,optimizer,pkeys))==record['tensors']
    assert groups(optimizer)==record['optimizer_groups']
    return pin(path)
