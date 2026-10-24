"""Offline owned snapshots and FP64 references; never a prediction supplier."""
import math
import numpy as np
ROW_BLOCK=64
SNAPSHOT_LIMIT=250_000_000
WORKSPACE_LIMIT=400_000_000
def norm(x):return float(np.linalg.norm(x))
def ratio(a,b):
    a=float(a);b=float(b)
    return a/b if b else (0.0 if a==0 else None)
def matrix(x):
    a=np.asarray(x);assert a.ndim==2 and a.dtype==np.float32
    return a
def snapshot(left,right,lam,indices,state):
    left=matrix(left);right=matrix(right);a=np.asarray(lam)
    old=int(state['columns']);width=left.shape[1];idx=np.asarray(indices)
    assert right.shape[1]==width and a.shape==(width,) and a.dtype==np.float32
    assert 0<=old<=width and idx.shape==(len(right),) and np.issubdtype(idx.dtype,np.integer)
    inputs=dict(left=left,young=right[:,old:],lam=a)
    metadata=dict(old=old,split=None,mode='young_only')
    if old:
        basis=matrix(state['pre_basis']);assert np.all((idx>=0)&(idx<len(basis)))
        group=state['lazy']
        if group is None:
            inputs['F']=matrix(state['F']);assert inputs['F'].shape==(basis.shape[1],old)
            metadata['mode']='materialized'
        else:
            split=int(group['prefix']);assert 0<split<old and int(group['joined'])==old-split
            inputs.update(base=matrix(group['base']),join=matrix(group['join']),map=matrix(group['map']))
            rank=basis.shape[1]
            assert inputs['base'].shape==(rank,split) and inputs['join'].shape==(rank,old-split) and inputs['map'].shape==(rank,rank)
            metadata.update(mode='lazy',split=split)
        basis_bytes=len(idx)*basis.shape[1]*4
    else:basis_bytes=0
    size=sum(v.nbytes for v in inputs.values())+basis_bytes
    assert size<=SNAPSHOT_LIMIT,(size,SNAPSHOT_LIMIT)
    # Peak includes all persistent snapshots, widened right factors, a row tile,
    # two full output references, discrepancy and conservative small temporaries.
    outcells=len(left)*len(right)
    factor_cells=max(v.size for k,v in inputs.items() if k not in ('left','lam'))
    workspace=size+24*factor_cells+32*ROW_BLOCK*width+64*outcells+16_000_000
    assert workspace<=WORKSPACE_LIMIT,(workspace,WORKSPACE_LIMIT)
    copied={k:np.array(v,copy=True,order='C') for k,v in inputs.items()}
    if old:copied['basis']=np.array(basis[idx,:],copy=True,order='C')
    assert all(np.isfinite(v).all() for v in copied.values())
    assert all(not np.shares_memory(v,inputs[k]) for k,v in copied.items() if k in inputs)
    for v in copied.values():v.flags.writeable=False
    return dict(**metadata,arrays=copied,snapshot_bytes=size,workspace_upper_bytes=workspace)
def reference(s,column_block=None):
    arrays=s['arrays'];L=arrays['left'];a=arrays['lam'].astype(np.float64);old=s['old']
    n=len(arrays['young']);out=np.empty((len(L),n),np.float64)
    basis=arrays['basis'].astype(np.float64) if old else None
    transform=arrays['map'].astype(np.float64) if s['mode']=='lazy' else None
    def contraction(left,right,coeff):
        width=left.shape[1];answer=np.zeros((len(left),len(right)),np.float64)
        step=width if column_block is None else column_block
        if width==0:return answer
        for begin in range(0,width,step):
            end=min(begin+step,width)
            x=left[:,begin:end].astype(np.float64)
            r=right[:,begin:end].astype(np.float64)*coeff[begin:end]
            product=(x*x)@r.T
            if column_block is None:answer=product
            else:answer+=product
        return answer
    for start in range(0,len(L),ROW_BLOCK):
        left=L[start:start+ROW_BLOCK]
        young=contraction(left[:,old:],arrays['young'],a[old:])
        if old:
            if s['mode']=='materialized':inner=contraction(left[:,:old],arrays['F'],a[:old])
            else:
                split=s['split']
                inner=(contraction(left[:,:split],arrays['base'],a[:split])@transform.T
                       +contraction(left[:,split:old],arrays['join'],a[split:old]))
            out[start:start+len(left)]=young+inner@basis.T
        else:out[start:start+len(left)]=young
    assert np.isfinite(out).all()
    return out
def scalar_reference(s):
    """Tiny independent reconstruction and scalar fsum; no factorized read."""
    ar=s['arrays'];a=ar['lam'];left=ar['left'].astype(np.float64)
    if s['old']:
        if s['mode']=='materialized':right=ar['basis'].astype(np.float64)@ar['F'].astype(np.float64)
        else:
            b=ar['basis'].astype(np.float64)
            right=np.concatenate([(b@ar['map'].astype(np.float64))@ar['base'].astype(np.float64),b@ar['join'].astype(np.float64)],axis=1)
        right=np.concatenate([right,ar['young'].astype(np.float64)],axis=1)
    else:right=ar['young'].astype(np.float64)
    return np.array([[math.fsum(float(a[k])*float(left[i,k])**2*float(right[j,k]) for k in range(len(a)))
                      for j in range(len(right))] for i in range(len(left))],np.float64)
def measure(actual,s,response,live,width):
    ref=reference(s);alt=reference(s,257);denom=norm(ref);referr=norm(alt-ref)
    agreement=ratio(referr,denom);assert agreement is not None and agreement<=1e-8,agreement
    actual=np.asarray(actual,dtype=np.float64);assert actual.shape==ref.shape and np.isfinite(actual).all()
    delta=ref-actual
    diagonal=np.asarray(response)[:,None]==np.asarray(live)[None,:]
    effective=delta.copy();effective[diagonal]=0
    er=norm(delta);eff=norm(effective)
    clean_ref=ref.copy();clean_ref[diagonal]=0
    full=np.zeros((width,width),np.float64)
    full[np.asarray(response)[:,None],np.asarray(live)[None,:]]=effective
    fields=dict(shape=list(actual.shape),mode=s['mode'],old_columns=s['old'],total_columns=len(s['arrays']['lam']),
        reference_norm=denom,absolute_error_norm=er,relative_error=ratio(er,denom),
        reference_disagreement_norm=referr,reference_relative_disagreement=agreement,
        reference_error_to_measured_error=ratio(referr,er),effective_error_norm=eff,
        effective_reference_norm=norm(clean_ref),effective_read_relative_error=ratio(eff,norm(clean_ref)),
        maxabs=float(np.max(np.abs(delta),initial=0)),diagonal_entries_excluded=int(diagonal.sum()),
        snapshot_bytes=s['snapshot_bytes'],workspace_upper_bytes=s['workspace_upper_bytes'])
    return fields,full
def covariance_response(delta,aux,cov):
    d=np.asarray(aux['pdf'],dtype=np.float64)/np.asarray(aux['sigma'],dtype=np.float64)
    g=np.asarray(aux['gate'],dtype=np.float64)
    change=(d[:,None]*g[None,:]*delta+g[:,None]*d[None,:]*delta.T)/2
    assert not np.any(np.diag(delta)) and not np.any(np.diag(change))
    covariance=np.array(cov,dtype=np.float64,copy=True);np.fill_diagonal(covariance,0)
    cn=norm(change);base=norm(covariance)
    return dict(local_covariance_error_norm=cn,original_covariance_offdiag_norm=base,
                local_covariance_relative=ratio(cn,base),local_covariance_maxabs=float(np.max(np.abs(change),initial=0))),change
def audit_read(original,left,right,lam,live,state,response,width,ctx):
    before=int(ctx._flops_used)
    owned=snapshot(left,right,lam,live,state)
    answer=original(left,right,lam,live,state)
    after_original=int(ctx._flops_used)
    row,delta=measure(answer,owned,response,live,width)
    assert int(ctx._flops_used)==after_original
    row['original_read_flops']=after_original-before
    return answer,row,delta
