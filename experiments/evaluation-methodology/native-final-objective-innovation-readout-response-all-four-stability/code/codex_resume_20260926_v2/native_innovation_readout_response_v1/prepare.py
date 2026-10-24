import gc,time
from common import *
verify();P={}
for name in ('model.npz','normalizers.npz'):
    with np.load(FIT/name) as z:P.update({k:z[k].copy() for k in z.files})
base={x['mini']:x for x in read(BASE/'capture/complete.json')['rows']}
pred={x['mini']:x['packet'] for x in read(FIT/'fit_result.json')['rows']}
G=np.zeros((32,32));scale=0.;rows=[];start=time.monotonic()
for mini in TRAIN:
    a=packet(base[mini]['packet'],[f'{k}_{l}' for l in LAYERS for k in ('F','C','K','sigma','m')]);z=packet(pred[mini])
    for l in LAYERS:
        s=a[f'sigma_{l}'];m=a[f'm_{l}'];ss=np.outer(s,s)
        F=np.clip(a[f'F_{l}']/P['feature_scale'],-8,8)
        R=(a[f'C_{l}']/ss).astype('float32')/P['rho_scale']
        S=(a[f'K_{l}']/np.outer(s*s,s)).astype('float32')/P['third_scale']
        L=F@P['left.weight'].T;B=F@P['right.weight'].T;H=np.empty((1024,1024,32),dtype='float32')
        for lo in range(0,1024,64):
            hi=lo+64;v=L[lo:hi,None,:]+B[None,:,:]+P['bias']
            v=v+R[lo:hi,:,None]*P['edge.weight'][:,0]
            v=v+S[lo:hi,:,None]*P['edge.weight'][:,1]
            v=v+S[:,lo:hi].T[:,:,None]*P['edge.weight'][:,2]
            H[lo:hi]=np.tanh(v)
        X=np.empty((len(UPPER[0]),32),dtype='float64')
        for j in range(32):
            A=.001*ss*(H[:,:,j].astype('float64')+H[:,:,j].T.astype('float64'))
            Q=project(A,m);assert np.max(abs(Q@(m/np.linalg.norm(m))))<1e-10
            X[:,j]=(Q/ss)[UPPER]
        cellG=2*(X.T@X)/(1024**2)
        Dupper=z[f'D_{l}_upper'];cellscale=float(2*np.sum((Dupper/ss[UPPER])**2)/(1024**2))
        # Replay the old output including its separate bias basis.
        old=(H@P['output.weight'].T+P['output.bias'])[:,:,0]
        replay=project(.001*ss*(old.astype('float64')+old.T.astype('float64')),m)[UPPER]
        err=float(np.linalg.norm(replay-Dupper)/np.linalg.norm(Dupper));assert err<2e-5
        G+=cellG;scale+=cellscale;rows.append(dict(mini=mini,layer=l,discarded_output_scale=cellscale,replay_relative=err))
        print(json.dumps(dict(event='gram_cell',mini=mini,layer=l,elapsed_s=time.monotonic()-start)),flush=True)
        del H,X,A,Q,old,replay;gc.collect()
    del a,z;gc.collect()
G/=len(rows);scale/=len(rows);values,vectors=np.linalg.eigh(G);order=np.argsort(values)[::-1]
values=values[order];vectors=vectors[:,order];assert np.min(values[:8])>0 and scale>0
for j in range(8):
    if vectors[np.argmax(abs(vectors[:,j])),j]<0:vectors[:,j]*=-1
V=vectors[:,:8]*np.sqrt(scale/values[:8])[None,:]
np.testing.assert_allclose(V.T@G@V,scale*np.eye(8),rtol=1e-8,atol=scale*1e-9)
p=HERE/'basis.npz'
with p.open('xb') as f:np.savez(f,gram=G,eigenvalues=values,directions=V,source_scale=np.asarray(scale))
once(HERE/'basis_frozen.json',dict(packet=pin(p),rows=rows,retained_fraction=float(values[:8].sum()/values.sum()),
    output_weights_discarded=True,output_bias_zero=True,final_targets_read=False,check_labels_read=False,
    model=pin(FIT/'model.npz'),normalizers=pin(FIT/'normalizers.npz'),elapsed_s=time.monotonic()-start))
print(json.dumps(dict(event='basis_frozen',retained_fraction=float(values[:8].sum()/values.sum()),source_scale=scale,eigenvalues=values[:8].tolist())),flush=True)
