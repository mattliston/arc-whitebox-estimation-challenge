import zipfile
from common import *
verify();g=read(PILOT/'assessment_guard.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
r=read(PILOT/'result.json');assert r['verification_passed'] and r['arms'][1]['passed']
old=P2/'codex_resume_20260919_v1/incoming_third_pair_head_v1'
def layer(path,key,l):
    with zipfile.ZipFile(path) as z,z.open(key+'.npy') as f:
        v=np.lib.format.read_magic(f)
        rd=np.lib.format.read_array_header_1_0 if v==(1,0) else np.lib.format.read_array_header_2_0
        shape,order,dtype=rd(f);assert not order and not dtype.hasobject and shape[0]==16
        n=math.prod(shape[1:])*dtype.itemsize;f.seek(f.tell()+l*n)
        raw=f.read(n);assert len(raw)==n
        return np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).astype('float64')
base={x['mini']:x for x in read(PREV/'capture/complete.json')['rows']}
out=HERE/'references';out.mkdir();rows=[]
for mini in MINIS:
    b=base[mini];assert pin(b['weights']['path'])==b['weights'] and pin(b['packet']['path'])==b['packet']
    e=read(old/f'download_mini{mini:04d}.json')['payload'];assert pin(e['path'])==e
    with np.load(e['path']) as z:
        count=int(z['n_samples'].item());ordinal=int(z['global_index'].item());name=z['mlp_name'].item()
        if isinstance(name,bytes):name=name.decode()
    assert count==100_000_000 and ordinal==mini and name==b['identity']['mlp_name']
    W=np.load(b['weights']['path'],mmap_mode='r');packet={};cells=[]
    for l in LAYERS:
        m=layer(e['path'],'mean',l);M=layer(e['path'],'M11',l);m2=layer(e['path'],'post_m2',l)
        pm=layer(e['path'],'pre_mean',l+1);pM=layer(e['path'],'pre_M11',l+1);pm2=layer(e['path'],'pre_m2',l+1)
        C=M-np.outer(m,m);pc=pM-np.outer(pm,pm);w=W[l+1].astype('float64')
        dm=max(float(np.max(abs(np.diag(M)-m2)/(1+abs(m2)))),float(np.max(abs(np.diag(pM)-pm2)/(1+abs(pm2)))))
        em=float(np.linalg.norm(w.T@m-pm)/np.linalg.norm(pm));ec=float(np.linalg.norm(w.T@C@w-pc)/np.linalg.norm(pc))
        assert dm<=2e-6 and em<=1e-4 and ec<=1e-4
        assert np.isfinite(C).all() and np.max(abs(C-C.T))<1e-12 and np.min(np.diag(C))>=0
        packet[f'C_{l}']=C;cells.append(dict(layer=l,diagonal_error=dm,affine_mean_relative=em,affine_covariance_relative=ec))
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**packet)
    rows.append(dict(mini=mini,packet=pin(p),reference=e,identity=b['identity'],cells=cells,final_targets_read=False))
    print(json.dumps(dict(event='reference',mini=mini)),flush=True)
once(HERE/'references.json',dict(rows=rows,final_targets_read=False))
