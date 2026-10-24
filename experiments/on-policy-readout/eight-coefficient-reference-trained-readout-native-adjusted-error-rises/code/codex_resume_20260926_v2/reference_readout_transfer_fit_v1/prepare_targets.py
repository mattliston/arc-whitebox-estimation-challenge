import zipfile
from common import *
verify();guard('assemble')
for e in read(HERE/'fit_seal.json')['sources']:assert pin(e['path'])==e
frozen=read(HERE/'features_frozen.json');assert frozen['all_directions_retained'] and not frozen['final_targets_read']
assert pin(frozen['packet']['path'])==frozen['packet']
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
def final_array(path):
    with zipfile.ZipFile(path) as z,z.open('gt_mean.npy') as f:
        version=np.lib.format.read_magic(f)
        parser=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=parser(f)
        assert shape==(16,1024) and not fortran and not dtype.hasobject
        size=1024*dtype.itemsize;f.seek(f.tell()+15*size);raw=f.read(size);assert len(raw)==size
        return np.frombuffer(raw,dtype=dtype).astype('float64')
ys=[];rows=[]
for mini in TRAIN:
    receipt=read(OLD/f'download_mini{mini:04d}.json');assert receipt['passed'] and receipt['ordinal']==mini and receipt['role']=='train'
    e=receipt['payload'];assert pin(e['path'])==e
    b=base[mini];assert pin(b['weights']['path'])==b['weights']
    with np.load(e['path']) as z:
        count=int(z['n_samples'].item());ordinal=int(z['global_index'].item());name=z['mlp_name'].item()
        if isinstance(name,bytes):name=name.decode()
    assert count==100_000_000 and ordinal==mini and name==b['identity']['mlp_name']
    y=final_array(e['path']);assert y.shape==(1024,) and np.isfinite(y).all() and np.min(y)>=0
    ys.append(y);rows.append(dict(mini=mini,identity=b['identity'],weights=b['weights'],archive=e,
        target_key='gt_mean[15]',target_provenance='copied public1B final means; archive higher moments have100M samples'))
path=HERE/'targets.npz'
with path.open('xb') as f:np.savez(f,targets=np.stack(ys),train=np.array(TRAIN,dtype='int64'))
once(HERE/'targets.json',dict(packet=pin(path),train=list(TRAIN),rows=rows,check_targets_read=False,
    features_frozen_before_target_decode=True,features=pin(HERE/'features_frozen.json')))
print(json.dumps(dict(event='TRAIN_final_targets_prepared',packet=pin(path),train=list(TRAIN))),flush=True)
