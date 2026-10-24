import time,urllib.request
from common import *
verify();verify('fit_sources_sealed.json');assert read(HERE/'frozen_inputs_completed.json')['completed'];resource_floor()
ordinal=int(sys.argv[1]);r=row(ordinal)
if r['role']=='heldout':
    frozen=read(HERE/'model_frozen.json');assert frozen['training_completed']
    check(frozen['model']);check(frozen['normalizers']);check(frozen['training'])
entry=r['moment_file'];start=time.monotonic();directory=HERE/'public_moments';directory.mkdir(exist_ok=True)
if ordinal==57:
    path=A/'public_p2_higher_moments_pilot_v1/mlp_00057.npz'
    assert path.stat().st_size==entry['bytes'] and sha(path)==entry['lfs_sha256']
    total=0
else:
    assert ordinal in TRAIN+HELDOUT
    path=directory/f'mlp_{ordinal:05d}.npz';assert not path.exists()
    metadata=read(A/'public_p2_moments_metadata_v1.json')
    url=f"https://huggingface.co/datasets/{metadata['repository']}/resolve/{metadata['commit']}/{entry['path']}"
    request=urllib.request.Request(url,headers={'User-Agent':'phase2-directed-third-pilot/1'})
    total=0;digest=hashlib.sha256()
    with urllib.request.urlopen(request,timeout=25) as response,path.open('xb') as output:
        while True:
            b=response.read(8<<20)
            if not b:break
            total+=len(b);assert total<=entry['bytes'];output.write(b);digest.update(b)
            if total%(64<<20)==0:print(dict(ordinal=ordinal,bytes=total,expected=entry['bytes']),flush=True)
    assert total==entry['bytes'] and digest.hexdigest()==entry['lfs_sha256']
write(HERE/f'download_mini{ordinal:04d}.json',dict(passed=True,ordinal=ordinal,role=r['role'],payload=pin(path),new_bytes=total,
      values_decoded=0,wall_s=time.monotonic()-start))
resource_floor();print(read(HERE/f'download_mini{ordinal:04d}.json'),flush=True)
