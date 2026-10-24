"""Exactly24 new honest rows, three bounded inherited-nice CPU workers."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_pilot_rows_v1.json';assert not OUT.exists()
assert json.loads((HERE/'positive_path_recursive_r35_l12_v1.json').read_text())['recursive_source_gate']
assert json.loads((HERE/'positive_path_replication_r100_v1.json').read_text())['recursive_source_gate']
ROWS=HERE/'rows_positive_path_pilot_v1';PY='/Users/matt/arc-white-box-estimation/.venv/bin/python'
names=('positive_path_resummation_v1.py','positive_path_birth_reference_v1.py','generate_positive_path_rows_v1.py','generate_radial_rows.py','radial_features.py')
hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names};start=time.time()
def run(rank):
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    p=subprocess.run([PY,'-u',str(HERE/'generate_positive_path_rows_v1.py'),'--rank',str(rank),'--rows',str(ROWS),'--cov32'],cwd=HERE,capture_output=True,text=True)
    if p.returncode:raise RuntimeError((rank,p.stdout,p.stderr))
    return rank,json.loads((ROWS/f'row_{rank:04d}/result.json').read_text())
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(run,r) for r in range(35,59)]
    for i,f in enumerate(concurrent.futures.as_completed(futures),1):
        rank,row=f.result();records.append(row)
        print(json.dumps(dict(rows=i,rank=rank,one_step_final=row['one_step_mse'][-1],wall_s=time.time()-start)),flush=True)
profile=[sum(row['one_step_mse'][l] for row in records)/len(records) for l in range(16)]
old=[json.loads((HERE/'rows_compact_feed_fullpair_v1'/f'row_{r:04d}/result.json').read_text()) for r in range(35,59)]
old_profile=[sum(row['one_step_mse'][l] for row in old)/len(old) for l in range(16)]
assert 1e-10<profile[-1]<2*profile[10],('unhealthy honest rows',profile)
record=dict(ranks=list(range(35,59)),source_hashes=hashes,profile=profile,parent_profile=old_profile,
            profile_ratio=[x/y for x,y in zip(profile,old_profile)],honest_profile_gate=True,wall_s=time.time()-start,
            qualification='Same pre-correction feature capture and later consistent mean-feedback update.24 pilot rows only, no model fit or score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
