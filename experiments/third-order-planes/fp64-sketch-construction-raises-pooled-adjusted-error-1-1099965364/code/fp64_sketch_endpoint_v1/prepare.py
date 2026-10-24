from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

HERE=Path(__file__).resolve().parent;D=HERE.parent;RESPONSE=D/'fp64_sketch_response_v1';OLD=D/'covariance_head_native_response_v1'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
rows=[]
for rank in [36,38]:
    old=next(x for x in read(OLD/'inputs.json')['qualification_parent'] if x['rank']==rank)
    receipt=RESPONSE/f'r{rank:04d}_hbase_base/result.json';new=read(receipt);c=new['records'][0]
    assert c['rank']==rank and c['weights_sha256']==old['weights_sha256']
    for p in [old['output'],old['truth'],old['source'],old['receipt']]:assert pin(p['path'])==p
    rows.append(dict(rank=rank,parent=old,candidate=c,candidate_receipt=pin(receipt),
                     candidate_output=pin(c['output']),candidate_source=pin(RESPONSE/'candidate.py')))
write(HERE/'inputs.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
     labels_decoded=False,response_result_known=False))
paths=[HERE/'PROTOCOL.md',HERE/'prepare.py',HERE/'score.py',HERE/'inputs.json',D/'run_guarded.py',RESPONSE/'sources_sealed.json']
paths += [x[k]['path'] for row in rows for x,k in [(row,'candidate_receipt'),(row,'candidate_output'),(row,'candidate_source'),
          (row['parent'],'source'),(row['parent'],'output'),(row['parent'],'truth'),(row['parent'],'receipt')]]
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted({str(Path(p).resolve()) for p in paths})],labels_decoded=False))
print(dict(prepared=True,labels_decoded=False),flush=True)
