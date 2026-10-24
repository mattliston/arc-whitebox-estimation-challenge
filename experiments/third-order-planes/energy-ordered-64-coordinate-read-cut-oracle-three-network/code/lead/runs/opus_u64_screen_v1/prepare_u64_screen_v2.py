"""v2: runner reads the opus protocol file (v1 runner pointed at the astra protocol name). No model calls."""
from pathlib import Path
import hashlib,json,datetime
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    p=D/name
    with p.open('x') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,indent=2);f.write('\n')
    return p
r=(D/'finite_u64_screen_v1.py').read_text();a="p=D/'PROTOCOL_REGISTERED_astra_v1.json'";assert r.count(a)==1
write('finite_u64_screen_v2.py',r.replace(a,"p=D/'PROTOCOL_REGISTERED_opus_v2.json'"))
c=json.loads((D/'PROTOCOL_REGISTERED_opus_v1.json').read_text())
c['source_pins'].pop(str(D/'finite_u64_screen_v1.py'))
for p in (D/'finite_u64_screen_v2.py',Path(__file__).resolve(),D/'PROTOCOL_REGISTERED_opus_v1.json'):c['source_pins'][str(p)]=sha(p)
c['registered_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for path,digest in c['source_pins'].items():assert sha(path)==digest,path
write('PROTOCOL_REGISTERED_opus_v2.json',c)
compile((D/'finite_u64_screen_v2.py').read_text(),'r','exec')
print(sha(D/'PROTOCOL_REGISTERED_opus_v2.json'),sha(D/'finite_u64_screen_v2.py')[:8])
