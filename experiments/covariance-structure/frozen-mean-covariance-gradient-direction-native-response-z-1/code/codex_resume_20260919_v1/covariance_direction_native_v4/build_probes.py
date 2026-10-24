"""Four fixed literal probes from the already-sealed direction."""
from common import HERE,H,check,meta,read,sha,write
from compose import compose
check()
import numpy as np
p=read(HERE/'proposal/result.json');assert p['completed'] and p['optimizer_steps']==0 and p['calls']==4
assert sha(p['direction']['path'])==p['direction']['sha256']
d=np.load(p['direction']['path'],allow_pickle=False);assert d.shape==(14,65) and d.dtype==np.float32 and np.isfinite(d).all()
out=HERE/'probes';out.mkdir(exist_ok=False);rows=[]
for name,t in [('plus',H),('minus',-H),('halfplus',H/2),('halfminus',-H/2)]:
    Q=(d*np.float32(t)).astype(np.float32)
    source,proof=compose(Q.tolist());path=out/(name+'.py')
    path.write_text(source)
    rows.append(dict(name=name,t=t,source=meta(path),proof=proof))
write(out/'manifest.json',dict(completed=True,direction=p['direction'],proposal=meta(HERE/'proposal/result.json'),
    probes=rows,seal=meta(HERE/'sources_sealed.json'),literal_only_differences=True))
print('PROBES_SEALED',flush=True)
