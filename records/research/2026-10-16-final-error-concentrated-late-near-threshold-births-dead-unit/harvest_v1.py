"""Harvest per-seed layer MSEs from every native certificate under lead/runs (read-only)."""
import json,os,re
from pathlib import Path
O=Path(__file__).resolve().parents[3];out={}
for root,dirs,files in os.walk(O/'lead/runs'):
    for f in files:
        m=re.match(r'native_(\d{7})_(\w+?)(_v\d+)?\.json$',f)
        if not m or 'guard' in f:continue
        try:d=json.loads((Path(root)/f).read_text())
        except Exception:continue
        if not isinstance(d,dict) or 'rows' not in d or not d['rows']:continue
        r=d['rows'][0]
        if 'mse_layers' not in r or len(r['mse_layers'])!=16:continue
        out.setdefault(int(m.group(1)),[]).append(dict(sha=d.get('sha256','')[:8],mtime=os.path.getmtime(Path(root)/f),mse=r['mse_layers'],flops=r.get('flops'),path=str(Path(root)/f)))
json.dump({str(k):v for k,v in out.items()},open(Path(__file__).parent/'HARVEST_v1.json','x'))
print(len(out),'seeds;',sum(len(v) for v in out.values()),'certs')
from collections import Counter
print(sorted(Counter(len(v) for v in out.values()).items())[:10])
print('range',min(out),max(out)); print(sorted(out)[:40])
