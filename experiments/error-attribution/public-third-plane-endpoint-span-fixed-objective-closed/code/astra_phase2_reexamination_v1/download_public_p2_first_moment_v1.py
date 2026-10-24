"""One pinned public training file; bounded download, SHA-verified, no decoding."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
outdir = HERE/'public_p2_higher_moments_pilot_v1'
outdir.mkdir(exist_ok=True)
admission = json.loads((HERE/'public_p2_moments_metadata_v1.json').read_text())
frozen = json.loads((HERE/'public_p2_third_cell_mini0057_v1.json').read_text())
assert admission['passed'] and frozen['passed'] and frozen['target_values_decoded'] == 0
row = admission['selected'][0]
assert row['ordinal'] == 57 and row['role'] == 'train'
entry = row['moment_file']
OUT = outdir/'mlp_00057.npz'
assert not OUT.exists() and not OUT.with_suffix('.download.json').exists()
url = f"https://huggingface.co/datasets/{admission['repository']}/resolve/{admission['commit']}/{entry['path']}"
request = urllib.request.Request(url,headers={'User-Agent':'phase2-public-training-admission/1'})
start = time.time()
total = 0
digest = hashlib.sha256()
with urllib.request.urlopen(request,timeout=25) as response, OUT.open('xb') as destination:
    while True:
        data = response.read(8<<20)
        if not data:
            break
        total += len(data)
        assert total <= entry['bytes'], 'payload exceeded pinned size'
        destination.write(data)
        digest.update(data)
        if total%(64<<20) == 0:
            print(json.dumps(dict(bytes=total,expected=entry['bytes'],wall_s=time.time()-start)),flush=True)
passed = total == entry['bytes'] and digest.hexdigest() == entry['lfs_sha256']
record = dict(passed=passed,bytes=total,sha256=digest.hexdigest(),repository=admission['repository'],
              commit=admission['commit'],path=entry['path'],ordinal=57,role='train',
              target_values_decoded=0,wall_s=time.time()-start,utc_finished=datetime.now(timezone.utc).isoformat())
OUT.with_suffix('.download.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
assert passed, record
