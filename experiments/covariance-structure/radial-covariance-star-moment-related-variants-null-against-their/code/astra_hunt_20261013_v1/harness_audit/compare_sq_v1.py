"""Read-only, stdlib parity and paired8 audit. Run 'control' before launching the arm."""
import hashlib
import json
import math
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
manifest = json.loads((H / 'sq_base_manifest_v1.json').read_text())
for row in manifest:
    p = Path(row['base'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['base_sha256']

control = json.loads((H / 'run_sq_control_v1/results/sc_astractl_r7100584_ctl.json').read_text())
base0 = json.loads(Path(manifest[0]['base']).read_text())
assert control['params'] == base0['params']
assert control['mse'] == base0['mse'] and control['mse_layers'] == base0['mse_layers'], 'Private-control mismatch'
print('Parity control: all 16 layer MSEs identical to existing S51+gk control')
if len(sys.argv) > 1 and sys.argv[1] == 'control':
    raise SystemExit(0)

pairs = []
for row in manifest:
    r = json.loads((H / 'run_sq_arm_v1/results' / (row['arm_id'] + '.json')).read_text())
    b = json.loads(Path(row['base']).read_text())
    assert r['rank'] == b['rank'] and r['params'] == b['params']
    assert len(r['mse_layers']) == len(b['mse_layers']) == 16
    pairs.append((r, b))
logs = [math.log(r['mse'] / b['mse']) for r, b in pairs]
n = len(logs)
m = sum(logs) / n
se = math.sqrt(sum((x - m) ** 2 for x in logs) / (n - 1) / n)
print(json.dumps(dict(n=n, mean_log_percent=100*m, se_percent=100*se,
                      pooled_percent=100*(sum(r['mse'] for r,b in pairs)/sum(b['mse'] for r,b in pairs)-1),
                      lower=sum(x < 0 for x in logs),
                      per_network=[dict(seed=r['rank'], log_percent=100*x) for (r,b),x in zip(pairs,logs)],
                      satisfies_torch_screen=100*m <= -0.15), indent=2))
