"""Stdlib-only control and paired8 comparator; invoked in worker's guarded process."""
import hashlib
import json
import math
from pathlib import Path
import sys

H = Path(__file__).resolve().parent
manifest = json.loads((H / 'cfcov_manifest_v1.json').read_text())
for row in manifest['rows']:
    for field, key in (('base', 'base_sha256'), ('base_request', 'base_request_sha256')):
        assert hashlib.sha256(Path(row[field]).read_bytes()).hexdigest() == row[key]
control = json.loads(Path(manifest['control_result']).read_text())
base0 = json.loads(Path(manifest['rows'][0]['base']).read_text())
assert control['params'] == base0['params']
assert control['mse'] == base0['mse'] and control['mse_layers'] == base0['mse_layers'], 'Private-control mismatch'
print('Parity control: all 16 layer MSEs identical to true S51+gk11 t134 learnedON')
if len(sys.argv) > 1 and sys.argv[1] == 'control':
    raise SystemExit(0)
pairs = []
for row in manifest['rows']:
    result = json.loads((H / 'run_cfcov_arm_v1/results' / (row['arm_id'] + '.json')).read_text())
    base = json.loads(Path(row['base']).read_text())
    assert result['rank'] == base['rank'] and result['params'] == base['params']
    assert len(result['mse_layers']) == len(base['mse_layers']) == 16
    pairs.append((result, base))
logs = [math.log(r['mse'] / b['mse']) for r, b in pairs]
n = len(logs)
mean = sum(logs) / n
se = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1) / n)
print(json.dumps(dict(n=n, mean_log_percent=100 * mean, se_percent=100 * se,
                      pooled_percent=100 * (sum(r['mse'] for r,b in pairs) / sum(b['mse'] for r,b in pairs) - 1),
                      lower=sum(x < 0 for x in logs),
                      per_network=[dict(seed=r['rank'], log_percent=100*x) for (r,b),x in zip(pairs,logs)],
                      pooled_layers_percent=[100 * (sum(r['mse_layers'][l] for r,b in pairs) /
                                                     sum(b['mse_layers'][l] for r,b in pairs) - 1) for l in range(16)],
                      satisfies_torch_screen=100*mean <= manifest['threshold_mean_log_percent']), indent=2))
