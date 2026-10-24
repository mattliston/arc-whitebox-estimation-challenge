"""Original paired8 calculation, gated by the pinned additive-control parity."""
import json
import math
from pathlib import Path
import sys
from rscov_additive_gate_v1 import control_reference, sha, verify_pins

manifest = verify_pins()
assert len(sys.argv) == 3
pin_path = Path(sys.argv[1]).resolve()
assert sha(pin_path) == sys.argv[2], 'Control-reference pin changed'
reference = json.loads(pin_path.read_text())
assert control_reference(reference['comparison_artifact']) == reference, 'Completed control changed during arm'
old = json.loads(Path(manifest['original_rscov_manifest']).read_text())
pairs = []
for row in old['rows']:
    claimed = Path(manifest['run']) / 'claimed' / (row['arm_id'] + '.w0.json')
    assert sha(claimed) == row['arm_request_sha256'], 'Arm request changed'
    result = json.loads((Path(manifest['run']) / 'results' / (row['arm_id'] + '.json')).read_text())
    base = json.loads(Path(row['base']).read_text())
    assert result['rank'] == base['rank'] and result['params'] == base['params']
    assert len(result['mse_layers']) == len(base['mse_layers']) == 16
    pairs.append((result, base))
logs = [math.log(r['mse'] / b['mse']) for r, b in pairs]
n = len(logs)
assert n == 8
mean = sum(logs) / n
se = math.sqrt(sum((value - mean) ** 2 for value in logs) / (n - 1) / n)
print(json.dumps(dict(n=n, mean_log_percent=100 * mean, se_percent=100 * se,
                      pooled_percent=100 * (sum(r['mse'] for r,b in pairs) / sum(b['mse'] for r,b in pairs) - 1),
                      lower=sum(value < 0 for value in logs),
                      per_network=[dict(seed=r['rank'], log_percent=100*value) for (r,b),value in zip(pairs,logs)],
                      pooled_layers_percent=[100 * (sum(r['mse_layers'][layer] for r,b in pairs) /
                                                     sum(b['mse_layers'][layer] for r,b in pairs) - 1) for layer in range(16)],
                      satisfies_torch_screen=100*mean <= old['threshold_mean_log_percent'],
                      threshold_mean_log_percent=old['threshold_mean_log_percent'],
                      control_reference=reference, control_reference_pin=str(pin_path),
                      control_reference_pin_sha256=sha(pin_path)), indent=2, allow_nan=False))
