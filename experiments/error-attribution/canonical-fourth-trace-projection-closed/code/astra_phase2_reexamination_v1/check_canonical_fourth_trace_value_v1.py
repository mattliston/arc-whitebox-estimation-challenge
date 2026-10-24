"""The one canonical alternative, exact same frozen targets and value gate."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
assert json.loads((HERE/'canonical_fourth_trace_certificate_v1.json').read_text())['passed']
original = HERE/'check_fourth_trace_oracle_value_v1.py'
source = original.read_text()
for before, after in (
    ("OUT = HERE/'fourth_trace_oracle_value_v1.json'", "OUT = HERE/'canonical_fourth_trace_value_v1.json'"),
    ('from thin_fourth_wide_helpers_v1 import radial_query', 'from thin_fourth_wide_helpers_v1 import radial_query\nfrom canonical_fourth_trace_v1 import canonical_trace_core'),
    ('metric = w13.T@C@w13', 'metric = w13.T@w13'),
    ('A = inverse_core(C, residual)', 'A = canonical_trace_core(residual)'),
    ('core_trace(A, C)', 'core_trace(A, np.eye(1024))'),
    ('One fixed full-rank covariance-metric oracle trace core.', 'One fixed canonical Euclidean harmonic oracle trace projection, not covariance-metric trace matching.'),
):
    assert source.count(before) == 1, before
    source = source.replace(before, after)
before = '    consumers = {}'
after = '''    previous = dict(np.load(HERE/'fourth_trace_oracle_value_v1.npz'))
    target_parity = {key: bool(np.array_equal(value, previous[key])) for key, value in saved.items() if key.startswith('target_')}
    assert all(target_parity.values()), target_parity
    consumers = {}'''
assert source.count(before) == 1
source = source.replace(before, after)
before = 'positive_signal_qualified=all(c[\'baseline_replica_cross_energy\'] > 0 for c in consumers.values()),'
assert source.count(before) == 1
source = source.replace(before, before+'\n                  previous_target_byte_parity=target_parity,')
exec(compile(source, str(original), 'exec'), dict(__file__=str(Path(__file__).resolve()), __name__='__main__'))
