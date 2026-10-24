"""Tiny extracted-function proof. Never imports or sets up an estimator."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

D = Path(__file__).resolve().parent
K = D.parent.parent
SOURCE = K / 'candidate_micro_copyall_ecut_nogov_astra_sci_v1.py'
PIN = '51aec0294ab23ced8280cfd104d6ebc97647585904721822c4b96be3049af6d5'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == PIN
tree = ast.parse(SOURCE.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gq']
assert [n.lineno for n in functions] == [13948, 24030]
assert ast.unparse(tree.body[-1]) == '_FAST_RESCUE_FRAC = {}'
settings = json.loads((D / 'settings_v1.json').read_text())
assert len(settings) == 6
assert all(set(s) == {'_GSK', '_GK4', '_GQ33_CONST', '_GSK_CONST'}
           and s['_GQ33_CONST'] == s['_GSK_CONST'] == {} for s in settings)


class CountingNP:
    float64 = np.float64

    def __init__(self):
        self.binding_calls = 0
        self.binding_elements = 0

    def ones(self, *a, **k):
        x = np.ones(*a, **k)
        self.binding_calls += 1
        self.binding_elements += x.size
        return x

    def asarray(self, *a, **k):
        x = np.asarray(*a, **k)
        self.binding_calls += 1
        self.binding_elements += x.size
        return x

    def __getattr__(self, name):
        return getattr(np, name)


fnp = CountingNP()
env = dict(fnp=fnp, _E2Hold=lambda: (lambda x: x), _GSK=.9, _GK4=1.04,
           _GBK=0.0, _GSK_CONST={}, _GQ33_CONST={}, _GQ33_STATS={'calls': 0})
exec(compile(ast.Module(body=[functions[0]], type_ignores=[]), '<legacy-gq-only>', 'exec'), env)
legacy = env['_gq']
exec(compile(ast.Module(body=[functions[1]], type_ignores=[]), '<live-gq33-only>', 'exec'), env)
live = env['_gq']
g = np.linspace(.13, .71, 8)
d = np.linspace(-.14, .11, 8)
bpre = np.linspace(.20, .36, 8)
aux = dict(sigma=np.linspace(.7, 1.4, 8), alpha=np.linspace(-1.4, 1.7, 8),
           pdf=np.linspace(.09, .31, 8), pre_k3=np.linspace(-.32, .41, 8),
           pre_k4=np.linspace(.19, .53, 8))
args = (g, d, aux, bpre)

# Same n + no reset demonstrably retains stale coefficients. This witnesses
# the actual bug that every setting below must prevent.
baseline = live(*args).copy()
legacy_baseline = legacy(*args).copy()
env['_GSK'] = .8
assert np.array_equal(live(*args), baseline)
assert np.array_equal(legacy(*args), legacy_baseline)

outputs = []
checks = []
for index, setting in enumerate(settings + [settings[0]]):
    for key, value in setting.items():
        env[key] = copy.deepcopy(value)  # Identical setting semantics to v9.
    assert env['_GSK_CONST'] == env['_GQ33_CONST'] == {}
    before_calls, before_elements = fnp.binding_calls, fnp.binding_elements
    output = live(*args).copy()
    coefficient = env['_GQ33_CONST']['coefficient'].copy()
    gsk, gk4 = setting['_GSK'], setting['_GK4']
    expected = np.array([1., gsk/6, -gsk/6, -gk4/24, 3*gk4/24,
                         0., -1/72, 10/72, -15/72])
    assert np.array_equal(coefficient, expected)
    assert fnp.binding_calls - before_calls == 3
    assert fnp.binding_elements - before_elements == 8 + 14*9 + 9
    before_repeat = fnp.binding_calls
    assert np.array_equal(live(*args), output)
    assert fnp.binding_calls == before_repeat
    # The old cache is empty until its own direct call, and then uses the
    # same active setting. No accidental legacy population can pin defaults.
    assert env['_GSK_CONST'] == {}
    old_output = legacy(*args)
    assert np.array_equal(env['_GSK_CONST']['coefficient'], expected[:6])
    if index in (1, 2, 3, 4, 5):
        assert not np.array_equal(output, baseline)
        assert not np.array_equal(old_output, legacy_baseline)
    if index in (0, 6):
        assert np.array_equal(output, baseline)
        assert np.array_equal(old_output, legacy_baseline)
    outputs.append(output)
    checks.append(dict(index=index, gsk=gsk, gk4=gk4,
                       coefficients=coefficient.tolist(),
                       max_abs_delta_from_default=float(np.max(abs(output-baseline))),
                       binding_calls=3, binding_elements=143))

# The gate formula is affine in these two coefficients when its inputs are
# fixed. This is a dispatch proof, not a claim that whole-network responses
# or score gains add linearly.
assert np.allclose(outputs[5], outputs[1] + outputs[3] - outputs[0],
                   rtol=0, atol=2e-16)
assert settings == json.loads((D / 'settings_v1.json').read_text())
receipt = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
               status='PASS_TINY_EXTRACTED_GQ_CACHE_SEQUENCE', source=str(SOURCE),
               source_sha256=PIN, live_gq_line=functions[-1].lineno,
               stale_without_reset_witness=True, both_caches_reset_each_setting=True,
               same_width_reuse_proved=True, default_restoration_proved=True,
               final_no_governor_assignment=True, setting_checks=checks,
               limitations=['NumPy tiny arrays, no complete estimator import/setup/prediction.',
                            'Does not measure native score, bill or runtime.',
                            'Fresh v9 resets add equal tiny constant bindings to each screen setting; bank [{}] remains its actual warm path.'])
(D / 'CACHE_AST_PROOF_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'setting_checks'}, indent=2))
