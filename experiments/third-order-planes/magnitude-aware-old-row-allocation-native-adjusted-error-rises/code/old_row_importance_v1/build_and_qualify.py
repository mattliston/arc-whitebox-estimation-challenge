"""Fixed helper qualification and write-once S31-derived candidate build."""
from pathlib import Path
import ast
import hashlib
import json
import math
import numpy as np

D = Path(__file__).resolve().parent
H = D.parent
BANK = H / 'candidate_old_both_tail_trace_bounded_v6.py'
EXPECTED = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
source = BANK.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED
fragment = (D / 'fragment.py').read_text()
env = dict(fnp=np, _OTT_Z=2.5, _OTT_NEG_Z=2.5, _E2_LAYER={})
exec(compile(fragment, str(D / 'fragment.py'), 'exec'), env)


def direct(u, a, v, live):
    m = len(a)
    base = [float(x + 2.5) if x < -2.5 and not live[i] else float(2.5 - x)
            for i, x in enumerate(a)]
    base_k = min(m, 64 * ((sum(x >= 0 for x in base) + 63) // 64))
    must = [bool(x < 0 and live[i]) for i, x in enumerate(a)]
    k = min(m, max(64, 64 * ((max(0, base_k - 128) + 63) // 64),
                   64 * ((sum(must) + 63) // 64)))
    lever = [sum(float(x) * float(x) for x in row) / max(float(v[i]), 1e-10) for i, row in enumerate(u)]
    average = math.fsum(lever) / m
    priority = [-.5 * float(x) * float(x) + math.log(max(lever[i] / (average + 1e-30), 1e-30)) for i, x in enumerate(a)]
    top = max(priority) + 1
    order = sorted(range(m), key=lambda i: -(top if must[i] else priority[i]))
    return k, order[:k], order[k:], priority


rng = np.random.Generator(np.random.PCG64DXSM(9262701))
cases = []
for m in [7, 63, 64, 129, 256, 320]:
    u = rng.normal(size=(m, 8)).astype(np.float32)
    a = rng.normal(size=m) * 2
    v = np.exp(rng.normal(size=m))
    live = a > -1.8
    cases.append((u, a, v, live, None))
cases.append((np.zeros((192, 8), np.float32), np.zeros(192), np.zeros(192), np.ones(192, bool), None))
cases.append((np.ones((320, 8), np.float32), np.zeros(320), np.ones(320), np.ones(320, bool), None))
u, a, v, live, _ = cases[-3]
full_v = np.zeros(2 * len(v)); full_v[::2] = v
cases.append((u, a, full_v, live, np.arange(0, 2 * len(a), 2)))
for u, a, v, live, rows in cases:
    before = [x.tobytes() for x in [u, a, v, live]]
    env['_E2_LAYER']['vc'] = v
    ctx = dict(a=a, live=live, rows=rows)
    k, kept, tail = env['_ori_select'](u, ctx)
    vv = v if rows is None else v[rows]
    expected = direct(u, a, vv, live)
    assert (k, kept.tolist(), tail.tolist()) == expected[:3]
    assert all(i in set(kept) for i in range(len(a)) if a[i] < 0 and live[i])
    assert sorted([*kept, *tail]) == list(range(len(a)))
    assert before == [x.tobytes() for x in [u, a, v, live]]
# Invariance of the unfloored mathematical score on well-conditioned inputs.
u = rng.normal(size=(256, 8)); a = rng.normal(size=256); v = np.exp(rng.normal(size=256))
q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
live = a > -1.8
original = direct(u, a, v, live)
rotated = direct(u @ q, a, v, live)
scale = np.exp(rng.normal(size=256))
scaled = direct(u * scale[:, None], a, v * scale * scale, live)
assert np.allclose(original[3], rotated[3], rtol=0, atol=1e-13)
assert np.allclose(original[3], scaled[3], rtol=0, atol=1e-13)
assert original[:3] == rotated[:3] == scaled[:3]

tree = ast.parse(source)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_slr_reconstruct'][-1]
old = ast.get_source_segment(source, node)
start = old.index("    a = ctx['a']")
end = old.index("    columns = int(state['columns'])")
new = old[:start] + '''    m = int(U.shape[0])
    kept, rows, tail = _ori_select(U, ctx)
    if kept == m:
        return _OTT_ORIGINAL_RECON(U, state)
''' + old[end:]
# Append a new binding; leave every original byte intact.
candidate = source + '\n\n' + fragment + '\n\n' + new + '\n'
dest = D / 'candidate.py'
compile(candidate, str(dest), 'exec')
with dest.open('x') as f:
    f.write(candidate)
evaluation = (H / 'native_eval.py').read_text()
evaluation = evaluation.replace('HERE=Path(__file__).resolve().parent', 'HERE=Path(' + repr(str(H)) + ')')
old_telemetry = "p6=dict(getattr(mod,'_P6_STATS',{})))"
assert evaluation.count(old_telemetry) == 1
evaluation = evaluation.replace(old_telemetry, "p6=dict(getattr(mod,'_P6_STATS',{})),old_row_importance=list(getattr(mod,'_ORI_STATS',[])))")
with (D / 'evaluate.py').open('x') as f:
    f.write(evaluation)
qualification = dict(passed=True, direct_cases=len(cases), mandatory_membership=True,
    stable_ties=True, partition_complete=True, finite_zero_magnitude=True, inputs_preserved=True,
    mathematical_orthogonal_invariance=True, mathematical_positive_row_scale_invariance=True,
    candidate_first_original_bytes_preserved=candidate.startswith(source))
with (D / 'qualification.json').open('x') as f:
    json.dump(qualification, f, indent=2)
files = [BANK, D / 'PROTOCOL.md', D / 'fragment.py', Path(__file__), dest, D / 'evaluate.py']
with (D / 'pre_endpoint_seal.json').open('x') as f:
    json.dump(dict(files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                   endpoint_scored=False, qualification=qualification), f, indent=2)
print(json.dumps(dict(qualification=qualification, candidate=str(dest),
                     candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest())), flush=True)
