from pathlib import Path
import ast, hashlib, json
import numpy as np

D = Path(__file__).resolve().parent
A = D.parent.parent
R = A.parent
parent = A / 'setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
expected = '95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
body = parent.read_bytes()
assert hashlib.sha256(body).hexdigest() == expected
fragment = (D / 'fragment_v1.py').read_bytes()
candidate = D / 'candidate_95c_re22cross_old0_HELD_v1.py'
source = body + b'\n' + fragment
ast.parse(source)
candidate.open('xb').write(source)

rng = np.random.default_rng(93712)
cases = []
for m, p, r in [(8, 7, 2), (7, 9, 2), (11, 6, 1), (9, 5, 0)]:
    P = rng.normal(size=(m, p))
    Q = np.linalg.qr(rng.normal(size=(m, max(r, 1))))[0][:, :r]
    projected = Q @ (Q.T @ P)
    residual = P - projected
    rows = np.arange(m)
    cols = np.arange(p) + 2
    _, ir, ic = np.intersect1d(rows, cols, return_indices=True)
    residual[ir, ic] = 0
    denom = np.exp(rng.normal(size=(m, 1))) * np.exp(rng.normal(size=(1, p)))
    N, PN = residual / denom, P / denom
    formula = np.einsum('ij,ij->i', N, 2 * PN - N)
    P0, A0 = PN.copy(), projected / denom
    P0[ir, ic] = 0
    A0[ir, ic] = 0
    direct = np.sum(P0 * P0 - A0 * A0, axis=1)
    error = float(np.max(np.abs(formula - direct)))
    assert error < 1e-10, error
    cases.append(dict(shape=[m, p], rank=r, max_abs_error=error))

rowsfile = R / 'astra_oct18_bill_v1/post935_compute_v1/w3_early_restore_v1/native2_v1/rows.jsonl'
row = json.loads(rowsfile.read_text().splitlines()[0])
reads = row['post']['rtt_state']['reads'] if 'post' in row else row['rtt_state']['reads']
shapes = [x for x in reads if x['changed'] and not x['old_rank']]
# Original energy dot unchanged; extra normalize+scale+subtract=3 elements.
extra_flops = sum(3 * x['m'] * x['c15'] for x in shapes)
receipt = dict(status='SOURCE_AND_NUMPY_IDENTITY_PASS_NO_NATIVE',
    parent=str(parent), parent_sha256=expected, source=str(candidate),
    source_sha256=hashlib.sha256(source).hexdigest(),
    fragment_sha256=hashlib.sha256(fragment).hexdigest(),
    selection='all and only RTT reads whose state columns is zero',
    semantic='Prefix row-energy difference, not an exact tail-energy oracle; unchanged inverse-variance transport ratio.',
    diagonal='Residual overlap zeroed before cross term; physical K3/K21 untouched.',
    old_component='Original _re22_energy retained for every old>0 read.',
    additional_public_calls_per_affected_read=3,
    static_additional_flops_rank35=extra_flops,
    affected_rank35=[x['layer'] for x in shapes], tiny_cases=cases,
    prior_closed='Exact feature24 restore: eight native networks raw+.16264%, bill+.18434%, adjusted+.34699%; do not reopen.',
    no_new_einsum_signature=True, models=0)
(D / 'BUILD_v1.json').open('x').write(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
