"""Source-only GN correlation-order-two restoration on the fixed RRPI parent."""
from pathlib import Path
import hashlib, json

D = Path(__file__).resolve().parent
P = D.parent/'hill_climb_pair_rr_inverse_v1/candidate.py'
EXPECTED = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
raw = P.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
original = raw.decode()
edits = [
    ("    # E1: the GN plane read uses the same retained r=0,1 terms.",
     "    # GN-R2: restore only this correction's r=2 read; the main pair map stays E1."),
    ("             left1=FT[3:5, 0, :, None], right1=FT[2, 0, None, :],\n             rhoT=PB[0].T,",
     "             left1=FT[3:5, 0, :, None], right1=FT[2, 0, None, :],\n"
     "             left2=FT[4:6, 0, :, None], right2=FT[3, 0, None, :],\n"
     "             rho2=PB[1], rhoT=PB[0].T,"),
    ("    hold(fnp.add(g['B'], g['H'], out=g['B']))\n    hold(fnp.multiply(g['D'], g['B'], out=g['D']))",
     "    hold(fnp.add(g['B'], g['H'], out=g['B']))\n"
     "    hold(fnp.multiply(g['left2'], g['right2'], out=g['H']))\n"
     "    hold(fnp.multiply(g['rho2'], g['H'], out=g['H']))\n"
     "    hold(fnp.add(g['B'], g['H'], out=g['B']))\n"
     "    hold(fnp.multiply(g['D'], g['B'], out=g['D']))"),
]
candidate = original
for before, after in edits:
    assert candidate.count(before) == 1 and after not in candidate
    candidate = candidate.replace(before, after, 1)
inverse = candidate
for before, after in reversed(edits):
    assert inverse.count(after) == 1
    inverse = inverse.replace(after, before, 1)
assert inverse == original
compile(candidate, str(D/'candidate.py'), 'exec')
out = candidate.encode()
with (D/'candidate.py').open('xb') as f: f.write(out)
record = dict(parent=str(P), parent_sha256=EXPECTED, candidate_sha256=hashlib.sha256(out).hexdigest(),
              candidate_bytes=len(out), inverse_whole_source_exact=True, compiled=True,
              replacements=[dict(before=a, after=b, count=1) for a,b in edits],
              current_layers=list(range(2,15)), no_fit_or_sweep=True,
              scope='Restore existing historic GN r2 term only; main pair map stays E1; fallback already has r2.',
              formula='B[0]+=rho^2/2*A4_i*A3_j; B[1]+=rho^2/2*A5_i*A3_j; existing D and divisors 2/6 unchanged.',
              source_cost=dict(extra_flops_per_layer=6*1024**2, layers=13, extra_flops_per_network=6*1024**2*13),
              history='Original GN3923-3944 had r2; precision_g3_plane_e1_v1 removed main-plane and GN r2 together. No isolated current GN-only restoration found in bounded review.',
              numerical_execution=False)
with (D/'source_delta.json').open('x') as f: json.dump(record,f,indent=2); f.write('\n')
print(json.dumps({k:record[k] for k in ('candidate_sha256','candidate_bytes','inverse_whole_source_exact')}))
