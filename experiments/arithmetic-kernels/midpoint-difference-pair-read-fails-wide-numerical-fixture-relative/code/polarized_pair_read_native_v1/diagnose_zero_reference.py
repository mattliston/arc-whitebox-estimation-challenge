"""Inspect the failed exact-cancellation fixture, without changing any source."""
from common import *
text=(HERE/'unit.py').read_text();marker='for m,n,pp,eps in ((7,9'
assert text.count(marker)==1
ns={'__name__':'zero_reference_diagnostic'}
exec(compile(text[:text.index(marker)],str(HERE/'unit.py')+':definitions','exec'),ns)
for m,n,p,eps in ((7,9,[3],.7),(17,11,[2,5],.1),(9,5,[4,3],2**-12),(7,7,[3],0.)):
    left,right,lam,cohorts=ns['fixture'](m,n,p,eps)
np=ns['np'];flops=ns['flops'];fnp=ns['fnp'];mod=ns['mod']
assert np.array_equal(left[:,:3],left[:,3:]) and np.array_equal(right[:,:3],right[:,3:])
assert np.array_equal(lam[:3],-lam[3:])
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    actual=mod._polar_pair_product(fnp.asarray(left),fnp.asarray(right),fnp.asarray(lam),cohorts,0,6,mod._mm)
actual=np.asarray(actual);dense=ns['direct64'](left,right,lam);scalar=ns['scalar64'](left,right,lam,cohorts)
assert not np.any(actual) and not np.any(scalar) and np.any(dense)
write(HERE/'zero_reference_diagnostic.json',dict(passed=True,fixture_exact_pair_cancellation=True,
    actual_exactly_zero=True,independent_scalar_exactly_zero=True,dense64_maxabs=float(np.max(np.abs(dense))),
    dense64_norm=float(np.linalg.norm(dense)),failed_relative_error=relative(actual,dense),
    candidate=pin(HERE/'candidate.py'),original_unit=pin(HERE/'unit.py'),original_failure=pin(HERE/'unit.log'),
    candidate_unchanged=True,targets_read=False,full_native_calls=0))
print(read(HERE/'zero_reference_diagnostic.json'),flush=True)
