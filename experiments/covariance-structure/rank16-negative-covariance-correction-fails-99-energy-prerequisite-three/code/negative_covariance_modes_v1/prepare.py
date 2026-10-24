import ast

from common import *

assert sha(PARENT) == PARENT_SHA
for p in HERE.glob('*.py'):
    ast.parse(p.read_text())
recs = captures();assert [r['rank'] for r in recs] == list(RANKS)
for r in recs:
    assert r['all16_and_bill_exact'] and r['cache_retired']
inputs = [PARENT, D/'run_guarded.py', CAPTURE_RESULT,
          D/'full_lower_covariance_innovation_supervision_v1/captures_sealed.json',
          D/'covariance_symmetry_v1/NEGATIVE_RESULT.md',
          P2/'FULL_BUDGET_COST_REOPEN_AUDIT.md',
          D/'lazy_cost_census_v1/COST_REPORT.md',
          D/'lazy_mixed_fmm_reprice_v3/RESULT.md',
          A/'EXACT_HEAVY_MIXED_BIRTH_REPRESENTATION_SOURCE_AUDIT_V1.md',
          A/'RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md',
          A/'PUBLIC_P2_THIRD_ENDPOINT_SPAN_V1_CLOSED.md']
for r in recs:
    inputs.extend([Path(r['capture']['path']), Path(r['weights']['path']), truth(r['rank'])])
for r in recs:
    assert sha(r['capture']['path']) == r['capture']['sha256']
write(HERE/'sources_sealed.json', dict(pins=[pin(p) for p in sorted(HERE.glob('*.py'))]
      +[pin(HERE/'PROTOCOL.md')]+[pin(p) for p in inputs], targets_decoded=False,
      purpose='Fixed current-parent spectral geometry and covariance/response information test.'))
print('Sources and existing inputs sealed before numerical work.')
