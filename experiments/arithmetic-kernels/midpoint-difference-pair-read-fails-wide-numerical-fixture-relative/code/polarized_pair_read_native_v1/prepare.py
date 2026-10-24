import ast
from common import *
cp=D/'continuation_checkpoint_v88.json';assert sha(cp)=='e6ad69beacd3113e210fa4d915b3d2dc2a086483292423e2bdbcde3080e1ad9e'
prior=read(cp)
for k in ('study','workers','verification','official_bank','pending_release','pending_candidate','pending_archive','rejection'):verify(prior[k])
assert sha(PARENT)=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
body=PARENT.read_bytes();suffix=b'\n\n'+(HERE/'production.py').read_bytes();candidate=body+suffix
assert candidate[:-len(suffix)]==body;ast.parse(candidate.decode())
with (HERE/'candidate.py').open('xb') as f:f.write(candidate)
meta_path=D/'current_edge_nonlinear_native_v1/inputs.json';meta=read(meta_path)
external=[PARENT,cp,meta_path,D/'run_guarded.py',D/'official_best_331684.json']
def collect(x):
    if isinstance(x,dict):
        if {'path','sha256','bytes'}<=x.keys():verify(x);external.append(Path(x['path']))
        for v in x.values():collect(v)
    elif isinstance(x,list):
        for v in x:collect(v)
collect(meta)
for k in ('pending_release','pending_candidate','pending_archive','rejection','study'):external.append(Path(prior[k]['path']))
for name in ('JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md','PAIR_COALESCENCE_INDEPENDENT_REVIEW_V1.md','RAW_LEADS_AFTER_FIXED_V29_OBSERVER_20260912.md'):
    external.append(A/name)
for name in ('pair384_metric_v1/NEGATIVE_RESULT.md','orthogonal_cubic_merge_information_v1/SOURCE_REVIEW.md','shared_sketch_roundoff_diagnostic_v1/RESULT.md','fp64_sketch_endpoint_v1/RESULT.md'):
    external.append(D/name)
write(HERE/'inputs.json',meta)
write(HERE/'build.json',dict(completed=True,candidate=pin(HERE/'candidate.py'),parent=pin(PARENT),
    exact_parent_prefix_inverse=True,append_only_bytes=len(suffix),no_parameter_changes=True,targets_read=False))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)],
    before_units_and_native_calls=True,before_targets=True))
print(dict(prepared=True,candidate=pin(HERE/'candidate.py')),flush=True)
