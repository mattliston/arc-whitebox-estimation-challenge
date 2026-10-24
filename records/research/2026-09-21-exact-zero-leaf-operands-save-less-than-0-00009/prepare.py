import ast
from common import *
assert sha(SOURCE)=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
cp=D/'continuation_checkpoint_v87.json';assert sha(cp)=='171e9d35edb2543613c9520b77638725b8bee3ad3fd2e3df636576a411d94795'
prior=read(cp)
for k in ('study','official_bank','workers','verification','pending_release','pending_candidate','pending_archive','rejection'):verify(prior[k])
oldinputs=D/'current_product_identity_census_v1/inputs.json';inputs=read(oldinputs)
oldbound=D/'current_fixed_leaf_overhead_ceiling_v1/result.json';bound=read(oldbound)
assert not bound['both_pass_planning_gate']
tree=ast.parse(SOURCE.read_text());defs={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
for (name,line),names in SITES.items():
    node=defs[name];assert node.lineno<=line<=node.end_lineno
    expr=next(x for x in ast.walk(node) if isinstance(x,(ast.Call,ast.BinOp)) and x.lineno==line)
    if name=='_square_product':assert isinstance(expr,ast.BinOp) and isinstance(expr.op,ast.MatMult) and [expr.left.id,expr.right.id]==list(names)
    else:assert ast.unparse(expr.func)=='fnp.matmul' and [x.id for x in expr.args[:2]]==list(names)
external=[SOURCE,cp,oldinputs,oldbound,D/'run_guarded.py',D/'official_best_331684.json',
 D/'current_fixed_leaf_overhead_ceiling_v1/RESULT.md',D/'all_positive_leaf_catalog_v1/RESULT.md']
for row in inputs['rows']:
    for k in ('weights','prediction','control','old_result','attribution'):verify(row[k]);external.append(Path(row[k]['path']))
for name in ('_pointwise.py','_flops.py','_ndarray.py'):
    external.append(A/'openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope'/name)
for name in ('NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md','nonlinear_memory_capacity_current_v1/VERDICT_2026_09_07_PDT.md',
 'MATRIX_ERROR_OBSERVER_V1_CLOSED.md','CONDITIONAL_SKEW_ENERGY_PROBE_V1_CLOSED.md','POSITIVE_PATH_MATCHED_PILOT_V1_CLOSED.md',
 'CUMULANT_TILT_MEAN_RESPONSE_V1_CLOSED.md','current_covariance_gradient_gap_v1/RESULT_V1.md',
 'JOINT_DOUBLE_INSERTION_AND_INPUT_CHAOS_ADMISSION_2026_09_07_V1.md'):
    external.append(A/name)
for k in ('study','pending_release','pending_candidate','pending_archive','rejection'):external.append(Path(prior[k]['path']))
write(HERE/'inputs.json',dict(rows=inputs['rows'],previous_bound=pin(oldbound),source=pin(SOURCE),targets_not_read=True))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)],before_units_and_native_calls=True,no_targets=True))
print(dict(sealed=True,ranks=[35,100],source_sites=4),flush=True)
