"""Create independently versioned source adapters without changing old studies."""
import ast
from common import *

def put(name,text):
    with (HERE/name).open('x') as f:
        f.write(text)

old=D/'current_p_batch_process_endpoint_v1'
for name in ('guard.py','retirement.py'):
    put(name,(old/name).read_text())
put('head.py',(MODEL_DIR/'head.py').read_text())

builder=(MODEL_DIR/'build.py').read_text()
start=builder.index("f=read(HERE/'model_frozen.json')")
builder='''"""Explicit new direct-objective study; the original proxy failure is preserved."""
import ast,base64,resource,zlib
from common import *
import numpy as np
verify()
assert not read(MODEL_DIR/'information_result.json')['passed']
assert read(MODEL_DIR/'replay_result.json')['passed']
'''+builder[start:]
builder=builder.replace("f=read(HERE/'model_frozen.json')","f=read(MODEL_DIR/'model_frozen.json')",1)
builder=builder.replace("information_result=pin(HERE/'information_result.json'),","information_result=pin(MODEL_DIR/'information_result.json'), original_proxy_gate_passed=False,\n      research_decision='Direct actual-objective test of the unchanged frozen model after a failed proxy gate.',\n      null_candidate=pin(HERE/'null_candidate.py'),",1)
marker="write(HERE/'build.json',"
assert builder.count(marker)==1
builder=builder.replace(marker,"with (HERE/'null_candidate.py').open('x') as stream:stream.write(candidate+'\\n_JOINT_ENABLED=False\\n')\n"+marker,1)
ast.parse(builder)
put('build.py',builder)

native=(old/'native.py').read_text()
begin=native.index("    ap.add_argument('stage'")
end=native.index('    source = admitted[',begin)
native=native[:begin]+'''    ap.add_argument('stage', choices=['development'])
    ap.add_argument('role', choices=['null','parent','trained'])
    args = ap.parse_args()
    stage_ready(args.stage)
    admitted = admission()
    meta = read(HERE/'inputs.json')
    ranks, name = [35,100,35], 'native_'+args.role
'''+native[end:]
ast.parse(native)
put('native.py',native)
def budget_scopes(source):
    return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source))
            if isinstance(n,ast.With) and any('BudgetContext' in ast.unparse(x.context_expr) for x in n.items)]
assert budget_scopes(native)==budget_scopes((old/'native.py').read_text())

inputs=read(old/'inputs.json')
fit=read(D/'current_full_lower_p_fit_v1/inputs.json')
targets=fit.get('targets',fit.get('train_targets'))
assert targets is not None, list(fit)
meta=dict(ranks=[35,100,35],weights={str(r):inputs['weights'][str(r)] for r in (35,100)},
          targets={str(r):targets[str(r)] for r in (35,100)},
          official_parent=pin(SOURCE),model=pin(MODEL_DIR/'model.npz'),
          reused_development_networks=True,target_arrays_decoded=False)
for p in list(meta['weights'].values())+list(meta['targets'].values()):
    check(p)
write(HERE/'inputs.json',meta)
external=[SOURCE,MODEL_DIR/'model.npz',MODEL_DIR/'normalizers.npz',MODEL_DIR/'model_frozen.json',
          MODEL_DIR/'information_result.json',MODEL_DIR/'replay_result.json',MODEL_DIR/'build.py',MODEL_DIR/'head.py',
          MODEL_DIR/'summary.json',old/'native.py',old/'guard.py',old/'retirement.py',
          A/'birth_direction_fit_v1/native_v2.py',D/'official_best_331760.json']
frozen=read(MODEL_DIR/'model_frozen.json')
for k in ('model','normalizers','training'):
    check(frozen[k])
    external.append(Path(frozen[k]['path']))
pins=[pin(p) for p in sorted(set(external+list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'inputs.json']))]
write(HERE/'sources_sealed.json',dict(pins=pins,original_proxy_gate_passed=False,
      old_budget_context_ASTs_exact=True,frozen_model_no_fit=True,source_parent_submission=331760))
print(dict(prepared=True,pins=len(pins),native_scopes_exact=True,old_proxy_gate_preserved=True))
