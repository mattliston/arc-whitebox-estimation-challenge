"""Verify exact fitted edge6 tail, NumPy-free imports separately, and live official arithmetic."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import flopscope as fs
from flopscope import numpy as fnp
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert fs.__version__.startswith('0.12.3') and not os.environ.get('FLOPSCOPE_SERVER_URL')
fit=read(HERE/'FITTED_v1.json')
assert fit['no_truth_read'] and fit['no_validation_predictions_read'] and fit['train']==[35,150,200,250]
assert len(fit['delta'])==7 and all(abs(v)<=.10000000001 for v in fit['delta'])
assert fit['linear_training_ratio']<1
curv=[r['relative_second_difference'] for r in fit['curvature']]
assert all(np.isfinite(curv)),curv
base_dir=HERE.parent/'edge_hidden6_nonlinear_v1/native_hidden6_v1'
for role in ('science','release'):
    parent=base_dir/('candidate_msg8_edge_hidden6_nonlinear_'+role+'_v1.py')
    output=Path(fit['outputs'][role]['path'])
    assert sha(parent)==fit['outputs'][role]['parent_sha256']
    assert sha(output)==fit['outputs'][role]['sha256']
    pt=ast.parse(parent.read_text());ot=ast.parse(output.read_text())
    assert all(ast.dump(a)==ast.dump(b) for a,b in zip(pt.body,ot.body[:len(pt.body)]))
    extra=ot.body[len(pt.body):]
    assert len(extra)==2 and all(isinstance(x,ast.Assign) for x in extra)
    assert extra[0].targets[0].id=='_GN_EDGE' and extra[1].targets[0].id=='_FAST_RESCUE_FRAC'
    assert ast.literal_eval(extra[0].value)=={int(k):v for k,v in fit['GN_EDGE'].items()}
    assert ast.literal_eval(extra[1].value)=={}
parent=base_dir/'candidate_msg8_edge_hidden6_nonlinear_science_v1.py'
pt=ast.parse(parent.read_text())
defs={n.name:n for n in pt.body if isinstance(n,ast.FunctionDef)}
assign={t.id:n.value for n in pt.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
base=ast.literal_eval(assign['_GN_EDGE'])
new={int(k):v for k,v in fit['GN_EDGE'].items()}
assert set(base)==set(new)==set(range(8,15))
code=compile(ast.Module(body=[defs['_ps32'],defs['_message_edge_source']],type_ignores=[]),'<live-GN-wrapper>','exec')
env=dict(fnp=fnp,_GN_EDGE=dict(base),_E2_LAYER={'mu':object()},_RQD_PS32=[None,{}],_RQD_STATS={'hits':0})
exec(code,env)
values=np.arange(16,dtype='f4').reshape(4,4)/32-np.float32(.2)
cases=[]
with fs.BudgetContext(flop_budget=2**28,quiet=True) as ctx:
    native=fnp.asarray(values)
    env['_gn_message_edge_source_base']=lambda *args:native
    def run(layer):
        before,count=ctx.flops_used,len(ctx.op_log)
        y=env['_message_edge_source'](layer,None,None,None,None,None,None,None)
        return np.asarray(y).copy(),ctx.flops_used-before,len(ctx.op_log)-count
    expected={layer:run(layer) for layer in range(8,15)}
    for phase in ('cold','warm','restore','prewarmed'):
        env['_GN_EDGE']=dict(base if phase=='restore' else new)
        for layer in range(8,15):
            got,bill,calls=run(layer)
            assert (bill,calls)==expected[layer][1:]
            t=env['_GN_EDGE'][layer]
            oracle=(values*np.float32(1+t)).astype('f8')
            np.fill_diagonal(oracle,np.diag(values).astype('f8')*(1+t))
            np.testing.assert_array_equal(got,oracle)
            if phase=='restore':np.testing.assert_array_equal(got,expected[layer][0])
            cases.append(dict(phase=phase,layer=layer,bill=bill,calls=calls,oracle_exact=True))
receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    successful=True,fitted_sha256=sha(HERE/'FITTED_v1.json'),script_sha256=sha(__file__),
    sources=fit['outputs'],source_prefix_AST_exact=True,only_GN_literal_tail=True,
    official_version=fs.__version__,zero_added_operations=True,cases=cases,
    curvature_max=max(curv),curvature_median=float(np.median(curv)),native_accuracy_unknown=True)
with (HERE/'native_v1/TINY_SCALE_v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('cases','sources')},indent=2))

