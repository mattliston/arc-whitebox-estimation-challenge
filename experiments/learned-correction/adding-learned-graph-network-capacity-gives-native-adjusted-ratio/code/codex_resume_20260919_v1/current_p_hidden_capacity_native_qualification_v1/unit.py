from common import *
b=admission();verify()
import ast
import base64
import copy
import gc
import zlib
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

def parsed(source):
    tree=ast.parse(Path(source['path']).read_text())
    selected={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('gnn_step','silu')}
    assert set(selected)=={'gnn_step','silu'}
    module=ast.Module(body=[selected['silu'],selected['gnn_step']],type_ignores=[])
    env={'fnp':fnp};exec(compile(ast.fix_missing_locations(module),'qualified_gnn_unit','exec'),env)
    literals=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_GNN_LITERAL']
    assert len(literals)==1
    p={k:np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype=np.float32).reshape(shape).copy()
       for k,(shape,blob) in literals[0].items()}
    return env,p,selected['gnn_step']

roles={role:parsed(b[role]) for role in ('disabled','zero','sentinel')}
draft=roles['zero'][2]
added=[copy.deepcopy(n) for n in draft.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
       and (n.targets[0].id=='extra_hidden' or
            (n.targets[0].id=='state' and isinstance(n.value,ast.BinOp)
             and isinstance(n.value.left,ast.Name) and n.value.left.id=='state'))]
assert len(added)==2
branch=ast.parse('def branch(h,state,P):\n    pass\n').body[0]
branch.body=added+[ast.Return(value=ast.Name(id='state',ctx=ast.Load()))]
branch_module=ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[]))
branch_env={'silu':roles['zero'][0]['silu']};exec(compile(branch_module,'exact_added_branch_unit','exec'),branch_env)
component=[];full=[];fixtures=[]
for width in (5,7,9):
    h=(np.sin(np.arange(width*128,dtype=np.float64)*.073+width)*.4).astype(np.float32).reshape(width,128)
    state=(np.cos(np.arange(width*64,dtype=np.float64)*.051-width)*.2).astype(np.float32).reshape(width,64)
    p0=roles['zero'][1];ps=roles['sentinel'][1]
    fixture=HERE/f'branch_fixture_{width}.npz'
    with fixture.open('xb') as f:
        np.savez(f,h=h,state=state,in_w=p0['extra_hidden_w'],in_b=p0['extra_hidden_b'],
                 out_w_zero=p0['extra_state_w'],out_w_sentinel=ps['extra_state_w'])
    fixtures.append(pin(fixture))
    for role in ('zero','sentinel'):
        params=roles[role][1]
        a=h.astype(np.float64)@params['extra_hidden_w'].astype(np.float64).T+params['extra_hidden_b'].astype(np.float64)
        hidden=a/(1+np.exp(-a))
        expected=state.astype(np.float64)+hidden@params['extra_state_w'].astype(np.float64).T
        with flops.BudgetContext(flop_budget=2**30,wall_time_limit_s=30,quiet=True):
            paid={k:fnp.array(params[k],dtype=fnp.float32) for k in ('extra_hidden_w','extra_hidden_b','extra_state_w')}
            output=branch_env['branch'](fnp.array(h,dtype=fnp.float32),fnp.array(state,dtype=fnp.float32),paid)
        actual=np.asarray(output).copy()
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        error=float(np.max(np.abs(actual.astype(np.float64)-expected))/(1+np.max(np.abs(expected))))
        assert error<5e-6,(width,role,error)
        if role=='zero':assert actual.tobytes()==state.tobytes()
        else:assert actual.tobytes()!=state.tobytes()
        component.append(dict(width=width,role=role,relative_to_one_plus_max_error=error,
                              output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        del paid,output,actual
    for layer in (0,1,8,15):
        x=(p0['centers'][layer]+p0['scales'][layer]*np.sin(np.arange(width*30,dtype=np.float64).reshape(width,30)*.029+layer)).astype(np.float32)
        sigma=(.7+.02*np.cos(np.arange(width,dtype=np.float64))).astype(np.float32)
        w=(np.sin(np.arange(width*width,dtype=np.float64)*.097+layer)*(.2/width**.5)).astype(np.float32).reshape(width,width)
        answers={}
        for role,(env,params,_) in roles.items():
            with flops.BudgetContext(flop_budget=2**30,wall_time_limit_s=30,quiet=True):
                paid={k:fnp.array(v,dtype=fnp.float32) for k,v in params.items()}
                output=env['gnn_step'](layer,fnp.array(x,dtype=fnp.float32),fnp.array(sigma,dtype=fnp.float32),
                    fnp.array(w,dtype=fnp.float32),fnp.array(state,dtype=fnp.float32),paid)
            values=tuple(np.asarray(v).copy() for v in output)
            assert values[0].shape==(width,) and values[1].shape==(width,64)
            assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in values)
            answers[role]=tuple(v.tobytes() for v in values)
            del output,paid,values
        assert answers['zero']==answers['disabled'],(width,layer,'zero identity')
        if layer==0:assert answers['sentinel']==answers['disabled']
        else:assert answers['sentinel']!=answers['disabled']
        full.append(dict(width=width,layer=layer,zero_delta_and_state_exact=True,
                         sentinel_analytic_layer_exact=layer==0,sentinel_effect=layer>0))
    gc.collect()
write(HERE/'unit_result.json',dict(passed=True,component_cases=component,full_gnn_cases=full,
    fixture_pins=fixtures,component_expression_AST_exact=True,independent_float64_forward_oracle=True,
    max_component_forward_error=max(r['relative_to_one_plus_max_error'] for r in component),
    tiny_gnn_calls=36,full_network_calls=0,targets_decoded=False,fitted=False,
    parameter_gradients_tested=False,native_derivative_certified=False))
print(dict(passed=True,component_cases=len(component),gnn_cases=len(full)),flush=True)
