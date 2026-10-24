from common import *
verify()
import ast
import copy
import os
import numpy as np
import torch
import model_hidden as model
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9409215)
tree=model.extracted();gnn=next(n for n in tree.body if n.name=='gnn_step');silu=next(n for n in tree.body if n.name=='silu')
added=[copy.deepcopy(n) for n in gnn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
       and (n.targets[0].id=='extra_hidden' or
            (n.targets[0].id=='state' and isinstance(n.value,ast.BinOp)
             and isinstance(n.value.left,ast.Name) and n.value.left.id=='state'))]
assert len(added)==2
branch=ast.parse('def branch(h,state,P):\n    pass\n').body[0]
branch.body=added+[ast.Return(value=ast.Name(id='state',ctx=ast.Load()))]
unit_tree=ast.fix_missing_locations(model.Translate().visit(ast.Module(body=[copy.deepcopy(silu),branch],type_ignores=[])))
env={'fnp':model.TorchNP(torch)};exec(compile(unit_tree,'exact_added_branch_torch','exec'),env)
_,_,_,initial=model.parameters(np)
native=read(N/'unit_result.json');assert native['passed'] and len(native['fixture_pins'])==3
records=[]
for fixture in native['fixture_pins']:
    check(fixture)
    with np.load(fixture['path'],allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    h=data['h'];state=data['state'];width=h.shape[0]
    assert width in (5,7,9) and h.shape==(width,128) and state.shape==(width,64)
    assert data['in_w'].tobytes()==initial['extra_hidden_w'].tobytes()
    assert data['in_b'].tobytes()==initial['extra_hidden_b'].tobytes()
    assert data['out_w_zero'].tobytes()==initial['extra_state_w'].tobytes()
    direction=(np.cos(np.arange(width*64,dtype=np.float64)*.047+width)/(width*64)).reshape(width,64)
    for mode in ('zero','sentinel'):
        p={'extra_hidden_w':data['in_w'].astype(np.float64),'extra_hidden_b':data['in_b'].astype(np.float64),
           'extra_state_w':data['out_w_'+mode].astype(np.float64)}
        x=h.astype(np.float64);s=state.astype(np.float64)
        z=x@p['extra_hidden_w'].T+p['extra_hidden_b'];sig=1/(1+np.exp(-z));u=z*sig
        oracle=s+u@p['extra_state_w'].T
        dz=(direction@p['extra_state_w'])*(sig+z*sig*(1-sig))
        gradients={'extra_hidden_w':dz.T@x,'extra_hidden_b':dz.sum(0),'extra_state_w':direction.T@u}
        e={k:torch.tensor(v,dtype=torch.float32,requires_grad=True) for k,v in p.items()}
        before={k:model.tensor_sha(v) for k,v in e.items()}
        value=env['branch'](torch.tensor(h),torch.tensor(state),e)
        loss=(value.to(torch.float64)*torch.tensor(direction)).sum();loss.backward()
        assert {k:model.tensor_sha(v) for k,v in e.items()}==before
        forward_error=float(np.linalg.norm(value.detach().numpy().astype(np.float64)-oracle)/max(1.,np.linalg.norm(oracle)))
        assert forward_error<=5e-6
        errors={}
        for k in EKEYS:
            assert e[k].grad is not None and bool(torch.isfinite(e[k].grad).all())
            errors[k]=float(np.linalg.norm(e[k].grad.numpy().astype(np.float64)-gradients[k])/max(1.,np.linalg.norm(gradients[k])))
            assert errors[k]<=5e-6,(width,mode,k,errors[k])
        if mode=='zero':
            assert value.detach().numpy().tobytes()==state.tobytes()
            assert torch.count_nonzero(e['extra_hidden_w'].grad)==0 and torch.count_nonzero(e['extra_hidden_b'].grad)==0
            assert bool(torch.count_nonzero(e['extra_state_w'].grad)>0)
        else:assert all(bool(torch.count_nonzero(e[k].grad)>0) for k in EKEYS)
        vector={k:np.sin(np.arange(v.size,dtype=np.float64)*.031+.3).reshape(v.shape) for k,v in p.items()}
        length=sum(float((v*v).sum()) for v in vector.values())**.5
        vector={k:v/length for k,v in vector.items()}
        analytic=sum(float((gradients[k]*vector[k]).sum()) for k in EKEYS)
        def loss64(sign):
            pp={k:p[k]+sign*1e-5*vector[k] for k in EKEYS}
            zz=x@pp['extra_hidden_w'].T+pp['extra_hidden_b']
            yy=s+(zz/(1+np.exp(-zz)))@pp['extra_state_w'].T
            return float((yy*direction).sum())
        finite=(loss64(1)-loss64(-1))/(2e-5)
        fd_error=abs(finite-analytic)/max(1e-10,abs(analytic),abs(finite))
        assert fd_error<=2e-6,(width,mode,fd_error)
        records.append(dict(width=width,mode=mode,fixture=fixture,forward_error=forward_error,
            gradient_errors=errors,directional_finite_difference_error=fd_error,
            parameter_values_unchanged=True,all_gradients_finite=True,
            incoming_zero_gradient_expected=mode=='zero',nonzero_mode_all_gradients_nonzero=mode=='sentinel'))
        del value,loss,e
write(HERE/'unit_result.json',dict(passed=True,cases=records,exact_added_expression_AST=True,
    independent_FP64_forward_and_three_parameter_pullbacks=True,
    independent_directional_central_difference=True,finite_difference_step=1e-5,
    targets_decoded=False,fitted=False,native_derivative_certified=False))
print(dict(passed=True,cases=len(records),max_forward_error=max(r['forward_error'] for r in records),
    max_gradient_error=max(v for r in records for v in r['gradient_errors'].values())),flush=True)
