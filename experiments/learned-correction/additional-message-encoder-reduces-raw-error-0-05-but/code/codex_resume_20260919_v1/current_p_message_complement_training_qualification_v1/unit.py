"""Independent component forward and encoder/state pullbacks, without targets."""
from common import *
verify()
import ast
import copy
import numpy as np
import torch
import model_complement as model
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9409216)
gnn=next(n for n in model.extracted().body if n.name=='gnn_step')
added=[copy.deepcopy(n) for n in gnn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('extra_')]
assert len(added)==9
branch=ast.parse('def branch(g,w,w2,w3,state,P):\n    pass\n').body[0]
branch.body=added+[ast.Return(value=ast.Name(id='extra_pre',ctx=ast.Load()))]
tree=ast.fix_missing_locations(model.Translate().visit(ast.Module(body=[branch],type_ignores=[])))
env={'fnp':model.TorchNP(torch)};exec(compile(tree,'exact_complement_branch_torch','exec'),env)
p,_,_,e=model.parameters(np);fixed=p['complement_msg_w'];zero=e['complement_enc_w']
# Consume the already authenticated native sentinel exactly; do not reconstruct
# transcendental initialization values in a different NumPy runtime.
literal=next(n.value for n in ast.parse((SMALL/'sentinel.py').read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_GNN_LITERAL')
shape,blob=ast.literal_eval(literal)['complement_enc_w']
assert shape==(128,288)
sentinel=np.frombuffer(model.zlib.decompress(model.base64.b64decode(blob,validate=True)),dtype='<f4').reshape(shape).copy()
assert np.isfinite(sentinel).all() and np.any(sentinel!=0)
records=[]
for width in (5,7,9):
    state=(np.cos(np.arange(width*64,dtype=np.float64)*.051-width)*.2).astype(np.float32).reshape(width,64)
    w=(np.sin(np.arange(width*width,dtype=np.float64)*.097+width)*(.2/width**.5)).astype(np.float32).reshape(width,width)
    g=(.4+.3*np.sin(np.arange(width,dtype=np.float64)*.21)).astype(np.float32)[:,None]
    wd=w.astype(np.float64);gd=g.astype(np.float64);sd=state.astype(np.float64);basis=fixed.astype(np.float64)
    direction=(np.cos(np.arange(width*128,dtype=np.float64)*.047+width)/(width*128)).reshape(width,128)
    def messages(s):
        src=s@basis.T
        signed=[gd*((wd**power).T@src[:,48*i:48*(i+1)]) for i,power in enumerate((1,2,3))]
        energies=[(wd**(2*power)).T@(src[:,48*i:48*(i+1)]**2) for i,power in enumerate((1,2,3))]
        raw=np.concatenate(signed+[np.log1p(np.maximum(v,0)) for v in energies],axis=1)
        return np.clip(raw,-8,8),raw,src,energies
    features,raw,src,energies=messages(sd)
    assert float(np.min(np.abs(np.abs(raw)-8)))>1e-3
    for mode,encoder in [('zero',zero),('sentinel',sentinel)]:
        ed=encoder.astype(np.float64);oracle=features@ed.T
        grad_encoder=direction.T@features
        grad_messages=(direction@ed)*(np.abs(raw)<8)
        grad_src=np.zeros_like(src)
        for group,power in enumerate((1,2,3)):
            section=slice(48*group,48*(group+1));energy_section=slice(144+48*group,144+48*(group+1))
            grad_src[:,section]=(wd**power)@(gd*grad_messages[:,section])
            grad_src[:,section]+=2*src[:,section]*((wd**(2*power))@(grad_messages[:,energy_section]/(1+energies[group])))
        grad_state=grad_src@basis
        leaves={'complement_enc_w':torch.tensor(encoder,requires_grad=True),'complement_msg_w':torch.tensor(fixed)}
        incoming=torch.tensor(state,requires_grad=True);pw=torch.tensor(w);pw2=pw*pw;pw3=pw2*pw
        before={key:model.tensor_sha(value) for key,value in leaves.items()}
        value=env['branch'](torch.tensor(g),pw,pw2,pw3,incoming,leaves)
        loss=(value.to(torch.float64)*torch.tensor(direction)).sum();loss.backward()
        assert {key:model.tensor_sha(value) for key,value in leaves.items()}==before
        assert model.tensor_sha(incoming)==hashlib.sha256(state.tobytes()).hexdigest()
        assert leaves['complement_msg_w'].grad is None
        forward=float(np.linalg.norm(value.detach().numpy().astype(np.float64)-oracle)/max(1.,np.linalg.norm(oracle)))
        errors={}
        for key,actual,expected in [('encoder',leaves['complement_enc_w'].grad,grad_encoder),('incoming_state',incoming.grad,grad_state)]:
            assert actual is not None and bool(torch.isfinite(actual).all())
            errors[key]=float(np.linalg.norm(actual.numpy().astype(np.float64)-expected)/max(1.,np.linalg.norm(expected)))
            assert errors[key]<=5e-6,(width,mode,key,errors[key])
        assert forward<=5e-6
        assert bool(torch.count_nonzero(leaves['complement_enc_w'].grad)>0)
        if mode=='zero':assert torch.count_nonzero(incoming.grad)==0 and torch.count_nonzero(value)==0
        else:assert bool(torch.count_nonzero(incoming.grad)>0) and bool(torch.count_nonzero(value)>0)
        ve=np.sin(np.arange(ed.size,dtype=np.float64)*.031+.3).reshape(ed.shape)
        vs=np.cos(np.arange(sd.size,dtype=np.float64)*.023-.2).reshape(sd.shape)
        length=float(np.sqrt((ve*ve).sum()+(vs*vs).sum()));ve/=length;vs/=length
        analytic=float((grad_encoder*ve).sum()+(grad_state*vs).sum())
        def loss64(sign):return float(((messages(sd+sign*1e-5*vs)[0]@(ed+sign*1e-5*ve).T)*direction).sum())
        finite=(loss64(1)-loss64(-1))/(2e-5)
        fd_error=abs(finite-analytic)/max(1e-12,abs(analytic),abs(finite))
        assert fd_error<=2e-6,(width,mode,fd_error)
        records.append(dict(width=width,mode=mode,forward_error=forward,gradient_errors=errors,
            directional_finite_difference_error=fd_error,parameters_and_inputs_unchanged=True,
            fixed_basis_gradient_free=True,encoder_gradient_nonzero=True,zero_state_gradient_expected=mode=='zero'))
        del value,loss,leaves,incoming,pw,pw2,pw3
write(HERE/'unit_result.json',dict(passed=True,cases=records,exact_added_expression_AST=True,
    independent_FP64_forward_encoder_and_state_pullbacks=True,independent_directional_central_difference=True,
    finite_difference_step=1e-5,targets_decoded=False,fitted=False,native_derivative_certified=False))
print(dict(passed=True,cases=len(records),max_forward_error=max(r['forward_error'] for r in records),
    max_gradient_error=max(v for r in records for v in r['gradient_errors'].values())),flush=True)
