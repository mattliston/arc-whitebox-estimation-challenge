"""Independent small physical-head equations and readout/input derivatives."""
import ast
import math
from common import *
verify()
import numpy as np
import torch
import model_readout as model
import head_translation
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9409221)
_,_,_,values=model.parameters(np)
fixed={k:values['joint_'+k] for k in JKEYS}
scale=values['joint_output_scale']
literal=next(ast.literal_eval(n.value) for n in ast.parse((SMALL/'sentinel.py').read_text()).body
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_JOINT_LITERAL')
shape,blob=literal['output'];assert tuple(shape)==(32,3)
old=np.frombuffer(model.zlib.decompress(model.base64.b64decode(blob,validate=True)),dtype='<f4').reshape(shape).copy()
theta_sentinel=(old/scale).astype(np.float32)
folded=(theta_sentinel*scale).astype(np.float32)
fold_error=float(np.max(np.abs(folded.astype(np.float64)-old)/np.maximum(np.abs(old),1e-12)))
assert fold_error<=2e-7 and np.any(theta_sentinel!=0)
path=HERE/'sentinel_theta.npy'
with path.open('xb') as f:np.save(f,theta_sentinel,allow_pickle=False)
env=head_translation.environment(torch)
input_keys=('mu','C','K','radial','W','H')
records=[]

def oracle(layer,inputs,theta,work_dtype):
    mu,C,K,radial,W,H=(inputs[k].astype(np.float64) for k in input_keys)
    sigma=np.sqrt(np.maximum(np.diag(C),1e-10));alpha=mu/sigma
    pdf=np.exp(-.5*alpha*alpha)/2.5066282746310002
    gate=np.asarray([math.erfc(-float(a)/math.sqrt(2))/2 for a in alpha])
    rho=C/np.outer(sigma,sigma);third=K/(sigma[:,None]**2*sigma[None,:])
    one=np.ones(len(mu));logs=np.log(sigma)
    node=np.stack([one,one*(layer/15),alpha,gate,pdf,alpha*pdf,(alpha*alpha-1)*pdf,
        logs-np.mean(logs),np.diag(third),one*radial,np.mean(rho,axis=1),np.sqrt(np.mean(rho*rho,axis=1))],axis=1).astype(work_dtype)
    ww=W.astype(work_dtype);hh=H.astype(work_dtype)
    feature=np.concatenate([node,ww.T@hh,(ww*ww).T@hh],axis=1)
    par={k:v.astype(work_dtype) for k,v in fixed.items()}
    feature=np.clip(feature/par['feature_scale'],-8,8)
    rho=rho.astype(work_dtype)/par['rho_scale'];third=third.astype(work_dtype)/par['third_scale']
    left=feature@par['left'];right=feature@par['right']
    pre=left[:,None,:]+right[None,:,:]+par['bias']
    pre=pre+rho[:,:,None]*par['edge'][0]+third[:,:,None]*par['edge'][1]+third.T[:,:,None]*par['edge'][2]
    output=theta.astype(work_dtype)*scale.astype(work_dtype)
    raw=(np.tanh(pre)@output)*par['target_scale']
    a=raw[:,:,0].copy();b=raw[:,:,1].copy();c=.5*(raw[:,:,2]+raw[:,:,2].T)
    np.fill_diagonal(a,0);np.fill_diagonal(b,0);np.fill_diagonal(c,0)
    variance=np.diag(C)
    return [a.astype(np.float64)*(variance[:,None]*sigma[None,:]),b.astype(np.float64),
            c.astype(np.float64)*np.outer(variance,variance)]

for width in (5,9):
    q=np.arange(width*width,dtype=np.float64).reshape(width,width)
    B=np.sin(.19*q+.3)/np.sqrt(width)
    C=B@B.T+.7*np.eye(width)
    inputs=dict(mu=np.sin(np.arange(width,dtype=np.float64)*.23)*.2,C=C,
        K=np.cos(q*.071-.2)*.03,radial=np.asarray(.02,dtype=np.float64),
        W=(np.sin(q*.13+.7)*(.35/np.sqrt(width))).astype(np.float32),
        H=(np.cos(np.arange(width*64,dtype=np.float64).reshape(width,64)*.051-.2)*.2).astype(np.float32))
    for layer in (8,14):
        directions=[np.cos(q*.097+.1+j)/(width*width) for j in range(3)]
        for mode,theta in [('zero',values['joint_theta']),('sentinel',theta_sentinel)]:
            dtype_records={};high_grads=None
            for label,work in [('float64',np.float64),('float32',np.float32)]:
                td=torch.float64 if work==np.float64 else torch.float32
                api=model.TorchNP(torch);api.float32=td
                leaves={k:torch.tensor(v,dtype=torch.float64 if k not in ('W','H') else td,requires_grad=True)
                        for k,v in inputs.items()}
                th=torch.tensor(theta,dtype=td,requires_grad=True)
                params={k:torch.tensor(v,dtype=td) for k,v in fixed.items()}
                params['output']=th*torch.tensor(scale,dtype=td)
                before={k:model.tensor_sha(v) for k,v in leaves.items()};before['theta']=model.tensor_sha(th)
                corrections=env['correction'](api,lambda a:torch.erfc(-a/math.sqrt(2))/2,layer,
                    *(leaves[k] for k in input_keys),params)
                expected=oracle(layer,inputs,theta,work)
                errors=[]
                for actual,want in zip(corrections,expected):
                    assert bool(torch.isfinite(actual).all())
                    assert torch.count_nonzero(torch.diagonal(actual))==0
                    errors.append(float(np.linalg.norm(actual.detach().numpy()-want)/max(np.linalg.norm(want),1e-10)))
                assert max(errors)<=3e-5,(width,layer,mode,label,errors)
                assert torch.equal(corrections[2],corrections[2].T)
                loss=sum((value*torch.tensor(direction)).sum() for value,direction in zip(corrections,directions))
                loss.backward()
                grads={k:v.grad.detach().numpy().astype(np.float64) for k,v in leaves.items()}
                grads['theta']=th.grad.detach().numpy().astype(np.float64)
                assert all(np.isfinite(v).all() for v in grads.values()) and np.linalg.norm(grads['theta'])>0
                assert {**{k:model.tensor_sha(v) for k,v in leaves.items()},'theta':model.tensor_sha(th)}==before
                assert all(v.grad is None for k,v in params.items() if k!='output')
                if mode=='zero':
                    assert all(torch.count_nonzero(v)==0 for v in corrections)
                    assert all(np.count_nonzero(grads[k])==0 for k in input_keys)
                else:
                    assert np.linalg.norm(grads['H'])>0 and np.linalg.norm(grads['C'])>0
                if label=='float64':
                    high_grads=grads
                    fd={}
                    for j,key in enumerate((*input_keys,'theta')):
                        value=theta.astype(np.float64) if key=='theta' else inputs[key].astype(np.float64)
                        direction=np.sin(np.arange(value.size,dtype=np.float64)*.173+.3+j).reshape(value.shape)
                        if key=='C':direction=(direction+direction.T)/2
                        direction/=max(float(np.linalg.norm(direction)),1e-12)
                        analytic=float(np.sum(grads[key]*direction))
                        losses=[]
                        for sign in (-1,1):
                            xx={k:v.astype(np.float64).copy() for k,v in inputs.items()};tt=theta.astype(np.float64).copy()
                            if key=='theta':tt+=sign*1e-5*direction
                            else:xx[key]+=sign*1e-5*direction
                            yy=oracle(layer,xx,tt,np.float64)
                            losses.append(float(sum(np.sum(v*d) for v,d in zip(yy,directions))))
                        finite=(losses[1]-losses[0])/2e-5
                        error=abs(finite-analytic)/max(1e-8,abs(finite),abs(analytic))
                        assert error<=3e-5,(width,layer,mode,key,error,finite,analytic)
                        fd[key]=dict(analytic=analytic,central_difference=finite,relative_error=error)
                    dtype_records[label]=dict(forward_errors=errors,input_and_output_directional_derivatives=fd)
                else:
                    gradient_errors={k:float(np.linalg.norm(grads[k]-high_grads[k])/max(np.linalg.norm(high_grads[k]),1e-10)) for k in grads}
                    assert max(gradient_errors.values())<=3e-4,(width,layer,mode,gradient_errors)
                    dtype_records[label]=dict(forward_errors=errors,gradient_relative_errors=gradient_errors)
            records.append(dict(width=width,layer=layer,mode=mode,precisions=dtype_records,
                zero_input_gradient_expected=mode=='zero',inputs_and_frozen_parameters_unchanged=True))
write(HERE/'unit_result.json',dict(passed=True,cases=records,case_count=len(records),precision_evaluations=2*len(records),
    source_head=pin(SMALL/'head.py'),functional_three_diagonal_translation=True,
    independent_numpy_physical_equations=True,FP64_input_and_readout_directional_derivatives=True,
    FP32_gradients_compared_to_FP64=True,finite_difference_step=1e-5,
    sentinel_theta=pin(path),sentinel_original_output_fold_relative_error=fold_error,
    sentinel_not_a_fit_initializer=True,targets_decoded=False,fitted=False,native_derivative_certified=False))
print(dict(passed=True,cases=len(records),precisions=2*len(records),sentinel_fold_error=fold_error),flush=True)

