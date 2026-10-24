"""Actual official0.12.3 tiny GNN functions vs independent NumPy equations.

No estimator import, weights, network predictions, or targets. This verifies
shape bindings and arithmetic implementation, not compression accuracy.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib
import numpy as np


def decode(x):
    return {k:np.frombuffer(zlib.decompress(base64.b64decode(v)),dtype='<f4').reshape(shape).copy()
            for k,(shape,v) in x.items()}


def load_functions(source):
    tree = ast.parse(source)
    definitions = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    names = ('_E2Hold','silu','_e2_gnn_consts','_e2_gnn_layer','_gop2_contract',
             '_gop2_messages','_e2_messages','_gop2_gnn_step','_gop2_message_product',
             '_w3r_metric','_w3r_product','_unpack_static_fp32')
    return compile(ast.Module(body=[definitions[name] for name in names],type_ignores=[]),'<tiny-actual-gnn>','exec')


def gop(T,X):
    mu=X.mean(axis=1);center=X-mu[:,None,:];wide=center.astype('f8')
    _,V=np.linalg.eigh(wide.transpose(0,2,1)@wide);V=V[:,:,-2:].astype('f4')
    projected=T@(center@V);Q,_=np.linalg.qr(projected.astype('f8'),mode='reduced');Q=Q.astype('f4')
    return Q@((Q.transpose(0,2,1)@T)@center)+T.sum(axis=2)[:,:,None]*mu[:,None,:]


def w3(T,X,E):
    e,V=np.linalg.eigh(E.T@E);s=np.sqrt(np.maximum(e,1e-30))
    root=(V*s)@V.T;inv=(V/s)@V.T;wide=X.astype('f8');G=wide.T@wide
    _,V=np.linalg.eigh((root@G)@root);V=V[:,-4:]
    L=(root@V).astype('f4');B=(V.T@inv).astype('f4')
    return (T@(X@L))@B


def oracle(P,E,layer,x,sigma,W,state,bias,one_hidden=False, lowrank=None):
    n=len(x);cm=P['msg_w'].shape[0]//3
    src=(state@P['msg_w'].T+P['msg_b']).reshape(n,3,cm).transpose(1,0,2)
    w2=W*W;w3_=w2*W;w4=w3_*W;w6=w3_*w3_
    M=np.empty((3,n,cm),'f4');M[0]=W.T@src[0];M[1:2]=gop(w2.T[None,:,:],src[1:2]);M[2]=w3(w3_.T,src[2],E)
    E2=gop(np.stack((w2.T,w4.T,w6.T)),src**2)
    Hx=np.clip((x-P['centers'][layer])/P['scales'][layer],-8,8)
    if layer==15:
        Hx=Hx*np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1],'f4')
    hm=np.concatenate((M*Hx[:,5:6],np.log1p(np.maximum(E2,0))),axis=0).transpose(1,0,2).reshape(n,6*cm)
    H=np.concatenate((Hx,np.clip(hm,-8,8)),axis=1)
    silu=lambda h:h/(1+np.exp(-h))
    h=silu(H@P['enc0_w'].T+bias)
    if not one_hidden:
        h=silu(h@P['enc2_w'].T+P['enc2_b']) if lowrank is None else silu((h@lowrank['left'])@lowrank['right']+lowrank['bias'])
    out=h@P['enc4_w'].T+P['enc4_b'];gain=1+.25*np.tanh(P['gain'][layer:layer+1])
    delta=(sigma.astype('f4')*P['rms'][layer])*(out@P['head'][layer]+P['head_b'][layer])*gain
    return out,delta


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--official',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();sys.path.insert(0,str(a.official))
    import flopscope as fs
    from flopscope import numpy as fnp
    assert fs.__version__.startswith('0.12.3')
    manifest=json.loads(a.manifest.read_text());rng=np.random.default_rng(20261020)
    cases=[]
    for cand in manifest['candidates']:
        source=Path(cand['science']['path']).read_text();code=load_functions(source)
        tail=Path(cand['tail']).read_text()
        for n,layer in ((16,5),(32,15)):
            env=dict(_O20_FLATEXTRA_ON=True,_O20_FLATEXTRA_OWNED_PARENT=lambda:None,_O20_FLATEXTRA_HZD_PARENT=lambda:None,fnp=fnp,_static_base64=base64,_static_zlib=zlib,_GOP2_ON=True,_GOP2_ACTIVE=[True],_GNN_PARAMS=None,
                     _E2_GNN_STATE={},_E2_MSG={},_FAST_RESCUE_FRAC={},
                     _HC_CONST={},_HCF_CONST={},_MESSAGE_EDGE_PARAMETERS=None,
                     _GOP2_STATS=dict(calls=0,layers=[],energy_blocks=0,message_misses=0),
                     _W3R_STATS=dict(calls=0,metric_builds=0))
            exec(code,env);exec(tail,env)
            P=decode(env['_GNN_LITERAL']);cm=P['msg_w'].shape[0]//3
            assert P['enc0_w'].shape[1]==30+6*cm
            assert P['enc2_w'].shape[1]==P['enc0_w'].shape[0]
            assert P['enc4_w'].shape[1]==P['enc2_w'].shape[0]
            assert np.asarray(env.get('_HC_U',[])).shape[-1:] in ((P['enc0_w'].shape[0],), (0,))
            assert np.asarray(env.get('_HCF_U',[])).shape[-1:] in ((P['enc0_w'].shape[0],), (0,))
            assert env['_FAST_RESCUE_FRAC']=={}
            W=rng.normal(0,.2,(n,n)).astype('f4');state=rng.normal(0,.3,(n,64)).astype('f4')
            x=(rng.normal(0,.5,(n,30))*P['scales'][layer]+P['centers'][layer]).astype('f4')
            sigma=rng.uniform(.2,1.2,n).astype('f4')
            bias=(P['enc0_b']+rng.normal(0,.03,(n,P['enc0_w'].shape[0]))).astype('f4')
            E=np.asarray(env['_VMFR_ORIGINAL_W3_ENCODER'],'f8')
            oracleP={k:v.copy() for k,v in P.items()}
            if '_VMFR_NL_LITERALS' in env:
                assert set(env['_VMFR_NL_LITERALS'])==set(range(1,16))
                oracleP.update(decode(env['_VMFR_NL_LITERALS'][layer]))
                bias += oracleP['enc0_b']-P['enc0_b']
            lowrank=decode(env['_VMFR_LR_LITERALS'][layer]) if '_VMFR_LR_LITERALS' in env else None
            expected,edelta=oracle(oracleP,E,layer,x,sigma,W,state,bias,env.get('_VMFR_ONE_HIDDEN',False),lowrank)
            with fs.BudgetContext(flop_budget=2**32,quiet=True) as ctx:
                Q={k:fnp.asarray(v) for k,v in P.items()};env['_GNN_PARAMS']=Q
                x_,s_,w_,st_,b_=[fnp.asarray(v) for v in (x,sigma,W,state,bias)]
                w2=w_*w_;w3_=w2*w_;w4=w3_*w_;w6=w3_*w3_
                wp=dict(T013=fnp.stack((w_.T,w2.T,w3_.T)),T135=fnp.stack((w2.T,w4.T,w6.T)))
                env['_e2_wpow']=lambda W,need6=False:wp
                env['_hc_bias']=lambda l,unused:b_
                bills=[];calls=[];outputs=[]
                for repeat in range(3):
                    env['_E2_MSG'].clear()
                    if repeat==2:
                        env['_e2_messages'](env['_e2_gnn_consts'](Q,n),w_,st_)
                    before=ctx.flops_used;cnt=len(ctx.op_log)
                    d,got=env['_gop2_gnn_step'](layer,x_,s_,w_,st_,Q)
                    bills.append(ctx.flops_used-before);calls.append(len(ctx.op_log)-cnt)
                    outputs.append((np.asarray(got).copy(),np.asarray(d).copy()))
                np.testing.assert_array_equal(outputs[0][0],outputs[1][0])
                np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
                np.testing.assert_array_equal(outputs[0][0],outputs[2][0])
                np.testing.assert_array_equal(outputs[0][1],outputs[2][1])
            err=float(np.max(np.abs(outputs[0][0]-expected)))
            derr=float(np.max(np.abs(outputs[0][1]-edelta)))
            np.testing.assert_allclose(outputs[0][0],expected,atol=5e-5,rtol=5e-5)
            np.testing.assert_allclose(outputs[0][1],edelta,atol=2e-8,rtol=1e-4)
            cases.append(dict(candidate=cand['name'],n=n,layer=layer,cm=cm,
                              hidden=P['enc0_w'].shape[0],max_state_difference=err,
                              max_delta_difference=derr,flops=bills,calls=calls,
                              repeat_byte_exact=True))
    result=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
                status='TINY_OFFICIAL_FUNCTIONS_PASS_NO_NETWORK_SCORE',cases=cases,
                manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
