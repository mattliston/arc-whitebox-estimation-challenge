"""Small covariance-input extension to the existing approximate Torch card."""
from pathlib import Path
import ast
import ctypes
import gc
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
H2=HERE.parent
P2=H2.parent
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
libsystem=ctypes.CDLL('/usr/lib/libSystem.B.dylib')
relief=libsystem.malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
relief.restype=ctypes.c_size_t
def collect():
    gc.collect()
    return int(relief(None,0))

def initialize():
    for relative in ['fable_analyst_v1','fable_covhunter_v1','fable_engineer_v1','fable_physicist_v1/refit6b','fable_porter_v1']:
        sys.path.insert(0,str(P2/relative))
    import numpy as np
    import torch
    card_path=P2/'fable_physicist_v1/refit6b/s6bcard_loop5js.py'
    spec=importlib.util.spec_from_file_location('rnm_loop5js',card_path)
    LC=importlib.util.module_from_spec(spec);spec.loader.exec_module(LC)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_grad_enabled(True)
    assert LC.SCHED=='384-320-384'
    checkpoint=P2/'fable_physicist_v1/refit6b/s6b_loopR8s_joint_3ep/params_epoch1.pt'
    dials_path=checkpoint.with_name('dials_epoch1.json')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    payload=(saved['p'],saved['a'],saved['c'],saved['e'])
    for group in payload:
        for value in (group.values() if isinstance(group,dict) else [group]):value.requires_grad_(False)
    D=json.loads(dials_path.read_text())
    LC.JT.DIALS_T={k:torch.tensor(D[k],dtype=torch.float64) for k in LC.JT.KEYS}
    store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
    manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    parts=json.loads((P2/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
    assert set(PROTOCOL['fit_ranks']).isdisjoint(PROTOCOL['check_ranks'])
    assert set(PROTOCOL['fit_ranks']+PROTOCOL['check_ranks'])<=set(parts['train'])
    assert sha(H2/'candidate_old_both_tail_trace_bounded_v6.py')==PROTOCOL['parent_sha256']
    # Only the single original function is translated; no literals are printed.
    source=LC.model.SOURCE.read_text();lines=source.splitlines()
    node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_message_edge_source'][-1]
    body='\n'.join(lines[node.lineno-1:node.end_lineno])
    old="    F=F/t['feature_scale'];R=R/t['rho_scale'];S=S/t['third_scale']"
    assert body.count(old)==1
    body=body.replace(old,"    nf=_RNM_INPUT(F,R,S,a,p,t)\n"+old)
    for side in ('left','right'):
        old=f"    {side}=F@t['{side}']+messages@context['{side}']"
        assert body.count(old)==1
        body=body.replace(old,old+f"\n    {side}={side}+nf@_RNM_{side.upper()}")
    tree=LC.model.Translate().visit(ast.parse(body));ast.fix_missing_locations(tree)
    compiled=compile(tree,str(HERE/'model.py')+':edge_extension','exec')
    binding=dict(protocol_sha256=sha(HERE/'protocol.json'),model_sha256=sha(HERE/'model.py'),
                 card_sha256=sha(card_path),checkpoint_sha256=sha(checkpoint),dials_sha256=sha(dials_path),
                 translated_edge_ast_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
                 torch=torch.__version__,threads=torch.get_num_threads(),targets_accessed=False)
    def weights(rank):
        assert rank in parts['train']
        return np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
    def feature(F,R,T,a,p,edge,mode):
        if mode=='control':return (F/edge['feature_scale'])[:,:12]
        assert mode=='message'
        phi=torch.stack((p,a*p,(a*a-1)*p),dim=1).to(torch.float32)
        R=R-torch.diag(torch.diag(R));T=T-torch.diag(torch.diag(T))
        return torch.cat((R@phi/32.,(R*R)@phi,T@phi/32.,T.T@phi/32.),dim=1)
    def card(w,theta=None,mode='message',mean=None,scale=None):
        cd=LC.card(torch,w,*payload,fine_ckpt=False)
        cd.hooks['k4_15']=lambda l,value,ctx:value
        if theta is None:return cd
        assert theta.shape==(2,12,16)
        if mean is None:mean=torch.zeros(12,dtype=torch.float32)
        if scale is None:scale=torch.ones(12,dtype=torch.float32)
        cd.env.update(_RNM_LEFT=theta[0],_RNM_RIGHT=theta[1],
                      _RNM_INPUT=lambda F,R,T,a,p,edge:torch.clamp((feature(F,R,T,a,p,edge,mode)-mean)/scale,-8,8))
        exec(compiled,cd.env)
        return cd
    def run(cd,checkpointed=False,progress=None):
        from torch.utils.checkpoint import checkpoint
        state=cd.initial();outputs=[]
        for layer in range(16):
            if checkpointed and layer>=8:
                state=checkpoint(lambda *args,layer=layer:cd.step(layer,args),*state,use_reentrant=False)
            else:state=cd.step(layer,state)
            outputs.append(state[0])
            collect()
            if progress:progress(layer)
        return torch.stack(outputs).clamp_min(0)
    return dict(np=np,torch=torch,LC=LC,payload=payload,weights=weights,feature=feature,card=card,run=run,binding=binding)
