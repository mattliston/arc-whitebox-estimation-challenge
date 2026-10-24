"""Independent endpoint adapter; old failed proxy study stays untouched."""
import ast,base64,hashlib,json,zlib
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent;P2=H.parents[1]
OLD=P2/'codex_resume_20260919_v1/joint_incoming_moment_head_v1'
PARENT=H.parent/'candidate_old_both_tail_trace_bounded_v6.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PARENT)=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
frozen=json.loads((OLD/'model_frozen.json').read_text())
for k in ('model','normalizers','training'):assert sha(Path(frozen[k]['path']))==frozen[k]['sha256']
assert not json.loads((OLD/'information_result.json').read_text())['passed']
t=dict(np.load(frozen['model']['path'],allow_pickle=False))
shapes=dict(left=(140,32),right=(140,32),edge=(3,32),bias=(32,),output=(32,3),feature_scale=(140,),rho_scale=(),third_scale=(),target_scale=(3,))
assert set(t)==set(shapes)
for k,v in t.items():assert v.dtype==np.float32 and v.shape==shapes[k] and np.isfinite(v).all()
packed={k:(list(v.shape),base64.b64encode(zlib.compress(v.tobytes())).decode()) for k,v in t.items()}
for k,(shape,blob) in packed.items():assert np.frombuffer(zlib.decompress(base64.b64decode(blob)),np.float32).reshape(shape).tobytes()==t[k].tobytes()
s=PARENT.read_text();tree=ast.parse(s);changed=s;inverse_pairs=[];proof=[]
edits={'_predict_learned_compact': [('                           final_hook=scalar_terminal_hook,', '                           final_hook=scalar_terminal_hook, joint_state_hook=(lambda: state),')], '_predict_binary': [('coefficient_hook=None, covariance_hook=None):', 'coefficient_hook=None, covariance_hook=None, joint_state_hook=None):'), ('            if l in _GN_COV_LAYERS:', '            joint_cov_delta = None\n            if _JM_ENABLED and 8 <= l <= 14:\n                if joint_state_hook is None:\n                    raise RuntimeError("Missing current causal joint-moment state")\n                joint_cov_delta = _jm_covariance_delta(l,mu,Cpre,K21,radial,w32,joint_state_hook(),aux)\n            if l in _GN_COV_LAYERS:'), ('            if hook is not None:', '            if joint_cov_delta is not None:\n                C = C + joint_cov_delta\n            del joint_cov_delta\n            if hook is not None:')]}
for name,pairs in edits.items():
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    old=ast.get_source_segment(s,node);new=old
    for a,b in pairs:assert new.count(a)==1,(name,a);new=new.replace(a,b,1)
    inv=new
    for a,b in reversed(pairs):assert inv.count(b)==1;inv=inv.replace(b,a,1)
    assert inv==old and changed.count(old)==1;changed=changed.replace(old,new,1)
    inverse_pairs.append((old,new));proof.append(dict(function=name,exact_inverse=True,edits=pairs))
mapping=dict(features='_jm_features',predict_planes='_jm_predict_planes',correction='_jm_correction_math')
class Rename(ast.NodeTransformer):
    def __init__(self,mapping):self.mapping=mapping
    def visit_FunctionDef(self,n):n.name=self.mapping.get(n.name,n.name);return self.generic_visit(n)
    def visit_Name(self,n):n.id=self.mapping.get(n.id,n.id);return n
head=(OLD/'head.py').read_text();helper=ast.unparse(Rename(mapping).visit(ast.parse(head)))
back=Rename({v:k for k,v in mapping.items()}).visit(ast.parse(helper))
assert ast.dump(back,include_attributes=False)==ast.dump(ast.parse(head),include_attributes=False)
extra='\n_JM_LITERAL='+repr(packed)+'\n_JM_PARAMETERS=None\n_JM_ENABLED=True\n'+helper+'''

def _jm_parameters():
    global _JM_PARAMETERS
    if _JM_PARAMETERS is None:
        _JM_PARAMETERS=_unpack_static_fp32(_JM_LITERAL)
    return _JM_PARAMETERS

def _jm_corrections(l,mu,C,K,radial,W,H):
    if int(H.shape[1]) != 64:
        raise RuntimeError("Frozen joint head requires H64")
    a,b,c=_jm_correction_math(fnp,flops.stats.norm.cdf,l,mu,C,K,radial,W,H,_jm_parameters())
    v=fnp.diag(C);s=fnp.sqrt(fnp.maximum(v,1e-10))
    return a,b*((v*s)[:,None]*s[None,:]),c
'''
extra += "\n"+(H/"delta_fragment.py").read_text()
candidate=changed+extra;ast.parse(candidate);inv=candidate[:-len(extra)]
for old,new in reversed(inverse_pairs):assert inv.count(new)==1;inv=inv.replace(new,old,1)
assert inv==s
with (H/'candidate.py').open('x') as out:out.write(candidate)
with (H/'disabled.py').open('x') as out:out.write(candidate+'\n_JM_ENABLED=False\n')
rec=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate_sha256=sha(H/'candidate.py'),disabled_sha256=sha(H/'disabled.py'),
         parent_sha256=sha(PARENT),model=frozen['model'],exact_source_inverse=True,helpers_ast_inverse=True,
         all_parameter_bytes_exact=True,proof=proof,old_information_gate_passed=False,
         explicit_separate_endpoint_followup=True,covariance_output_only=True)
with (H/'build.json').open('x') as out:json.dump(rec,out,indent=2);out.write('\n')
print(json.dumps(rec),flush=True)
