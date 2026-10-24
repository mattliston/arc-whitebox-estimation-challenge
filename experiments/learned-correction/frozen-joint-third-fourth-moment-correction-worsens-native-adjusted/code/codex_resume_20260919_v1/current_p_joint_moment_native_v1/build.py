"""Explicit new direct-objective study; the original proxy failure is preserved."""
import ast,base64,resource,zlib
from common import *
import numpy as np
verify()
assert not read(MODEL_DIR/'information_result.json')['passed']
assert read(MODEL_DIR/'replay_result.json')['passed']
f=read(MODEL_DIR/'model_frozen.json');check(f['model']);check(f['normalizers']);check(f['training'])
t=dict(np.load(f['model']['path'],allow_pickle=False));expected_shapes=dict(left=(140,32),right=(140,32),edge=(3,32),bias=(32,),output=(32,3),feature_scale=(140,),rho_scale=(),third_scale=(),target_scale=(3,))
assert set(t)==set(expected_shapes)
for k,v in t.items():assert v.shape==expected_shapes[k] and v.dtype==np.float32 and np.isfinite(v).all()
for k in ('feature_scale','rho_scale','third_scale','target_scale'):assert np.min(t[k])>0
packed={k:(list(v.shape),base64.b64encode(zlib.compress(v.tobytes(order='C'))).decode()) for k,v in t.items()}
for k,(shape,blob) in packed.items():assert np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype=np.float32).reshape(shape).tobytes()==t[k].tobytes()
source=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA;tree=ast.parse(source)
edits={
 '_predict_learned_compact':[(
   '                           final_hook=scalar_terminal_hook)',
   '                           final_hook=scalar_terminal_hook, joint_state_hook=(lambda: state))')],
 '_predict_binary':[(
   'final_hook=None, coefficient_hook=None):',
   'final_hook=None, coefficient_hook=None, joint_state_hook=None):'),(
   '            pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)',
   '            joint31 = None\n            if _JOINT_ENABLED and 8 <= l <= 14:\n                if joint_state_hook is None:\n                    raise RuntimeError("Missing causal joint-moment state")\n                joint21, joint31, joint22 = _joint_corrections(l,mu,Cpre,K21,radial,w32,joint_state_hook())\n                K21 = K21 + joint21\n                k22 = k22 + joint22\n                del joint21, joint22\n            pair_map = _nonlinear_first_gaussian if l == 0 else (_nonlinear_terminal_live if l == depth - 1 else nonlinear)'),(
   '            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)',
   '            if joint31 is None:\n                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n            else:\n                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric, joint31=joint31)\n            del joint31')],
 '_nonlinear_pair_batch':[(
   "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity'):",
   "def _nonlinear_pair_batch(mu, C, K21, k4, k22_override=None, metric='identity', joint31=None):"),(
   '    T31 = t4[:, None] * rho if k31 else None',
   '    T31 = t4[:, None] * rho if k31 else None\n    if joint31 is not None:\n        T31 = T31 + joint31')]
}
changed=source;proof=[];replacement=[]
for name,pairs in edits.items():
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1];old=ast.get_source_segment(source,node);new=old
    for a,b in pairs:assert new.count(a)==1,(name,a);new=new.replace(a,b,1)
    inverse=new
    for a,b in reversed(pairs):assert inverse.count(b)==1;inverse=inverse.replace(b,a,1)
    assert inverse==old and changed.count(old)==1;changed=changed.replace(old,new,1);replacement.append((old,new))
    proof.append(dict(function=name,exact_inverse=True,old_sha256=hashlib.sha256(old.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest(),edits=pairs))
head_source=(HERE/'head.py').read_text();head_tree=ast.parse(head_source)
mapping=dict(features='_joint_features',predict_planes='_joint_predict_planes',correction='_joint_correction_math')
class Rename(ast.NodeTransformer):
    def __init__(self,mapping):self.mapping=mapping
    def visit_FunctionDef(self,node):node.name=self.mapping.get(node.name,node.name);return self.generic_visit(node)
    def visit_Name(self,node):node.id=self.mapping.get(node.id,node.id);return node
renamed=Rename(mapping).visit(head_tree);helper=ast.unparse(renamed)
back=Rename({v:k for k,v in mapping.items()}).visit(ast.parse(helper))
assert ast.dump(back,include_attributes=False)==ast.dump(ast.parse(head_source),include_attributes=False)
extra='\n_JOINT_LITERAL='+repr(packed)+'\n_JOINT_PARAMETERS=None\n_JOINT_ENABLED=True\n'+helper+'''

def _joint_parameters():
    global _JOINT_PARAMETERS
    if _JOINT_PARAMETERS is None:
        _JOINT_PARAMETERS=_unpack_static_fp32(_JOINT_LITERAL)
    return _JOINT_PARAMETERS

def _joint_corrections(l,mu,C,K,radial,W,H):
    return _joint_correction_math(fnp,flops.stats.norm.cdf,l,mu,C,K,radial,W,H,_joint_parameters())
'''
candidate=changed+extra;ast.parse(candidate);inverse=candidate[:-len(extra)]
for old,new in reversed(replacement):assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
assert inverse==source
with (HERE/'candidate.py').open('x') as stream:stream.write(candidate)
with (HERE/'null_candidate.py').open('x') as stream:stream.write(candidate+'\n_JOINT_ENABLED=False\n')
write(HERE/'build.json',dict(completed=True,candidate=pin(HERE/'candidate.py'),parent=pin(SOURCE),model=f['model'],exact_source_inverse=True,
      helpers_ast_rename_inverse=True,all_parameter_bytes_exact=True,proof=proof,information_result=pin(MODEL_DIR/'information_result.json'), original_proxy_gate_passed=False,
      research_decision='Direct actual-objective test of the unchanged frozen model after a failed proxy gate.',
      null_candidate=pin(HERE/'null_candidate.py'),
      trainable_parameters=9184,extra_numerical_state='Only immutable parameters persist; recurrent state is the current parent closure.',
      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(built=True,candidate=pin(HERE/'candidate.py')),flush=True)
