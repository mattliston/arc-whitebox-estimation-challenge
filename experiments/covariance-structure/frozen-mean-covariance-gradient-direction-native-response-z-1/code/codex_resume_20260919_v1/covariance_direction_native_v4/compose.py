"""Only source construction; exact current Lazy inverse, no numerical imports."""
import ast,hashlib
from common import SOURCE
PARENT_SHA='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
FRAGMENT='''
_CDN_PARAMETERS = None
_CDN_LAST = None
def _cdn_parameters():
    global _CDN_PARAMETERS, _CDN_LAST
    if _CDN_PARAMETERS is None:
        _CDN_PARAMETERS=fnp.asarray(_CDN_LITERAL,dtype=fnp.float32)
    _CDN_LAST=[]
    return _CDN_PARAMETERS
def _cdn_covariance(l,C,H,Q):
    if not 1<=l<=14 or C.shape!=(1024,1024) or H.shape!=(1024,64) or Q.shape!=(14,65):
        raise ValueError('Covariance direction dimensions')
    if str(C.dtype)!='float64' or str(H.dtype)!=str(Q.dtype) or str(Q.dtype)!='float32':
        raise ValueError('Covariance direction precisions')
    row=Q[l-1]
    z=H@row[:64]+row[64]
    if not bool(fnp.all(fnp.isfinite(z))) or not bool(fnp.all(fnp.abs(z)<=.0625)):
        raise ValueError('Fixed covariance direction trust envelope')
    scale=fnp.exp(.005*fnp.tanh(z)).astype(fnp.float64)
    updated=(C*scale[:,None])*scale[None,:]
    if not bool(fnp.all(fnp.isfinite(updated))):
        raise ValueError('Finite covariance direction write')
    _CDN_LAST.append(l)
    return updated
'''
def one(source,before,after):
    assert source.count(before)==1,before[:100]
    return source.replace(before,after)
def last(source,name):
    node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
def compose(Q):
    raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    original=raw.decode();prefix=last(original,'_predict_learned_compact');binary=last(original,'_predict_binary')
    p=one(prefix,'    _bdnh_A = _bdnh_parameters()\n',
          '    _bdnh_A = _bdnh_parameters()\n    _cdn_Q = _cdn_parameters()\n')
    before='    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook,\n                           final_hook=scalar_terminal_hook)\n'
    after=('    def covariance_hook(l,C):\n'
           '        if _bdnh_layer != l or _bdnh_pending is None:\n'
           '            raise RuntimeError("Current original H owner")\n'
           '        return _cdn_covariance(l,C,state,_cdn_Q)\n'
           '    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook,\n'
           '                           final_hook=scalar_terminal_hook,covariance_hook=covariance_hook)\n')
    p=one(p,before,after)
    b=one(binary,'final_hook=None, coefficient_hook=None):\n',
          'final_hook=None, coefficient_hook=None, covariance_hook=None):\n')
    b=one(b,'            fnp.fill_diagonal(C, cv)\n',
          '            fnp.fill_diagonal(C, cv)\n'
          '            if covariance_hook is not None and 1 <= l <= 14:\n'
          '                C = covariance_hook(l,C)\n')
    changed=one(one(original,prefix,p),binary,b)
    assert one(one(changed,b,binary),p,prefix)==original
    final=changed+'\n\n_CDN_LITERAL = '+repr(Q)+'\n'+FRAGMENT
    ast.parse(final)
    return final,dict(parent_sha256=PARENT_SHA,full_source_inverse=True,
                      physical_function_inverse=True,only_C_write=True,mean_model_frozen=True,
                      inserted_after_mask_before_Bpost=True,original_newborn_and_star_repair=True)
