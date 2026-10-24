import ast,hashlib
from common import *
def observed():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA
    n=[x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary'][-1]
    old=ast.get_source_segment(s,n);new=old
    edits=[
      ('                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)',
       '                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n                _CHQ_TAP.update(P=pp,lam=lam,A=star,C=C,W=w32,b=bpre)'),
      ('                m = m + final_hook(l, m, aux, W[l])',
       '                m = m + final_hook(l, m, aux, W[l])\n                _CHQ_TAP.update(k3=k3,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"])')]
    for a,b in edits:assert new.count(a)==1;new=new.replace(a,b,1)
    back=new
    for a,b in reversed(edits):assert back.count(b)==1;back=back.replace(b,a,1)
    assert back==old;ast.parse(new)
    return new,dict(function='_predict_binary',exact_inverse=True,original_sha256=hashlib.sha256(old.encode()).hexdigest(),observed_sha256=hashlib.sha256(new.encode()).hexdigest())
