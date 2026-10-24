"""Build held native late-KC timing hypothesis from exact matched C. No imports/evals."""
import ast, hashlib, json, subprocess
from pathlib import Path
D=Path(__file__).resolve().parent; O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=O/'scout_native/k4liab_s51k2tw/candidate.py'
assert sha(P)=='6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41'
s=P.read_text(); nodes=ast.parse(s).body
core=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary' and n.lineno==3214)
old=ast.get_source_segment(s,core); new=old.replace('def _predict_binary(', 'def _astra_lkc_core(',1)
edits=[]
def edit(before,after):
    global new
    assert new.count(before)==1,before
    new=new.replace(before,after,1);edits.append((before,after))
edit('        for l in range(depth):\n','        for l in range(depth):\n            _astra_lkc_pending = None\n')
anchor="                _gn_rc = _GN_RC.get(l, 0.0) - _GN_KC.get(l, 0.0)"
edit(anchor,"                _astra_lkc_raw32, _astra_lkc_rawdiag = C, _ps_Cdiag\n"+anchor)
anchor='                m = m + dm\n'
early='''                # Preserve the original rounded outgoing covariance. Only the
                # within-layer consumers temporarily see the RC-only version.
                if 1 <= l <= 14 and _GN_KC.get(l, 0.0):
                    assert _FULL_LOWER_MOMENTS
                    _astra_lkc_pending = (C, _ps_post_cov32)
                    _astra_lkc_rc = _GN_RC.get(l, 0.0)
                    if _astra_lkc_rc:
                        _astra_lkc_diag = _astra_lkc_rawdiag - _astra_lkc_rc * (_ps_Cdiag0 - _astra_lkc_rawdiag)
                        _astra_lkc_early32 = _astra_lkc_raw32 - _ps32(_astra_lkc_rc) * (_ps_C0 - _astra_lkc_raw32)
                    else:
                        _astra_lkc_diag = _astra_lkc_rawdiag
                        # Copy because filling the diagonal must not mutate a
                        # saved outgoing owner in a degenerate dial case.
                        _astra_lkc_early32 = _astra_lkc_raw32.copy()
                    C = _astra_lkc_early32.astype(fnp.float64)
                    fnp.fill_diagonal(C, _astra_lkc_diag)
                    fnp.fill_diagonal(_astra_lkc_early32, _astra_lkc_diag)
                    _E2_LAYER['post_cov32'] = (C, _astra_lkc_early32)
                    _ASTRA_LKC_STATS['early_layers'].append(int(l))
'''
edit(anchor,early+anchor)
anchor='            fnp.fill_diagonal(star, _ps_star_diag)\n'
late='''            # No birth, F4, star or target consumer remains in this layer.
            # Restore both exact original owners before the next covariance
            # push. _ps32 consults this pair before its generic RQD cache.
            if _astra_lkc_pending is not None:
                C, _ps_post_cov32 = _astra_lkc_pending
                _E2_LAYER['post_cov32'] = (C, _ps_post_cov32)
                _astra_lkc_pending = None
                _ASTRA_LKC_STATS['late_layers'].append(int(l))
'''
edit(anchor,anchor+late)
reverse=new
for a,b in reversed(edits):
    assert reverse.count(b)==1
    reverse=reverse.replace(b,a,1)
reverse=reverse.replace('def _astra_lkc_core(', 'def _predict_binary(',1)
assert ast.dump(ast.parse(reverse),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
header='''# HELD new native-defined late-KC timing hypothesis, no numerical qualification.
# The old rounded outgoing covariance is preserved for each same-input step.
# Birth/F4/star consumers receive RC-only C; no new fitted parameter.
_ASTRA_LKC_ON = True
_ASTRA_LKC_ACTIVE = [False]
_ASTRA_LKC_STATS = dict(early_layers=[], late_layers=[], retired=True)
_ASTRA_LKC_PARENT_CORE = _UBC_PARENT_PREDICT
assert _ASTRA_LKC_PARENT_CORE.__name__ == '_predict_binary'
assert _ASTRA_LKC_PARENT_CORE.__code__.co_freevars == ('f',)
assert _ASTRA_LKC_PARENT_CORE.__wrapped__ is _ASTRA_LKC_PARENT_CORE.__closure__[0].cell_contents
assert _ASTRA_LKC_PARENT_CORE.__wrapped__.__code__.co_firstlineno == 3214
'''
tail='''
_ASTRA_LKC_ACTIVE_CORE = _tr_clearing(_astra_lkc_core)
def _astra_lkc_dispatch(*args, **kwargs):
    global _ASTRA_LKC_STATS
    if not _ASTRA_LKC_ON:
        return _ASTRA_LKC_PARENT_CORE(*args, **kwargs)
    assert not _ASTRA_LKC_ACTIVE[0], 'No nested late-KC prediction'
    _ASTRA_LKC_ACTIVE[0] = True
    _ASTRA_LKC_STATS = dict(early_layers=[], late_layers=[], retired=False)
    try:
        return _ASTRA_LKC_ACTIVE_CORE(*args, **kwargs)
    finally:
        _ASTRA_LKC_ACTIVE[0] = False
        _ASTRA_LKC_STATS['retired'] = True
_UBC_PARENT_PREDICT = _astra_lkc_dispatch
'''
fragment=header+'\n'+new+'\n'+tail
q=D/'fragment_native_late_kc_UNVALIDATED_v1.py'
with q.open('x') as f:f.write(fragment)
compile(fragment,str(q),'exec',dont_inherit=True)
candidates={}
for role,on in [('OFF',False),('ON',True)]:
    suffix=fragment if on else fragment.replace('_ASTRA_LKC_ON = True','_ASTRA_LKC_ON = False',1)
    p=D/f'native_matched_C_late_kc_{role}_UNVALIDATED_v1.py'
    with p.open('x') as f:f.write(s+'\n\n'+suffix)
    compile(p.read_text(),str(p),'exec',dont_inherit=True)
    candidates[role]=dict(path=str(p),sha256=sha(p),mode=on)
r=dict(status='HELD_SOURCE_ONLY_NATIVE_LATE_KC_HYPOTHESIS',utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),parent=dict(path=str(P),sha256=sha(P)),fragment=dict(path=str(q),sha256=sha(q)),builder_sha256=sha(__file__),candidates=candidates,inverse_core_AST_exact=True,edit_sites=4,physics='RC-only C through all within-layer moment consumers; original rounded RC+KC C restored after star',full_imports=0,models=0,requests=0,source_only=True)
out=D/'NATIVE_LATE_KC_SOURCE_MANIFEST_HELD_v1.json'
with out.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),candidates=candidates)))
