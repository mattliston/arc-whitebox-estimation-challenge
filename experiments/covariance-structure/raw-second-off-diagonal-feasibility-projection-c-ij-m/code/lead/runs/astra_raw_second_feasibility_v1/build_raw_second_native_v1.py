"""Source-only fixed raw-second feasibility composer; no candidate imports or model calls."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

D = Path(__file__).resolve().parent
O = D.parents[2]
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PARENTS = {
    'current': (O/'engineer/a62_modules/s49/s56ubq1tw_x/candidate.py', 'c149e5800751892eb4d203341154ab2444e824a067ffaf791293a49e035ca0a3', 3220, 8101),
    'matchedC': (O/'scout_native/k4liab_s51k2tw/candidate.py', '6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41', 3214, 8095),
}
COMMON = '''# HELD fixed offdiagonal represented raw-second feasibility, L1..14, unit strength.
# Original learned dm/RC/KC compensation runs first; diagonal and means are retained.
_ASTRA_R2F_ON = True
_ASTRA_R2F_ACTIVE = [False]
_ASTRA_R2F_STATS = dict(layers=[], changed_entries=[], retired=True)

def _astra_r2f_project(layer, cov32, mean, delta):
    assert _ASTRA_R2F_ACTIVE[0] and 1 <= int(layer) <= 14
    n = int(mean.shape[0])
    assert tuple(mean.shape) == tuple(delta.shape) == (n,)
    assert tuple(cov32.shape) == (n,n)
    assert str(cov32.dtype) == 'float32' and str(mean.dtype) == str(delta.dtype) == 'float64'
    hold = _E2Hold()
    future_mean = hold(fnp.asarray(hold(mean + delta), dtype=fnp.float32))
    lower = hold(fnp.outer(hold(-future_mean), future_mean))
    changed = hold(fnp.less(cov32, lower))
    fnp.fill_diagonal(changed, False)
    answer = fnp.where(changed, lower, cov32)
    # Diagnostic uses paid native comparison/reduction; never host array access.
    count = int(hold(fnp.sum(changed)))
    _ASTRA_R2F_STATS['layers'].append(int(layer))
    _ASTRA_R2F_STATS['changed_entries'].append([int(layer), count])
    return answer

'''
TAIL = '''
_ASTRA_R2F_ACTIVE_CORE = _tr_clearing(_astra_r2f_core)

def _astra_r2f_dispatch(*args, **kwargs):
    global _ASTRA_R2F_STATS
    assert not _ASTRA_R2F_ACTIVE[0]
    _ASTRA_R2F_STATS = dict(layers=[], changed_entries=[], retired=False)
    if not _ASTRA_R2F_ON:
        try:
            return _ASTRA_R2F_PARENT_CORE(*args, **kwargs)
        finally:
            _ASTRA_R2F_STATS['retired'] = True
    _ASTRA_R2F_ACTIVE[0] = True
    try:
        answer = _ASTRA_R2F_ACTIVE_CORE(*args, **kwargs)
        assert _ASTRA_R2F_STATS['layers'] == list(range(1,15))
        assert [x[0] for x in _ASTRA_R2F_STATS['changed_entries']] == list(range(1,15))
        return answer
    finally:
        _ASTRA_R2F_ACTIVE[0] = False
        _ASTRA_R2F_STATS['retired'] = True

# Only the dynamic raw-core alias changes; all existing outer wrappers stay.
_UBC_PARENT_PREDICT = _astra_r2f_dispatch
'''

records = {}
for role, (parent, digest, core_line, tr_line) in PARENTS.items():
    assert H(parent) == digest
    src = parent.read_text(); lines = src.splitlines(True); tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_predict_binary' and n.lineno == core_line)
    core = ''.join(lines[node.lineno-1:node.end_lineno])
    edits = [
        ('def _predict_binary(', 'def _astra_r2f_core('),
        ('        depth, n = (int(mlp.depth), int(mlp.width))\n', '        depth, n = (int(mlp.depth), int(mlp.width))\n        assert (depth, n) == (16, 1024) and metric == \'covariance\' and _FULL_LOWER_MOMENTS and hook is not None\n'),
        ('                _ps_post_cov32 = C\n', '                if 1 <= l <= 14:\n                    C = _astra_r2f_project(l, C, m, dm)\n                _ps_post_cov32 = C\n'),
    ]
    changed = core
    for old,new in edits:
        assert changed.count(old) == 1, (role,old)
        changed = changed.replace(old,new,1)
    restored = changed
    for old,new in reversed(edits):
        assert restored.count(new) == 1
        restored = restored.replace(new,old,1)
    assert restored == core
    custody = f'''_ASTRA_R2F_PARENT_CORE = _UBC_PARENT_PREDICT
assert _ASTRA_R2F_PARENT_CORE.__qualname__ == '_tr_clearing.<locals>.w'
assert _ASTRA_R2F_PARENT_CORE.__code__.co_firstlineno == {tr_line}
assert _ASTRA_R2F_PARENT_CORE.__code__.co_freevars == ('f',)
assert _ASTRA_R2F_PARENT_CORE.__wrapped__ is _ASTRA_R2F_PARENT_CORE.__closure__[0].cell_contents
assert _ASTRA_R2F_PARENT_CORE.__wrapped__.__code__.co_firstlineno == {core_line}
assert nonlinear.__code__.co_firstlineno == 844
assert _nonlinear_pair_batch.__wrapped__ is nonlinear

'''
    fragment = COMMON + custody + changed + '\n' + TAIL
    paths = {}
    for kind,body in [('fragment',fragment),('ON',src+'\n\n'+fragment),('OFF',src+'\n\n'+fragment.replace('_ASTRA_R2F_ON = True','_ASTRA_R2F_ON = False',1))]:
        path = D/f'{role}_{kind}_UNVALIDATED_astra_v1.py'; compile(body,str(path),'exec')
        with path.open('x') as f: f.write(body)
        paths[str(path)] = H(path)
    records[role] = dict(parent_path=str(parent),parent_sha256=digest,core_span=[node.lineno,node.end_lineno],core_sha256=hashlib.sha256(core.encode()).hexdigest(),core_reverse_text_exact=True,only_numeric_insertion='C=_astra_r2f_project(l,C,m,dm) on L1..14 after original compensation and before post_cov32 capture',files=paths)
record = dict(status='HELD_SOURCE_ONLY_NOT_IMPORTED_OR_NUMERICALLY_QUALIFIED',utc=subprocess.check_output(['/bin/date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),builder_sha256=H(__file__),parents=records,scientific_scope='Fixed represented offdiagonal raw-second projection; original means and exact diagonal retained; target/radial/gq formulas unchanged',counter_scope='charged O(n*n) less+sum per layer; actual changed offdiagonal entries, including either direction separately',model_calls=0,request_creation=0)
out = D/'NATIVE_SOURCE_MANIFEST_astra_v1.json'
with out.open('x') as f: json.dump(record,f,indent=2); f.write('\n')
print(out,H(out))
