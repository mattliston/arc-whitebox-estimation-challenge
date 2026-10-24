"""Source-only fixed nonnegative marginal variance composer; no candidate imports or model calls."""
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
COMMON = '# HELD fixed marginal-variance projection L1..14 after unchanged RC/KC.\n_ASTRA_NNV_ON = True\n_ASTRA_NNV_ACTIVE = [False]\n_ASTRA_NNV_STATS = dict(layers=[], changed_diagonal=[], retired=True)\n\n"""One fixed marginal-variance projection; synthetic/source-only until registered."""\ndef _astra_nnv_kernel(diagonal64, fnp):\n    assert len(diagonal64.shape) == 1 and str(diagonal64.dtype) == \'float64\'\n    assert bool(fnp.all(fnp.isfinite(diagonal64)))\n    answer = fnp.maximum(diagonal64, 0.0)\n    changed = int(fnp.sum(fnp.not_equal(answer, diagonal64)))\n    return answer, changed\n\ndef _astra_nnv_project(layer, diagonal64):\n    assert _ASTRA_NNV_ACTIVE[0] and 1 <= int(layer) <= 14\n    answer, count = _astra_nnv_kernel(diagonal64, fnp)\n    _ASTRA_NNV_STATS[\'layers\'].append(int(layer))\n    _ASTRA_NNV_STATS[\'changed_diagonal\'].append([int(layer), count])\n    return answer\n\n'
TAIL = '''
_ASTRA_NNV_ACTIVE_CORE = _tr_clearing(_astra_nnv_core)

def _astra_nnv_dispatch(*args, **kwargs):
    global _ASTRA_NNV_STATS
    assert not _ASTRA_NNV_ACTIVE[0]
    _ASTRA_NNV_STATS = dict(layers=[], changed_diagonal=[], retired=False)
    if not _ASTRA_NNV_ON:
        try:
            return _ASTRA_NNV_PARENT_CORE(*args, **kwargs)
        finally:
            _ASTRA_NNV_STATS['retired'] = True
    _ASTRA_NNV_ACTIVE[0] = True
    try:
        answer = _ASTRA_NNV_ACTIVE_CORE(*args, **kwargs)
        assert _ASTRA_NNV_STATS['layers'] == list(range(1,15))
        assert [x[0] for x in _ASTRA_NNV_STATS['changed_diagonal']] == list(range(1,15))
        return answer
    finally:
        _ASTRA_NNV_ACTIVE[0] = False
        _ASTRA_NNV_STATS['retired'] = True

# Only the dynamic raw-core alias changes; all existing outer wrappers stay.
_UBC_PARENT_PREDICT = _astra_nnv_dispatch
'''

records = {}
for role, (parent, digest, core_line, tr_line) in PARENTS.items():
    assert H(parent) == digest
    src = parent.read_text(); lines = src.splitlines(True); tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_predict_binary' and n.lineno == core_line)
    core = ''.join(lines[node.lineno-1:node.end_lineno])
    edits = [
        ('def _predict_binary(', 'def _astra_nnv_core('),
        ('        depth, n = (int(mlp.depth), int(mlp.width))\n', '        depth, n = (int(mlp.depth), int(mlp.width))\n        assert (depth, n) == (16, 1024) and metric == \'covariance\' and _FULL_LOWER_MOMENTS and hook is not None\n'),
        ('                _ps_post_cov32 = C\n', '                if 1 <= l <= 14:\n                    _ps_Cdiag = _astra_nnv_project(l, _ps_Cdiag)\n                _ps_post_cov32 = C\n'),
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
    custody = f'''_ASTRA_NNV_PARENT_CORE = _UBC_PARENT_PREDICT
assert _ASTRA_NNV_PARENT_CORE.__qualname__ == '_tr_clearing.<locals>.w'
assert _ASTRA_NNV_PARENT_CORE.__code__.co_firstlineno == {tr_line}
assert _ASTRA_NNV_PARENT_CORE.__code__.co_freevars == ('f',)
assert _ASTRA_NNV_PARENT_CORE.__wrapped__ is _ASTRA_NNV_PARENT_CORE.__closure__[0].cell_contents
assert _ASTRA_NNV_PARENT_CORE.__wrapped__.__code__.co_firstlineno == {core_line}
assert nonlinear.__code__.co_firstlineno == 844
assert _nonlinear_pair_batch.__wrapped__ is nonlinear

'''
    fragment = COMMON + custody + changed + '\n' + TAIL
    paths = {}
    for kind,body in [('fragment',fragment),('ON',src+'\n\n'+fragment),('OFF',src+'\n\n'+fragment.replace('_ASTRA_NNV_ON = True','_ASTRA_NNV_ON = False',1))]:
        path = D/f'{role}_{kind}_UNVALIDATED_astra_v1.py'; compile(body,str(path),'exec')
        with path.open('x') as f: f.write(body)
        paths[str(path)] = H(path)
    records[role] = dict(parent_path=str(parent),parent_sha256=digest,core_span=[node.lineno,node.end_lineno],core_sha256=hashlib.sha256(core.encode()).hexdigest(),core_reverse_text_exact=True,only_numeric_insertion='_ps_Cdiag=_astra_nnv_project(l,_ps_Cdiag) on L1..14 after original compensation and before unchanged paired diagonal writes',files=paths)
record = dict(status='HELD_SOURCE_ONLY_NOT_IMPORTED_OR_NUMERICALLY_QUALIFIED',utc=subprocess.check_output(['/bin/date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),builder_sha256=H(__file__),parents=records,scientific_scope='Fixed float64 marginal diagonal maximum with zero; original means/offdiagonal values and target/radial/gq formulas unchanged',counter_scope='charged finite assertion/maximum/not_equal/sum; changed diagonal values, including zero-count admissible',model_calls=0,request_creation=0)
out = D/'NATIVE_SOURCE_MANIFEST_astra_v1.json'
with out.open('x') as f: json.dump(record,f,indent=2); f.write('\n')
print(out,H(out))
