"""Source-only composer. Does not import candidates or evaluate models."""
import ast,hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[1]
parent=O/'engineer/a62_modules/s49/s56ubq1tw_x/candidate.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent)=='c149e5800751892eb4d203341154ab2444e824a067ffaf791293a49e035ca0a3'
source=parent.read_text();lines=source.splitlines(True);tree=ast.parse(source)
cores=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_predict_binary' and n.lineno<9083]
node=cores[-1];assert node.lineno==3220 and node.end_lineno==3517
raw=''.join(lines[node.lineno-1:node.end_lineno])
clone=raw.replace('def _predict_binary(', 'def _astra_raw21_core(',1)
anchor='                m = m + dm\n'
assert clone.count(anchor)==1
insert=anchor+'''                if 1 <= l <= 14:
                    target = _astra_raw21_target(l, target, host, dm, _gn_C0, C)
'''
clone=clone.replace(anchor,insert)
# Reverse the sole active numerical insertion and function rename.
assert clone.replace(insert,anchor).replace('def _astra_raw21_core(','def _predict_binary(',1)==raw
kernel_source=(D/'isolated_raw21_astra_v1.py').read_text();kt=ast.parse(kernel_source)
kn=next(x for x in kt.body if isinstance(x,ast.FunctionDef)and x.name=='raw21_actual_covariance')
kernel='\n'.join(kernel_source.splitlines()[kn.lineno-1:kn.end_lineno])+'\n'
kernel=kernel.replace('raw21_actual_covariance','_astra_raw21_actual_covariance')
header='''# HELD raw-R21 hypothesis. Exact current parent plus private raw-core clone.
# No model import/evaluation here. All existing outer compute wrappers remain.
_ASTRA_RAW21_ON = True
_ASTRA_RAW21_ACTIVE = [False]
_ASTRA_RAW21_STATS = dict(calls=0, layers=[], retired=True, owned_target_cache=False)
_ASTRA_RAW21_PARENT_CORE = _UBC_PARENT_PREDICT
assert _ASTRA_RAW21_PARENT_CORE.__name__ == '_predict_binary'
assert _ASTRA_RAW21_PARENT_CORE.__qualname__ == '_tr_clearing.<locals>.w'
assert _ASTRA_RAW21_PARENT_CORE.__code__.co_firstlineno == 8101
assert _ASTRA_RAW21_PARENT_CORE.__code__.co_freevars == ('f',)
assert _ASTRA_RAW21_PARENT_CORE.__wrapped__ is _ASTRA_RAW21_PARENT_CORE.__closure__[0].cell_contents
assert _ASTRA_RAW21_PARENT_CORE.__wrapped__.__code__.co_firstlineno == 3220

'''
tail='''
def _astra_raw21_target(layer, target, host, dm, covariance_before, covariance_after):
    assert _ASTRA_RAW21_ACTIVE[0] and 1 <= int(layer) <= 14
    n = int(host.shape[0])
    assert tuple(target.shape) == tuple(covariance_before.shape) == tuple(covariance_after.shape) == (n,n)
    assert tuple(dm.shape) == (n,)
    assert all(str(x.dtype) == 'float64' for x in (target,host,dm,covariance_before,covariance_after))
    answer = _astra_raw21_actual_covariance(fnp,target,host,dm,covariance_before,covariance_after)
    # The original plane has a float32 owner. Refresh both identity-keyed cache
    # fields together; no stale old target and no generic RQD retention.
    answer32 = fnp.asarray(answer,dtype=fnp.float32)
    _E2_LAYER['ps_target64'] = answer
    _E2_LAYER['ps_target32'] = answer32
    _E2_LAYER['_astra_raw21_owned_target'] = (answer,answer32)
    _ASTRA_RAW21_STATS['calls'] += 1
    _ASTRA_RAW21_STATS['layers'].append(int(layer))
    _ASTRA_RAW21_STATS['owned_target_cache'] = True
    return answer


_ASTRA_RAW21_ACTIVE_CORE = _tr_clearing(_astra_raw21_core)


def _astra_raw21_dispatch(*args, **kwargs):
    global _ASTRA_RAW21_STATS
    assert not _ASTRA_RAW21_ACTIVE[0], 'No concurrent or nested raw-R21 prediction'
    _ASTRA_RAW21_STATS = dict(calls=0,layers=[],retired=False,owned_target_cache=False)
    if not _ASTRA_RAW21_ON:
        try:
            return _ASTRA_RAW21_PARENT_CORE(*args,**kwargs)
        finally:
            _ASTRA_RAW21_STATS['retired'] = True
    _ASTRA_RAW21_ACTIVE[0] = True
    try:
        return _ASTRA_RAW21_ACTIVE_CORE(*args,**kwargs)
    finally:
        owned = _E2_LAYER.pop('_astra_raw21_owned_target',None)
        if owned is not None and _E2_LAYER.get('ps_target64') is owned[0]:
            _E2_LAYER.pop('ps_target64',None)
            _E2_LAYER.pop('ps_target32',None)
        _ASTRA_RAW21_ACTIVE[0] = False
        _ASTRA_RAW21_STATS['retired'] = True
        _ASTRA_RAW21_STATS['owned_target_cache'] = False


# Existing UBC wrapper resolves this alias dynamically. Other wrapper aliases,
# setup/prebind literals and compute arithmetic are not replaced.
_UBC_PARENT_PREDICT = _astra_raw21_dispatch
'''
fragment=header+kernel+'\n'+clone+tail
paths=[]
for name,data in [('fragment_raw21_UNVALIDATED_astra_v2.py',fragment),('native_ON_UNVALIDATED_astra_v2.py',source+'\n\n'+fragment),('native_OFF_UNVALIDATED_astra_v2.py',source+'\n\n'+fragment.replace('_ASTRA_RAW21_ON = True','_ASTRA_RAW21_ON = False',1))]:
 p=D/name;compile(data,str(p),'exec')
 with p.open('x')as f:f.write(data)
 paths.append(p)
record=dict(status='HELD_NATIVE_SOURCE_NOT_IMPORTED_OR_EVALUATED',utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),parent_path=str(parent),parent_sha256=sha(parent),core_span=[node.lineno,node.end_lineno],raw_core_sha256=hashlib.sha256(raw.encode()).hexdigest(),reverse_text_exact=True,hook='Only dynamic _UBC_PARENT_PREDICT alias replaced; outer UBC/YNT/UBX/UBX3/BCC/quarter wrappers retained',files={str(p):sha(p)for p in paths},builder_sha256=sha(__file__),kernel_sha256=sha(D/'isolated_raw21_astra_v1.py'),model_calls=0,whole_candidate_imports=0,requests_created=0)
p=D/'NATIVE_SOURCE_MANIFEST_astra_v2.json'
with p.open('x')as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(p),sha256=sha(p))))
