"""Fixed dense identities and rejection cases for the preserved rank384 fragment."""
import hashlib
import json
from pathlib import Path
import resource
import numpy as np

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE/'build.json').read_text())
assert sha(HERE/'candidate.py') == build['candidate_sha256']
assert sha(HERE/'kernel.py') == build['kernel_sha256']
OUT = HERE/'unit_v1'
OUT.mkdir(exist_ok=False)

def finite(value, name):
    if not np.isfinite(value).all():
        raise RuntimeError(name)
    return value

env = dict(fnp=np, _shared_old384_mm=lambda a,b: a@b, _shared_old384_finite=finite)
exec(compile((HERE/'kernel.py').read_bytes(), str(HERE/'kernel.py'), 'exec'), env)
assert env['_SLR_RANK'] == 384
assert env['_SLR_LAZY_LAYERS'] == (8,10,12,14)
assert env['_SLR_CONSOLIDATE_LAYERS'] == (9,11,13)
r = 384
I = np.eye(r, dtype=np.float32)
L = I/2 + np.roll(I, 1, axis=1)/8
N = 3*I/8 + I[np.arange(r)^1]/16
assert not np.array_equal(L@N, N@L)

def factors(columns):
    assert columns % 2 == 0
    rows = np.arange(r, dtype=np.float32)[:,None]
    cols = np.arange(columns//2, dtype=np.float32)[None,:]
    positive = (1 + (rows % 7)/16 + (cols+1)/128)/32
    return np.concatenate((positive, positive/2), axis=1)

def weights(columns):
    half = 1 + np.arange(columns//2, dtype=np.float32)/64
    return np.concatenate((half,-half))

U = (1 + ((np.arange(64)[:,None]+np.arange(r)[None,:])%11)/16).astype(np.float32)/64
V = (1 + ((2*np.arange(33)[:,None]+np.arange(r)[None,:])%13)/16).astype(np.float32)/64
checks = []

def compare(name, actual, expected):
    a = actual.astype(np.float64)
    ref = expected.astype(np.float64)
    norm = float(np.linalg.norm(ref))
    assert np.isfinite(a).all() and norm >= 2**-12
    absolute = float(np.max(np.abs(a-ref)))
    relative = float(np.linalg.norm(a-ref)/max(norm,2**-20))
    assert absolute <= 2e-5+5e-5*float(np.max(np.abs(ref)))
    assert relative <= 2e-4
    checks.append(dict(name=name, maxabs=absolute, relative_frobenius=relative, reference_norm=norm))

def consume(label, state, reference, lam):
    out = env['_slr_reconstruct'](U,state)
    compare(label+'_reconstruction',out,U.astype(np.float64)@reference)
    left = out[:19].copy()
    inner = env['_slr_inner'](left,lam,state)
    inner_ref = left.astype(np.float64)**2 @ (reference*lam.astype(np.float64)[None,:]).T
    compare(label+'_inner',inner,inner_ref)
    compare(label+'_plane',inner@V.T,inner_ref@V.astype(np.float64).T)

for layer in (8,14):
    F0,Fj = factors(64),factors(32)
    state = dict(columns=64,F=F0.copy(),lazy=None)
    new = env['_slr_join_factors'](layer,L,state,Fj)
    assert new is None
    state['columns'] = 96
    env['_slr_finish_factors'](state,new)
    reference = np.concatenate((L.astype(np.float64)@F0.astype(np.float64),Fj.astype(np.float64)),axis=1)
    lam = np.concatenate((weights(64),weights(32)))
    consume('lazy'+str(layer),state,reference,lam)
    if layer == 8:
        Fn = factors(16)
        materialized = env['_slr_join_factors'](9,N,state,Fn)
        state['columns'] = 112
        env['_slr_finish_factors'](state,materialized)
        assert state['lazy'] is None
        reference = np.concatenate((N.astype(np.float64)@reference,Fn.astype(np.float64)),axis=1)
        compare('consolidated_factors',state['F'],reference)
        consume('consolidated9',state,reference,np.concatenate((lam,weights(16))))
    else:
        consume('terminal15',state,reference,lam)

rejections = []
def rejected(name, function):
    try:
        function()
    except RuntimeError:
        rejections.append(name)
        return
    raise AssertionError('Expected rejection: '+name)

def lazy_state():
    s = dict(columns=64,F=factors(64),lazy=None)
    env['_slr_join_factors'](8,L,s,factors(32))
    s['columns'] = 96
    env['_slr_finish_factors'](s,None)
    return s

bad = lazy_state();bad['F'] = factors(96)
rejected('stale_materialized_owner',lambda:env['_slr_layout'](bad,96))
bad = lazy_state();bad['lazy']['joined']=30
rejected('wrong_column_roster',lambda:env['_slr_layout'](bad,96))
bad = lazy_state();bad['lazy']['map']=L.astype(np.float64)
rejected('wrong_dtype',lambda:env['_slr_layout'](bad,96))
bad = lazy_state()
rejected('missed_consolidation',lambda:env['_slr_join_factors'](11,N,bad,factors(16)))
rejected('chained_lazy_map',lambda:env['_slr_join_factors'](10,N,bad,factors(16)))
bad = lazy_state();bad['lazy']['base'][0,0] = np.inf
rejected('nonfinite_prefix',lambda:env['_slr_finish_factors'](bad,None))
bad = lazy_state();bad['lazy']['map']=L.copy();bad['lazy']['map'][0,0] = np.nan
rejected('nonfinite_map',lambda:env['_slr_finish_factors'](bad,None))
rejected('map_envelope',lambda:env['_slr_virtual_rotation_envelope'](np.full((r,r),2.**81,np.float32),np.full((r,2),2.**-40,np.float32)))
rejected('base_envelope',lambda:env['_slr_virtual_rotation_envelope'](np.full((r,r),2.**-40,np.float32),np.full((r,2),2.**81,np.float32)))
rejected('product_envelope',lambda:env['_slr_virtual_rotation_envelope'](np.full((r,r),2.**40,np.float32),np.full((r,2),2.**40,np.float32)))
result = dict(passed=True, comparisons=checks, rejections=rejections, truth_accessed=False,
              dense_only=True, rank=r, kernel_sha256=build['kernel_sha256'],
              candidate_sha256=build['candidate_sha256'], unit_sha256=sha(__file__),
              numpy_version=np.__version__, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (OUT/'result.json').open('x') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(dict(passed=True,comparisons=len(checks),rejections=len(rejections),
                     max_relative=max(x['relative_frobenius'] for x in checks),
                     peak_rss_bytes=result['peak_rss_bytes'])),flush=True)
