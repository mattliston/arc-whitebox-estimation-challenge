import ast
from common import *

verify(); guard('fit')
for e in read(HERE/'fit_seal.json')['sources']: assert pin(e['path'])==e
r=read(HERE/'fit_result.json'); f=packet(read(HERE/'features_frozen.json')['packet'])
t=packet(read(HERE/'targets.json')['packet']); m=packet(r['model'])
B=f['B']; E=f['fast_parent']-t['targets']; w=f['weights']; n=6*1024
# Independent scalar sums and eigensystem solution of the fixed trust ball.
G=np.array([[math.fsum(float(w[i])*float(B[i,j,k])*float(B[i,j,l])
    for i in range(6) for j in range(1024))/n for l in range(8)] for k in range(8)])
b=np.array([math.fsum(float(w[i])*float(B[i,j,k])*float(E[i,j])
    for i in range(6) for j in range(1024))/n for k in range(8)])
ridge=.001*float(np.trace(G))/8
eig,U=np.linalg.eigh(G+ridge*np.eye(8)); proj=U.T@b
def solve(mu): return -(U@(proj/(eig+mu)))
mu=0.; a=solve(mu)
if np.linalg.norm(a)>1:
    lo=0.; hi=max(float(np.linalg.norm(b)),float(np.max(eig)),1e-30)
    while np.linalg.norm(solve(hi))>1: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if np.linalg.norm(solve(mid))>1: lo=mid
        else: hi=mid
    mu=hi; a=solve(mu)
assert np.allclose(G,m['G'],rtol=1e-12,atol=1e-24)
assert np.allclose(b,m['b'],rtol=1e-12,atol=1e-24)
assert np.allclose(a,m['coefficients'],rtol=1e-10,atol=1e-12)
assert math.isclose(mu,float(m['multiplier']),rel_tol=1e-10,abs_tol=1e-24)
assert np.linalg.norm(a)<=1+1e-12 and np.max(np.abs(m['output_bias']))==0
assert pin(r['candidate']['path'])==r['candidate']
parent=(FIT/'candidate.py').read_text(); source=Path(r['candidate']['path']).read_text()
assert source.startswith(parent)
footer=source[len(parent):]; nodes=ast.parse(footer).body
assert len(nodes)==2
values={}
for node in nodes:
    assert isinstance(node,ast.Assign) and len(node.targets)==1
    target=node.targets[0]
    assert isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
    assert target.value.id=='_CF_PAYLOAD'
    values[ast.literal_eval(target.slice)]=ast.literal_eval(node.value)
assert set(values)=={'output.weight','output.bias'}
assert np.asarray(values['output.weight'],dtype='float32').tobytes()==m['readout'].tobytes()
assert m['readout'].tobytes()==(f['directions']@m['coefficients']).astype('float32')[None,:].tobytes()
assert values['output.bias']==[0.0]
assert '_SR_ON =' not in source and 'def _cr_dispatch(' not in source
once(HERE/'fit_verification.json',dict(passed=True,independent_scalar_gram=True,
    independent_eigensystem_fit=True,source_body_exact=True,only_readout_replaced=True,
    deployed_readout_exact=True,training_reference_deployed=False,candidate=r['candidate']))
print('FIXED_READOUT_INDEPENDENT_VERIFICATION_PASSED',flush=True)
