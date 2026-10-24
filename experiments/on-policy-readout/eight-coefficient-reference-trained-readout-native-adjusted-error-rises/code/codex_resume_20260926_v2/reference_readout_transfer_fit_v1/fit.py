import ast
from common import *
verify();guard('assemble');guard('targets')
for e in read(HERE/'fit_seal.json')['sources']:assert pin(e['path'])==e
f=packet(read(HERE/'features_frozen.json')['packet']);t=packet(read(HERE/'targets.json')['packet'])
assert tuple(f['train'])==TRAIN and tuple(t['train'])==TRAIN
B=f['B'];E=f['fast_parent']-t['targets'];w=f['weights'];n=6*1024
G=np.einsum('ind,ine,i->de',B,B,w,optimize=True)/n
b=np.einsum('ind,in,i->d',B,E,w,optimize=True)/n
G=(G+G.T)*.5;eigenvalues=np.linalg.eigvalsh(G)
assert eigenvalues[-1]>0 and eigenvalues[0]>=-1e-10*eigenvalues[-1]
ridge=.001*float(np.trace(G))/8;A=G+ridge*np.eye(8)
a=np.linalg.solve(A,-b);mu=0.
if np.linalg.norm(a)>1:
    lo=0.;hi=max(float(np.linalg.norm(b)),float(np.linalg.norm(A,ord=2)),1e-30)
    while np.linalg.norm(np.linalg.solve(A+hi*np.eye(8),-b))>1:hi*=2
    for _ in range(100):
        mid=(lo+hi)*.5
        if np.linalg.norm(np.linalg.solve(A+mid*np.eye(8),-b))>1:lo=mid
        else:hi=mid
    mu=hi;a=np.linalg.solve(A+mu*np.eye(8),-b)
norm=float(np.linalg.norm(a));stationarity=float(np.linalg.norm((A+mu*np.eye(8))@a+b)/max(np.linalg.norm(b),1e-30))
assert np.isfinite(a).all() and norm<=1+1e-12 and mu>=0 and stationarity<=1e-10
assert mu*abs(norm*norm-1)<=1e-10*max(mu,1e-30)
# Scalar residual arithmetic independently reproduces the quadratic objective.
linear=E+np.einsum('ind,d->in',B,a)
sq=lambda x:math.fsum(float(v)**2 for v in x.ravel())
scalar=sum(float(w[i])*sq(linear[i]) for i in range(6))/n+ridge*sq(a)
c=sum(float(w[i])*sq(E[i]) for i in range(6))/n
quadratic=c+2*float(b@a)+float(a@G@a)+ridge*float(a@a)
assert math.isclose(scalar,quadratic,rel_tol=1e-12,abs_tol=1e-24)
assert scalar<=c+1e-20
readout=(f['directions']@a).astype('float32')[None,:]
assert readout.shape==(1,32) and np.isfinite(readout).all()
model=HERE/'readout.npz'
with model.open('xb') as handle:np.savez(handle,coefficients=a,readout=readout,output_bias=np.zeros(1,dtype='float32'),G=G,b=b,ridge=np.array(ridge),multiplier=np.array(mu))
source=FIT/'candidate.py';assert sha(source)=='042ed153ecbf9f11095326c7aaecc95ef0b996d45ea35b4bfd04a201bc05ba49'
original=source.read_text();assert '_SR_ON =' not in original and 'def _cr_dispatch(' not in original
footer='\n\n# Fixed eight-coefficient final-objective readout proposal.\n_CF_PAYLOAD[\'output.weight\'] = '+repr(readout.tolist())+'\n_CF_PAYLOAD[\'output.bias\'] = [0.0]\n'
candidate=HERE/'candidate.py';text=original+footer;assert text[:-len(footer)]==original;ast.parse(text)
with candidate.open('x') as handle:handle.write(text)
raw_ratios=[float(np.mean(linear[i]**2)/np.mean(E[i]**2)) for i in range(6)]
result=dict(coefficients=a.tolist(),coefficient_norm=norm,ridge=ridge,lagrange_multiplier=mu,stationarity_relative=stationarity,
    quadratic_objective=quadratic,scalar_objective=scalar,weighted_linear_ratio=float(np.sum(w[:,None]*linear**2)/np.sum(w[:,None]*E**2)),
    per_network_linear_raw_ratios=raw_ratios,gram_eigenvalues=eigenvalues.tolist(),all_directions_retained=True,
    fitted_parameters=8,output_bias_zero=True,encoder_frozen=True,fast_source_body_exact=True,
    training_reference_deployed=False,model=pin(model),candidate=pin(candidate),features=pin(HERE/'features_frozen.json'),targets=pin(HERE/'targets.json'),
    native_transfer_measured=False,check_targets_read=False)
once(HERE/'fit_result.json',result)
print(json.dumps(dict(event='fixed_readout_fitted',coefficients=a.tolist(),coefficient_norm=norm,linear_ratio=result['weighted_linear_ratio'],candidate=result['candidate'],stationarity_relative=stationarity)),flush=True)
