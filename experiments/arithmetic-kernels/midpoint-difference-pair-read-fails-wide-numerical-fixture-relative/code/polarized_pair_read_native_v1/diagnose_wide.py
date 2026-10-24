"""Localize the failed fixed wide fixture; no candidate or gate change."""
import resource
from common import *
text=(HERE/'unit_v2.py').read_text();marker='for m,n,pp,eps in ((7,9';assert text.count(marker)==1
ns={'__name__':'wide_reference_diagnostic'}
exec(compile(text[:text.index(marker)],str(HERE/'unit_v2.py')+':definitions','exec'),ns)
np=ns['np'];flops=ns['flops'];fnp=ns['fnp'];mod=ns['mod']
for m,n,pp,eps in ((7,9,[3],.7),(17,11,[2,5],.1),(9,5,[4,3],2**-12),(7,7,[3],0.)):
    ns['check'](*ns['fixture'](m,n,pp,eps))
z=ns['fixture'](5,7,[3,2],.2);z[2][:]=0;ns['check'](*z)
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    arena=fnp.empty((mod._PACKED_ELEMENTS,),dtype=fnp.float32)
    mod._SCRATCH=[arena,mod._cp3_bind_arena_views(arena,mod._PACKED_PUSH),mod._cp3_bind_arena_views(arena,mod._PACKED_READ),None,None,None,None]
    mod._bind_push6(arena)
left,right,lam,cohorts=ns['fixture'](512,512,[512,512],2**-10)
before=[x.tobytes() for x in (left,right,lam)]
ref=ns['direct64'](left,right,lam)
L=left.astype(np.float64);R=right.astype(np.float64);A=lam.astype(np.float64)
stable=np.zeros_like(ref)
for start,width in cohorts:
    p=width//2;x,y=L[:,start:start+p],L[:,start+p:start+width];u,v=R[:,start:start+p],R[:,start+p:start+width]
    h,d=(x+y)/2,x-y;H,D=(u+v)/2,u-v;a=A[start:start+p]
    stable += (h*h+d*d/4)@(D*a).T + (2*h*d)@(H*a).T
identity=relative(stable,ref);assert identity<=3e-11
with flops.BudgetContext(flop_budget=2**41,quiet=True):args=[fnp.asarray(x) for x in (left,right,lam)]
results={}
for name,polar,product in [('polarized_existing_kernel',True,mod._mm),('original_existing_kernel',False,mod._mm),
                           ('polarized_direct_matmul_diagnostic_only',True,fnp.matmul),('original_direct_matmul_diagnostic_only',False,fnp.matmul)]:
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        value=(mod._polar_pair_product(*args,cohorts,0,len(lam),product) if polar else product(args[0]*args[0],(args[1]*args[2]).T))
    v=np.asarray(value);assert np.isfinite(v).all()
    results[name]=dict(relative=relative(v,ref),maxabs=float(np.max(np.abs(v-ref))),flops=int(ctx.summary_dict()['flops_used']))
assert results['polarized_existing_kernel']['relative']>3e-5
assert before==[x.tobytes() for x in (left,right,lam)]
write(HERE/'wide_diagnostic.json',dict(completed=True,fixed_fixture=[512,2048,512],small_fixtures=ns['rows'],
    fp64_pair_identity_relative=identity,results=results,
    candidate_unchanged=True,original_wide_gate_failed=True,direct_products_are_diagnostics_not_candidates=True,
    gates_unchanged=True,full_native_calls=0,targets_read=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    candidate=pin(HERE/'candidate.py'),unit_failure=pin(HERE/'unit_v2.log')))
print(dict(completed=True,fp64_pair_identity_relative=identity,results=results),flush=True)
