from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, sys, time
import numpy as np
from scipy.stats import t as student_t
from control import reference, coupled

D = Path(__file__).resolve().parent
rank = int(sys.argv[1]); assert rank in (35,100)
pins = json.loads((D/'seal.json').read_text())
for p,h in pins['sources'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
assert json.loads((D/'unit.json').read_text())['passed']
wp = Path(pins['weights'][str(rank)]['path'])
assert hashlib.sha256(wp.read_bytes()).hexdigest()==pins['weights'][str(rank)]['sha256']
out = D/f'rank{rank:04d}'; out.mkdir()
start = time.monotonic()
w = np.load(wp)
ref, reports, _ = reference(w, progress=lambda l,d,t:print(json.dumps(dict(event='reference',rank=rank,layer=l,min_d=d,min_tail=t)),flush=True))
np.savez(out/'reference.npz',**ref)
with (out/'reference_qualification.json').open('x') as f:
    json.dump(dict(passed=True,reports=reports,host_float64=True,source_sha256=pins['sources'][str(D/'control.py')]),f,indent=2)
setup_s = time.monotonic()-start
rng = np.random.default_rng(2709270000+rank)
keys = ('plain','antithetic','ordinary_residual','antithetic_residual','linear_residual')
sums=[]; sums2=[]; blockvars=[]
for b in range(32):
    X = rng.standard_normal((128,1024))
    r,_ = coupled(w,ref,X)
    fields = np.stack([r[k] for k in keys])
    assert fields.shape==(5,128,1024) and np.isfinite(fields).all()
    sums.append(np.sum(fields,axis=1)); sums2.append(np.sum(fields*fields,axis=1))
    blockvars.append(np.mean(np.var(fields,axis=1,ddof=1),axis=1))
    if b==0: np.savez(out/'witness.npz',X=X[:8],**{k:r[k][:8] for k in keys})
    if b%4==3: print(json.dumps(dict(event='sample',rank=rank,pairs=128*(b+1),elapsed_s=time.monotonic()-start)),flush=True)
sums=np.stack(sums); sums2=np.stack(sums2); blockvars=np.stack(blockvars)
np.savez(out/'sufficient_statistics.npz',sums=sums,sums2=sums2,blockvars=blockvars,keys=np.asarray(keys),block_size=128)
v=(np.sum(sums2,axis=0)-np.sum(sums,axis=0)**2/4096)/4095
pooled=np.mean(v,axis=1)
intervals={}
for i,k in enumerate(keys):
    bv=blockvars[:,i]; margin=float(student_t.ppf(.975,31)*np.std(bv,ddof=1)/math.sqrt(32))
    intervals[k]=[float(np.mean(bv)-margin),float(np.mean(bv)+margin)]
circuits={}
for k,mvs in (('ordinary_residual',46),('antithetic_residual',61)):
    c=2*1024**2*mvs; ix=keys.index(k); yard=float(pooled[ix]*c/2**41)
    circuits[k]=dict(matvecs=mvs,leading_flops=c,variance=float(pooled[ix]),
        optimistic_adjusted_variance_yardstick=yard,threshold_2e9=2e-9*2**41/c,
        threshold_1e9=1e-9*2**41/c,passes_variance_screen=bool(yard<=2e-9),
        empirical_block_interval=intervals[k])
rec=dict(rank=rank,pairs=4096,blocks=32,block_size=128,seed=2709270000+rank,
         numpy=np.__version__,setup_seconds=setup_s,elapsed_seconds=time.monotonic()-start,
         mean_per_output_variances=dict(zip(keys,map(float,pooled))),block_intervals=intervals,
         circuits=circuits,reference_anchor='Gaussian DAG m; not S31 or true mean',
         new_truth_arrays=False,native_bill_measured=False,utc_finished=datetime.now(timezone.utc).isoformat())
with (out/'result.json').open('x') as f: json.dump(rec,f,indent=2,allow_nan=False)
print(json.dumps(rec),flush=True)
