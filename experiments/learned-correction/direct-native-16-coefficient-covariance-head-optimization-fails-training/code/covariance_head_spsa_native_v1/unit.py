import ast,math
import numpy as np
from common import HERE,read,write,pin,seal_check,literal_source,build
from algorithm import schedule,gains,project,estimate,update

seal_check();rng=np.random.Generator(np.random.PCG64(55106))
m=rng.normal(size=(16,16));a=m.T@m+np.eye(16);b=rng.normal(size=16);x=rng.normal(size=16)
def loss(v):return float(.5*v@a@v+b@v)
h=np.ones((1,1))
for _ in range(4):h=np.block([[h,h],[h,-h]])
g=sum((estimate(loss(x+.02*d),loss(x-.02*d),.02,d) for d in h))/16
analytic=a@x+b;relative=float(np.linalg.norm(g-analytic)/np.linalg.norm(analytic));assert relative<1e-12
# Independent scalar-coordinate form for projection and clipped update.
base=np.linspace(-.2,.2,16);trial=base+np.arange(16)*.1
den=math.sqrt(sum(float(trial[j]-base[j])**2 for j in range(16)))
ref=np.array([base[j]+(trial[j]-base[j])*.75/den for j in range(16)])
assert np.allclose(project(trial,base),ref,rtol=0,atol=1e-16)
inside=base+.01;assert np.array_equal(project(inside,base),inside)
item=dict(a=.2,c=.125,delta=[1,-1]*8)
new,meta=update(base,base,2.,1.,item);expected=base-.2*np.asarray(item['delta'])/4
assert np.allclose(new,expected,rtol=0,atol=1e-16) and meta['gradient_norm']==16
rows=schedule();assert rows==schedule() and len(rows)==36
for cycle in range(6):assert sorted(r['rank'] for r in rows[6*cycle:6*cycle+6])==[36,38,40,41,43,44]
assert gains(0)==(.1,1/16) and all(abs(d)==1 for r in rows for d in r['delta'])
literal=literal_source()[2];head=np.asarray(literal['output'],np.float32)
(HERE/'variants').mkdir(exist_ok=False)
zero=build(HERE/'variants/unit_parent.py',head)
plus=build(HERE/'variants/unit_probe.py',head.astype(np.float64)+np.array(rows[0]['delta'])/16)
# Parse independent emitted source to certify deployment rounding.
node=next(n for n in ast.parse((HERE/'variants/unit_probe.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_EDGE_LITERAL' for t in n.targets))
actual=np.asarray(ast.literal_eval(node.value)['output'],np.float32)
assert actual.tobytes()==(head.astype(np.float64)+np.array(rows[0]['delta'])/16).astype(np.float32).tobytes()
write(HERE/'schedule.json',dict(seed=2026092066,rows=rows,base_head=head.tolist(),target_arrays_decoded=False))
write(HERE/'unit.json',dict(passed=True,quadratic_gradient_relative=relative,hadamard_directions=16,
    projection_clipping_schedule_rounding_passed=True,unchanged_parameter_checks=True,sources=[zero,plus],schedule=pin(HERE/'schedule.json'),target_arrays_decoded=False))
print(dict(unit_passed=True,quadratic_gradient_relative=relative),flush=True)
