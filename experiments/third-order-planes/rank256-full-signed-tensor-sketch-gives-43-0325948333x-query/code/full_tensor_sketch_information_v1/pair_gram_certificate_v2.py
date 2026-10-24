"""Add the required cross-PAIR witness; v1 compared individual cube Grams."""
import hashlib,json,resource
from pathlib import Path
import numpy as np
from algebra import contract

HERE=Path(__file__).resolve().parent
assert json.loads((HERE/'unit.json').read_text())['passed']
x,y=np.eye(3)[:2]
p=np.stack([x,x,y,y],axis=1)
lam=np.array([1.,-1.,-1.,1.])
pair_tensors=[]
for k in range(2):
    xx,yy=p[:,k],p[:,k+2]
    pair_tensors.append(lam[k]*(np.einsum('i,j,k->ijk',xx,xx,xx)-np.einsum('i,j,k->ijk',yy,yy,yy)))
full=sum(pair_tensors)
assert np.array_equal(full,np.zeros((3,3,3)))
aa=np.repeat(np.eye(3),3,axis=1)
bb=np.tile(np.eye(3),(1,3))
z=contract(p,lam,aa,bb)
assert np.array_equal(z,np.zeros((3,9)))
full_gram=z@z.T
sum_pair_gram=sum(t.reshape(3,9)@t.reshape(3,9).T for t in pair_tensors)
assert np.array_equal(full_gram,np.zeros((3,3)))
assert np.array_equal(sum_pair_gram,np.diag([2.,2.,0.]))
record=dict(passed=True,full_tensor_gram=full_gram.tolist(),sum_per_pair_grams=sum_pair_gram.tolist(),
            exact_cross_pair_cancellation=True,old_unit_unchanged=True,
            clarification='unit.json reports sum of individual signed-cube Grams, not sum of complete-pair Grams. This additive exact witness satisfies the separate cross-pair requirement.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'pair_gram_certificate_v2.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
