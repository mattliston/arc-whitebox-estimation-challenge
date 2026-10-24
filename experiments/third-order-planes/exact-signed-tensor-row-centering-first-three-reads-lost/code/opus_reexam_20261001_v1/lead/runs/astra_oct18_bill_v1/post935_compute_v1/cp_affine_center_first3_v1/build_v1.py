import ast, hashlib, json
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[2]
parent = R / 'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
pin = '95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
base = parent.read_bytes()
assert hashlib.sha256(base).hexdigest() == pin
fragment = D / 'fragment_v1.py'
source = base + b'\n' + fragment.read_bytes()
out = D / 'candidate_95c_cpcenter_first3_HELD_v1.py'
compile(source, str(out), 'exec')
assert source.startswith(base)
out.open('xb').write(source)
receipt = dict(status='SOURCE_ONLY_HELD', parent=str(parent), parent_sha256=pin,
    candidate=str(out), sha256=hashlib.sha256(source).hexdigest(),
    fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),
    mechanism='A diag(lambda) B.T = (A-u) diag(lambda) B.T + u (B lambda).T; u is each unweighted A row mean',
    scope='Fixed L1-3, old_columns==0, active RTT prefix reader; unchanged kernel shape/dispatch/lambda placement; RTP and RE22 see restored plane; original tail inputs retained.',
    ownership='u, centered, columns, correction and final add are owned. Parent output held until correction add. Inputs unchanged; no arena retention.',
    static_first3_extra_flops_upper=42407936,
    static_bill_pct=100*42407936/238836882388,
    nominal_added_calls_per_read=9,
    limits=['Not byte-identical: centering, rank1 restoration and signed-lambda multiplication reorder F32 arithmetic.', 'No numerical or endpoint test. Accuracy recovery is a hypothesis; added requests need the retained runtime component.', 'No new setup caches, owner arrays or programs. F32 matvec compensation rounding is included in the candidate.'])
(D/'BUILD_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
