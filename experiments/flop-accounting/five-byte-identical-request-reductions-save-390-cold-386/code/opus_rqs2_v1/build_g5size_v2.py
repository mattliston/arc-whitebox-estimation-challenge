"""[opus-lowres] _GSAFE5_LIMIT sizing builds v2 (v1 used the base without the stack constants): release's S1b stack preview
lead/runs/opus_passrel_v1/s1_v1/cand_s1b_stack_preview_v1.py bd238e1c (= S1b base 0368fc3a: LR2 -> strips_v3 -> fhmdiag_v2 -> RQX2 v1 ->
pairdiag_v1 -> misc_v1, + L15RC30 + DCV2 as stack_gate appends them) + a value-only line _GSAFE5_LIMIT = k appended LAST (k = 5 = as is)."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
t = (R/'opus_passrel_v1/s1_v1/cand_s1b_stack_preview_v1.py').read_text(); assert hashlib.sha256(t.encode()).hexdigest().startswith('bd238e1c')
for k in (5, 4, 3, 2):
    u = t if k == 5 else t + f'''

# BEGIN [opus-lowres] GSAFE5 LIMIT (value-only, append LAST): the first {k} eligible covariance sandwiches per predict take the full D6
# product (five-full had 5); the rest take the two-strip product
_GSAFE5_LIMIT = {k}
# END [opus-lowres] GSAFE5 LIMIT
'''
    o = D/f'cand_s1bp_g{k}_v2.py'
    with o.open('x') as f: f.write(u)
    compile(u, str(o), 'exec'); print(o.name, hashlib.sha256(u.encode()).hexdigest()[:8])
