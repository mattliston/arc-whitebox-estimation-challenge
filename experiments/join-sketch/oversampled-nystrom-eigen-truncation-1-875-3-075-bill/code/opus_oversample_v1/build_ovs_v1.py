"""Oversampled Nystrom on 2b108+R256: r+p pivots (R256 recipe at r+p), QR(Y), small core M = Q[piv]^-1 R^T ((r+p)^2, = R W^-1 R^T),
eigh, keep top-r: Qn = Q V_r, core = diag(lambda_r). Append-last redefinition of the R256 join (its own text + 2 hunks). Writes once."""
from pathlib import Path
import hashlib,sys
D=Path(__file__).resolve().parent;R=D.parent
P=int(sys.argv[1])
base=(R/'opus_gbfix_v1/cand_2b108_frag_r256_v1.py').read_text();fr=(R/'opus_gbfix_v1/fragment_join_greedy_r256_2b108lineage_v1.py.txt').read_text()
assert hashlib.sha256(fr.encode()).hexdigest().startswith('0c795483') and base.endswith(fr) or fr in base
lines=fr.splitlines(keepends=True)
a=next(i for i,l in enumerate(lines) if l.startswith('def _astra_bcc_push_impl('))
b=next(i for i in range(a+1,len(lines)) if lines[i][:1] not in (' ','\n') )
fn=''.join(lines[a:b])
o1="_jn_rows, Y = _ojx_greedy(J, weight, Qp if old_columns else None, state['core'] if old_columns else None, _jn_d, rank, _e2_H)"
assert fn.count(o1)==1
fn=fn.replace(o1,"_jn_rows, Y = _ojx_greedy(J, weight, Qp if old_columns else None, state['core'] if old_columns else None, _jn_d, rank + _OVS_P, _e2_H)")
o2='''        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:'''
assert fn.count(o2)==1
fn=fn.replace(o2,'''        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _OVS_P and int(Y.shape[1]) > rank:
            Qn, core = _ovs_truncate(_e2_H, Qn, unused_R, _jn_rows, rank, _need_core)
        elif _need_core:''')
frag=f'''

# BEGIN OPUS-OVERSAMPLE FRAGMENT ovs_p{P}_v1 (append LAST after the R256 fragment): oversampled Nystrom + small-core eigen-truncation.
_OVS_P = {P}
_OVS_STATS = dict(joins=0)


def _ovs_truncate(h, Q, R, rows, r, need_core):
    M = h(fnp.linalg.solve(h(h(Q[rows, :]).astype(fnp.float32)), h(h(R.astype(fnp.float32)).T)).astype(fnp.float32))
    M = h(h(M + h(M.T)) * 0.5)
    ev, V = fnp.linalg.eigh(M)
    k = int(M.shape[0])
    Vr = h(V[:, k - 1:k - r - 1:-1]) if k - r - 1 >= 0 else h(V[:, ::-1])
    Qn = _shared_old384_mm(Q, Vr)
    core = None
    if need_core:
        core = h(fnp.diag(h(ev[k - 1:k - r - 1:-1] if k - r - 1 >= 0 else ev[::-1])).astype(fnp.float32))
    _OVS_STATS['joins'] += 1
    return Qn, core


{fn}# END OPUS-OVERSAMPLE FRAGMENT ovs_p{P}_v1
'''
o=D/f'cand_2b108_r256_ovs{P}_v1.py';o.open('x').write(base+frag);compile(o.read_text(),str(o),'exec');print('ovs',P,hashlib.sha256(o.read_bytes()).hexdigest()[:12])
