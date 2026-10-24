"""[opus-ntsched] join-SCHEDULE variants on S2tr 46e9315d with optional NT joins (build_nt_v2's patches + fragment), constants only.
Fragment (appended LAST) sets _JOIN_SCHEDULE {layer: (first cohort, last cohort)}, _R320_RULE (rank 384 at any new join layer),
_SLR_MATERIALIZE_LAYERS (every non-first, non-lazy join materializes; L13 stays lazy) and _NT_P / _NT_Q.
usage: build_ntsched_v1.py TAG 'SCHEDULE' 'NT_P' Q   e.g. w13n '{7:(0,2),10:(3,5),13:(6,9)}' '{13:128}' 0"""
import ast, hashlib, sys
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
tag, sched, ntp, q = sys.argv[1], ast.literal_eval(sys.argv[2]), ast.literal_eval(sys.argv[3]), int(sys.argv[4])
L = sorted(sched); assert L[-1] == 13 and all(isinstance(v, tuple) and len(v) == 2 for v in sched.values())
assert all(sched[a][1] + 1 == sched[b][0] for a, b in zip(L, L[1:])) and sched[L[0]][0] == 0, 'contiguous cohort windows'
assert all(c <= l - 1 for l in L for c in sched[l]), 'cohort born before its join'
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
fn = src.index('def _astra_bcc_push_impl(')


def patch(old, new):
    global src
    i = src.rindex(old); assert i > fn and src.index('\ndef ', fn + 10) > i, old[:60]
    src = src[:i] + new + src[i + len(old):]


patch("""        Omega = w32[previous, :rank]
        if _JOIN_NYSTROM:""", """        Omega = w32[previous, :rank]
        _nt_p = int(_NT_P.get(layer, 0)) if _JOIN_NYSTROM else 0   # opus-u64idea NT: extra static rows at this join
        if _JOIN_NYSTROM:""")
patch("""            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])
            Y = _shared_old384_mm(J,""", """            _jn_rows = (fnp.argsort(-_jn_d.astype(fnp.float64), kind='stable')[:rank + _nt_p] if _nt_p else
                        fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]))
            Y = _shared_old384_mm(J,""")
patch("""        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _need_core:""", """        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _nt_p:
            Qn, core = _nt_truncate(layer, rank, Qn, unused_R, _jn_rows, _need_core)
        elif _need_core:""")
nt = (D/'build_nt_v2.py').read_text()
frag = nt[nt.index("frag = '''") + len("frag = '''"):nt.index("''' % (P, q)")].replace('%%', '%')
frag = frag.replace('_NT_P = %r', '_NT_P = ' + repr(ntp)).replace('_NT_Q = %d', '_NT_Q = %d' % q)
frag = frag.replace('build_nt_v2.py)', 'build_ntsched_v1.py)')
first, mats = L[0], tuple(l for l in L[1:] if l != 13)
frag += '''

# BEGIN OPUS-NTSCHED SCHEDULE (append LAST; build_ntsched_v1.py)
_JOIN_SCHEDULE = %r
_R320_RULE = {l: 384 for l in _JOIN_SCHEDULE}
_SLR_MATERIALIZE_LAYERS = %r
# END OPUS-NTSCHED SCHEDULE
''' % (sched, mats)
o = D/f'cand_s2tr_{tag}_v1.py'
with o.open('x') as f:
    f.write(src + frag)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8], sched, ntp, q, 'materialize', mats)
