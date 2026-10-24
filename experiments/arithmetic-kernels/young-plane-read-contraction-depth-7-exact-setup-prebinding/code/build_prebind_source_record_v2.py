from pathlib import Path
import json,hashlib
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_young_read7_v1'
s='''# YR7 v2: exact same prediction computation; prebind fixed arena views in setup.
_YR7_PARENT_BIND = _pb_bind
_YR7_BIND_STATS = dict(keys=0, retired=False)
def _yr7_prebind_views():
    keys = set()
    for m, k, n, rank in _YR7_SHAPES:
        M, N, K = ((v + 127) // 128 * 128 for v in (m, n, k))
        blocks = K // 128
        count = (blocks + 15) // 16
        q, extra = divmod(blocks, count)
        sizes = ([128 * (q + 1)] * extra + [128 * q] * (count - extra))
        if rank:
            sizes.append((rank + 127) // 128 * 128)
        keys.add((2, 7, M, N))
        for width in sizes:
            keys.add((0, 7, M, width))
            keys.add((1, 7, width, N))
    saved = _r7_enter()
    try:
        for kind, L, rows, cols in sorted(keys):
            side = _ipt_side(kind, L, rows, cols)
            if kind < 2 and side[2][1] is None:
                _l3_quad(side[2], kind, L)
        _r7_acc()
        _YR7_BIND_STATS.update(keys=len(keys), retired=True)
    finally:
        _r7_leave(saved)

def _pb_bind(t0):
    _YR7_PARENT_BIND(t0)
    if _YR7_ON:
        _yr7_prebind_views()
        _PB_LOG.append(('yr7_fixed_views', _YR7_BIND_STATS['keys'],
                        round(_pb_time.monotonic() - t0, 3)))
'''
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,text in [('fragment_prebind_HELD_v2.py',s),('candidate_ON_HELD_v2.py',(D/'candidate_ON_HELD_v1.py').read_text()+'\n\n'+s),('candidate_OFF_HELD_v2.py',(D/'candidate_OFF_HELD_v1.py').read_text()+'\n\n'+s)]:
 p=D/name;assert not p.exists();p.write_text(text);compile(text,str(p),'exec');print(name,sha(p))
