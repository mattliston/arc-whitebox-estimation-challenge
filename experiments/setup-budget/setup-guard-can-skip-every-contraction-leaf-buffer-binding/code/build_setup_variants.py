#!/usr/bin/env python3
"""TAG fable-porter (PT-65, lead 10:0x "S27r's setup hits the 3.5 s guard: the 22 k-split leaf buffers never bind; either bind them FIRST or trim the
prebind list to S27's count"): two setup-only variants of a pre-treuse candidate (predictions untouched; prebind is view binding):
  ksfirst  <out_frag> + <in> -> appends a _pb_bind wrapper (after k-split, before treuse; _pb_bind is not in treuse's _TR_CLEAR) that runs the v2b
           arena setup (copied verbatim from <in>'s first _pb_bind, assert-matched), binds the k-split leaf buffers (_KS_ACC_MN) first, then calls
           the lvfuse2 two-phase bind (k-split's _KS_OLD_PB_BIND; v2b then skips the existing arena)
  trim N   keeps the first N entries of the LIVE _PB_SINGLE table (the table is in the census rule's order, so these are the N highest-value keys)
usage: build_setup_variants.py ksfirst <in.py> <out.py> <frag.py> | build_setup_variants.py trim <N> <in.py> <out.py>"""
import sys, re
from pathlib import Path
mode = sys.argv[1]
if mode == 'ksfirst':
    src_p, out_p, frag_p = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    assert not out_p.exists() and not frag_p.exists()
    src = src_p.read_text()
    i = src.index('\ndef _pb_bind(t0):\n'); j = src.index('\ndef ', i + 5); first = src[i:j]
    a = first.index('    if _SCRATCH is None:'); b = first.index('    for kind, L, P, rows, cols in _PB_BATCHED:')
    arena = first[a:b]; assert arena.count('\n') == 4 and '_bind_push6(_SCRATCH[0])' in arena, arena
    assert src.count('\n_KS_OLD_PB_BIND = _pb_bind\n') == 1 and '_KS_ACC_MN = (' in src
    frag = ('\n\n# BEGIN FABLE-PORTER K-SPLIT LEAF BUFFERS FIRST (PT-65, lead 10:0x; setup order only, predictions unchanged): the arena setup (v2b,\n'
            '# verbatim), then the k-split leaf buffers, then the lvfuse2 two-phase prebind (k-split\'s _KS_OLD_PB_BIND), so the guard can only cut the\n'
            '# lowest-value list keys or their fusion, never the leaf buffers.\n'
            '_KSF_INNER = _KS_OLD_PB_BIND\n\n\n'
            'def _pb_bind(t0):\n    global _SCRATCH\n' + arena +
            '    done = 0\n    try:\n        for M, N in _KS_ACC_MN:\n            if _pb_time.monotonic() - t0 > _PB_WALL_S:\n                break\n'
            '            if (M, N) not in _KS_ACC:\n                _ks_acc(M, N)\n                done += 1\n'
            '    finally:\n        _PB_LOG.append((\'ksplit\', done, round(_pb_time.monotonic() - t0, 3)))\n        _BH_HOLD.clear()\n'
            '    _KSF_INNER(t0)\n# END FABLE-PORTER K-SPLIT LEAF BUFFERS FIRST\n')
    frag_p.write_text(frag); out = src.rstrip('\n') + '\n' + frag; compile(out, str(out_p), 'exec'); out_p.write_text(out)
    print(f'ksfirst: arena block of {arena.count(chr(10))} lines copied; fragment {len(frag)} bytes')
elif mode == 'trim':
    N, src_p, out_p = int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); assert not out_p.exists()
    src = src_p.read_text(); i = src.rindex('\n_PB_SINGLE = (') + 1; j0 = src.index('(', i); depth = 0
    for j in range(j0, len(src)):
        if src[j] == '(': depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0: break
    header = src[i:src.index('\n', i) + 1]
    keys = list(eval(re.sub(r'#[^\n]*', '', src[j0:j + 1]))); assert len(keys) > N, (len(keys), N)
    kept = keys[:N]; rows = [', '.join(repr(t) for t in kept[k:k + 8]) + ',' for k in range(0, N, 8)]
    table = header.rstrip('\n') + f'   PT-65: trimmed to the first {N} of {len(keys)}\n' + ''.join(f'    {r}\n' for r in rows) + ')'
    out = src[:i] + table + src[j + 1:]; compile(out, str(out_p), 'exec'); out_p.write_text(out)
    print(f'trim: {len(keys)} -> {N} keys; dropped {[k for k in keys[N:]]}')
