"""TAG fable-lead 2026-10-20: write a candidate whose learned-correction TERMINAL readout (head[15], head_b[15] of _GNN_LITERAL) is the refit of
fit_head15_v1.py (headfit/head15_fit.npz); every other parameter and all requests unchanged (bill identical). Writes <stem>.py (release form,
from the release base) and <stem>_sci_v1.py (from the science base). usage: build_head15_candidate_v1.py <release base> <science base> <fit npz> <stem> <note>"""
import re, sys, zlib, base64
from pathlib import Path
import numpy as np
rel, sci, fit, stem, note = Path(sys.argv[1]), Path(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4], sys.argv[5]
unpack = lambda sb: np.frombuffer(zlib.decompress(base64.b64decode(sb[1])), dtype=np.float32).reshape(sb[0]).copy()
pack = lambda a: (tuple(int(x) for x in a.shape), base64.b64encode(zlib.compress(np.ascontiguousarray(a, dtype=np.float32).tobytes())).decode())
def build(base, out, tail_marker):
    src = base.read_text(); L = src.split('\n')
    m = re.search(r"^_GNN_LITERAL = (\{.*\})$", src, re.M); lit = eval(m.group(1))
    head = unpack(lit['head']); head_b = unpack(lit['head_b'])
    assert head.shape[0] == 16 and head_b.shape[0] == 16, (head.shape, head_b.shape)
    assert np.allclose(head[15], fit['head_old'], atol=1e-6) and abs(float(head_b[15]) - float(fit['head_b_old'])) < 1e-6, 'the fit was made on a different head'
    head[15] = fit['head']; head_b[15] = float(fit['head_b'])
    lines = ['# [fable-lead 2026-10-20] TERMINAL READOUT REFIT: %s (fit_head15_v1.py on %d fresh/store nets; head[15], head_b[15] of _GNN_LITERAL replaced; bill and requests identical)' % (note, int(fit['n_nets'])),
             "_GNN_LITERAL = dict(_GNN_LITERAL, head=%r, head_b=%r)" % (pack(head), pack(head_b)), "assert _GNN_PARAMS is None"]
    if tail_marker:
        k = max(i for i, l in enumerate(L) if l.startswith(tail_marker)); L = L[:k] + lines + L[k:]
    else:
        L = L + lines
    s = '\n'.join(L); compile(s, str(out), 'exec'); out.write_text(s); print('written', out, len(L), 'lines')
build(sci, Path(stem + '_sci_v1.py'), '_FAST_RESCUE_CAP_S')
build(rel, Path(stem + '.py'), None)
