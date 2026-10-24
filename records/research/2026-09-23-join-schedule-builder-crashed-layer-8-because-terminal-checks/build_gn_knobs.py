#!/usr/bin/env python3
"""TAG fable-porter (PT-7): covhunter's zero-bill covariance-channel knobs (fable_covhunter_v1/gn2_knobs.py, theta v3) as a native stack
layer for the bank and engineer2's e2 v7/v8 forms, with or without build_physics.py. Edits are AST-located; disabled knobs run the original
statements.
  iso, k21n, k31 (default L2-14; pair-map COVARIANCE output only): right after the live pair-map call and BEFORE the omitted-radial source
    (covhunter's cov_pm hook point, so the source denominator sees the new C):  C = C + _gn_dcov(...), covhunter's certified closed form
      dcov = ss^T (X + X^T), zero diagonal, X = dT21 * t21a / 2 + dT31 * t31a / 6   (t21b = t21a^T and t31b = t31a^T because rho is symmetric)
      dK21 = iso kink_a k3_a Cpre_ac / v_a + k21n (1 - kink_a) K21_ac (off-diagonal; kink = |alpha| < 1.5), dT21 = dK21 / (v_a s_c),
      dT31 = k31 T31_row (the row-factored plane t4 r_a rho of the physics patch; t4 rho without it),
      t21a = A2 (x) A1 + rho A3 (x) A2 + rho^2/2 A4 (x) A3,  t31a = A3 (x) A1 + rho A4 (x) A2 + rho^2/2 A5 (x) A3,
      A_r = derivatives(alpha, pdf, gate, gaussian_mean / sigma, 5) (rows a on the left, c on the right), rho = Cpre / ss^T (zero diagonal).
    The n^2 products are float32 (the card's own series precision); the sum is added to the float64 C. Post21, post22 and radial stay those of
    the unmodified pair-map call.
  edge (default L8-14): the message-edge increment x (1 + edge), via a wrapper around the last _message_edge_source (appended).
  rc (default L1-14; L0 allowed, where the live loop also recenters): the recentering x (1 + rc): C0 = C is kept before the live loop's recentering statements, then C = C - rc (C0 - C).
usage: build_gn_knobs.py <base.py> <knobs.json> <out.py>   (refuses to overwrite)
  knobs.json: {"iso": -0.094, "k21n": -0.028, "k31": -0.212, "edge": 0.164, "rc": -0.08}  optional "cov_layers", "edge_layers", "rc_layers"."""
import sys, ast, json, hashlib
from pathlib import Path

APPEND = r'''

# fable-porter PT-7 (build_gn_knobs.py): covhunter's GN covariance-channel knobs.
_GN_KNOBS = {knobs!r}
_GN_COV_LAYERS = {cov_layers!r}
_GN_EDGE = {edge!r}
_GN_RC = {rc!r}
def _gn_dcov(l, mu, Cpre, K21, k4, aux):
    th = _GN_KNOBS
    s = aux['sigma']; a = aux['alpha']
    v = fnp.diag(Cpre).copy()
    A, _unused = derivatives(a, aux['pdf'], aux['gate'], aux['gaussian_mean'] / s, 5)
    A1, A2, A3, A4, A5 = [x.astype(fnp.float32) for x in A[1:6]]
    ss = fnp.outer(s, s)
    rho = (Cpre / ss).astype(fnp.float32)
    fnp.fill_diagonal(rho, 0)
    rho2 = rho * rho / 2
    k3 = fnp.diag(K21).copy()
    kink = (fnp.abs(a) < 1.5).astype(fnp.float64)
    dK21 = (th['iso'] * kink * k3 / v)[:, None] * Cpre + (th['k21n'] * (1 - kink))[:, None] * K21
    dT21 = (dK21 / (v[:, None] * s[None, :])).astype(fnp.float32)
    t4 = fnp.broadcast_to(k4, (len(mu),)) / v ** 2
    r31 = _k31_row_scale(mu, k3, v) if '_k31_row_scale' in globals() else None
    dT31 = (th['k31'] * (t4 * r31 if r31 is not None else t4)).astype(fnp.float32)[:, None] * rho
    t21a = fnp.outer(A2, A1) + rho * fnp.outer(A3, A2) + rho2 * fnp.outer(A4, A3)
    t31a = fnp.outer(A3, A1) + rho * fnp.outer(A4, A2) + rho2 * fnp.outer(A5, A3)
    X = dT21 * t21a / 2 + dT31 * t31a / 6
    dcov = (X + X.T).astype(fnp.float64) * ss
    fnp.fill_diagonal(dcov, 0)
    return dcov
_gn_message_edge_source_base = _message_edge_source
def _message_edge_source(layer, m, K21, aux, w, state, P, context):
    out = _gn_message_edge_source_base(layer, m, K21, aux, w, state, P, context)
    t = _GN_EDGE.get(layer)
    return out if not t else out * (1 + t)
'''


def live_function(src):
    return [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'][-1]


def patch(src, knobs):
    assert '_GN_KNOBS' not in src, 'base already carries the GN knobs'
    lines = src.split('\n')
    live = live_function(src)
    loop = [n for n in ast.walk(live) if isinstance(n, ast.For)]
    loop = [f for f in loop if any('pair_map(' in (ast.get_source_segment(src, s) or '') for s in f.body)]
    assert len(loop) == 1, 'live layer loop'
    body = loop[0].body
    seg = lambda node: ast.get_source_segment(src, node) or ''
    pm = [s for s in body if 'pair_map(' in seg(s) and 'radial_new' in seg(s)]
    assert len(pm) == 1, ('pair-map call statement', len(pm))
    pm = pm[0]
    omitted = [s for s in body if 'omitted_radial_source(' in seg(s)]
    assert len(omitted) == 1 and omitted[0].lineno > pm.end_lineno, 'the omitted-radial source must follow the pair-map call'
    hook = [s for s in body if isinstance(s, ast.If) and seg(s.test).strip() == 'hook is not None']
    assert len(hook) == 1, 'recentering block'
    hb = hook[0].body
    assigns = lambda s, name: any(isinstance(t, ast.Name) and t.id == name for n in ast.walk(s) if isinstance(n, ast.Assign) for t in n.targets)
    last_dm = max(i for i, s in enumerate(hb) if assigns(s, 'dm'))
    mstep = [i for i, s in enumerate(hb) if isinstance(s, ast.Assign) and seg(s).replace(' ', '') == 'm=m+dm']
    assert len(mstep) == 1, 'm = m + dm'
    rec = hb[last_dm + 1:mstep[0]]
    assert rec and assigns(rec[-1], 'C') and all(('outer' in seg(s) or '_e2_mdm' in seg(s)) for s in rec), 'recentering statements'
    ind_loop = ' ' * pm.col_offset; ind_hook = ' ' * rec[0].col_offset
    edits = [  # (line index after which to insert, text) -- applied bottom-up
        (rec[-1].end_lineno, [f'{ind_hook}if _GN_RC.get(l):', f'{ind_hook}    C = C - _GN_RC[l] * (_gn_C0 - C)']),
        (rec[0].lineno - 1, [f'{ind_hook}_gn_C0 = C']),
        (pm.end_lineno, [f'{ind_loop}if l in _GN_COV_LAYERS:', f'{ind_loop}    C = C + _gn_dcov(l, mu, Cpre, K21, k4, aux)']),
    ]
    for at, text in sorted(edits, key=lambda e: -e[0]):
        lines[at:at] = text
    th = {k: float(knobs.get(k, 0.0)) for k in ('iso', 'k21n', 'k31', 'edge', 'rc')}
    cov_on = any(th[k] for k in ('iso', 'k21n', 'k31'))
    cov_layers = tuple(knobs.get('cov_layers', range(2, 15))) if cov_on else ()
    edge = {l: th['edge'] for l in knobs.get('edge_layers', range(8, 15))} if th['edge'] else {}
    rc = {l: th['rc'] for l in knobs.get('rc_layers', range(1, 15))} if th['rc'] else {}
    assert all(1 <= l <= 14 for l in list(cov_layers) + list(edge)) and all(0 <= l <= 14 for l in rc), 'layers: cov/edge 1..14, rc 0..14'
    out = '\n'.join(lines) + APPEND.format(knobs=th, cov_layers=cov_layers, edge=edge, rc=rc)
    ast.parse(out)
    return out


if __name__ == '__main__':
    base, kf, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    knobs = json.loads(kf.read_text())
    text = patch(base.read_text(), knobs)
    with out.open('x') as f:
        f.write(text)
    print(json.dumps(dict(out=str(out), base=str(base), knobs=knobs, base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                          out_sha256=hashlib.sha256(text.encode()).hexdigest())))
