"""Source-only, fixed temporary-plane precision port; no numerical imports.

transform() requires the retained exact bundle and returns source plus reversible
whole-function edits. Scalar moment formulae and F64 interface caches stay native.
The generated code uses paid FNP operations only. No candidate is run here.
"""
import ast
import hashlib

PARENT_SHA = '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'

HELPERS = '''

# BEGIN FIXED TEMPORARY PLANE PRECISION PORT
def _ps32(x):
    return x if str(getattr(x, 'dtype', '')) == 'float32' else fnp.asarray(x, dtype=fnp.float32)

def _ps_pre32(C):
    L = _E2_LAYER
    if L.get('Cpre') is C:
        if 'ps_Cpre32' not in L:
            L['ps_Cpre32'] = _ps32(C)
        return L['ps_Cpre32']
    return _ps32(C)

def _ps_k2132(K):
    L = _E2_LAYER
    if L.get('ps_K21') is not K:
        L['ps_K21'] = K
        L['ps_K2132'] = _ps32(K)
    return L['ps_K2132']

def _ps_target32(T):
    L = _E2_LAYER
    return L['ps_target32'] if L.get('ps_target64') is T else _ps32(T)
# END FIXED TEMPORARY PLANE PRECISION PORT
'''


def transform(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA
    assert '\n_T31_LAYERS=()' in source or '\n_T31_LAYERS = ()' in source
    assert '\n_t31_commit=None' in source or '\n_t31_commit = None' in source
    lines = source.splitlines(keepends=True)
    nodes = [x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef)]
    changes = []

    def function(name, anchor=None):
        matches = [x for x in nodes if x.name == name and
                   (anchor is None or anchor in ''.join(lines[x.lineno-1:x.end_lineno]))]
        assert len(matches) == 1, (name, len(matches))
        node = matches[0]
        old = ''.join(lines[node.lineno-1:node.end_lineno])
        return node, old

    def patch(name, pairs, anchor=None):
        node, old = function(name, anchor)
        new = old
        for before, after in pairs:
            assert new.count(before) == 1, (name, before[:100], new.count(before))
            new = new.replace(before, after, 1)
        changes.append((node, old, new))

    # Normalize temporary pair planes in F32. Persistent ss/vs/vv, scalar
    # derivative tables and all scalar moments retain their parent F64 work.
    patch('_nonlinear_pair_batch', [
        ('    rho = C / ss', '    _ps_ss32 = _ps32(ss)\n    rho = _ps_pre32(C) / _ps_ss32'),
        ("fnp.divide(rho.astype(fnp.float32), st['DIV6'], out=P16)", "fnp.divide(rho, st['DIV6'], out=P16)"),
        ("    T21 = st['T21']", "    _ps_vs32 = _ps32(vs)\n    T21 = st['T21']"),
        ("fnp.divide(K21, vs, out=T21)", "fnp.divide(_ps_k2132(K21), _ps_vs32, out=T21)"),
        ("    _e2_hold.append(fnp.divide(K22, vv, out=st['T22']))",
         "    _ps_vv32 = L['ps_vv32'] if L.get('VV_for') is C and 'ps_vv32' in L else _ps32(vv)\n    _e2_hold.append(fnp.divide(_ps32(K22), _ps_vv32, out=st['T22']))"),
        ("fnp.multiply((t4 * _r31 if _r31 is not None else t4)[:, None], rho, out=st['T31'])",
         "fnp.multiply(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None], rho, out=st['T31'])"),
        ("fnp.add((t4 * _r31 if _r31 is not None else t4)[:, None] * rho, x31, out=st['T31'])",
         "fnp.add(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None] * rho, _ps32(x31), out=st['T31'])"),
        ("fnp.sum(st['Z'], axis=0)) * SVV", "fnp.sum(st['Z'], axis=0)) * fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32))"),
        ('    m2 = 2 * m', '    m2 = _ps32(2 * m)'),
        ('    fnp.fill_diagonal(post22, post_k4)',
         '    _ps_target = post21\n    cov = cov.astype(fnp.float64)\n    post21 = post21.astype(fnp.float64)\n    post22 = post22.astype(fnp.float64)\n    fnp.fill_diagonal(post22, post_k4)'),
        ('    fnp.fill_diagonal(post21, post_k3)',
         "    fnp.fill_diagonal(post21, post_k3)\n    fnp.fill_diagonal(_ps_target, _ps32(post_k3))\n    _E2_LAYER.update(ps_target64=post21, ps_target32=_ps_target)"),
        ("aux=aux, d=d, ss=ss, mgs=mgs)", "aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)"),
    ])

    # L0 has its own Gaussian series. Only plane operands/accumulators change;
    # gaussian_raw and the complete marginal/cumulant expressions do not.
    patch('_nonlinear_first_gaussian', [
        ('    rho = C / fnp.outer(s, s)',
         '    _ps_s32 = _ps32(s); _ps_v32 = _ps32(v)\n    rho = _ps_pre32(C) / fnp.outer(_ps_s32, _ps_s32)'),
        ('    T21 = K21 / (v[:, None] * s[None, :])',
         '    T21 = _ps_k2132(K21) / (_ps_v32[:, None] * _ps_s32[None, :])'),
        ('    T22 = K22 / fnp.outer(v, v)', '    T22 = _ps32(K22) / fnp.outer(_ps_v32, _ps_v32)'),
        ('    T31 = t4[:, None] * rho if k31 else None', '    T31 = _ps32(t4)[:, None] * rho if k31 else None'),
        ('    aa=fnp.stack(A,axis=0);bb=fnp.stack(B,axis=0)',
         '    aa=fnp.stack(A,axis=0,dtype=fnp.float32);bb=fnp.stack(B,axis=0,dtype=fnp.float32)'),
        ('    ans=fnp.zeros((3,n,n),dtype=fnp.float64)', '    ans=fnp.zeros((3,n,n),dtype=fnp.float32)'),
        ('    rr=fnp.ones_like(C)', '    rr=fnp.ones_like(rho)'),
        ('    cov=ans[0]*fnp.outer(s,s)', '    cov=ans[0]*fnp.outer(_ps_s32,_ps_s32)'),
        ('    pk21=ans[1]*fnp.outer(v,s)', '    pk21=ans[1]*fnp.outer(_ps_v32,_ps_s32)'),
        ('    pk22=ans[2]*fnp.outer(v,v)', '    pk22=ans[2]*fnp.outer(_ps_v32,_ps_v32)'),
        ('    post21 = pk21 - 2 * m[:, None] * cov',
         '    _ps_m32 = _ps32(m)\n    post21 = pk21 - 2 * _ps_m32[:, None] * cov'),
        ('    post22 = pk22 - 2 * m[:, None] * pk21.T - 2 * m[None, :] * pk21 + 4 * fnp.outer(m, m) * cov - 2 * cov * cov',
         '    post22 = pk22 - 2 * _ps_m32[:, None] * pk21.T - 2 * _ps_m32[None, :] * pk21 + 4 * fnp.outer(_ps_m32, _ps_m32) * cov - 2 * cov * cov'),
        ('    fnp.fill_diagonal(post22, post_k4)',
         '    _ps_target = post21\n    cov = cov.astype(fnp.float64)\n    post21 = post21.astype(fnp.float64)\n    post22 = post22.astype(fnp.float64)\n    fnp.fill_diagonal(post22, post_k4)'),
        ('    fnp.fill_diagonal(post21, post_k3)',
         "    fnp.fill_diagonal(post21, post_k3)\n    fnp.fill_diagonal(_ps_target, _ps32(post_k3))\n    _E2_LAYER.update(ps_target64=post21, ps_target32=_ps_target)"),
    ])

    # Learned summary planes only: physical vectors/normalizers stay F64 and
    # reductions explicitly accumulate in F64. No feature or weight changes.
    feature_pairs = [
        ("    rho = aux['pre_cov'] / (pr['ss'] if pr else fnp.outer(s, s))",
         "    _ps_s32 = _ps32(s); _ps_v32 = _ps32(v)\n    rho = _ps_pre32(aux['pre_cov']) / (pr['ss32'] if pr else fnp.outer(_ps_s32, _ps_s32))"),
        ("    skplane = K21 / _e2_H(fnp.outer(v, s))", "    skplane = _ps_k2132(K21) / _e2_H(fnp.outer(_ps_v32, _ps_s32))"),
        ("    postrho = C / _e2_H(fnp.outer(postscale, postscale))", "    _ps_postscale32 = _ps32(postscale)\n    postrho = _ps32(C) / _e2_H(fnp.outer(_ps_postscale32, _ps_postscale32))"),
    ]
    _, fs = function('features')
    for term in ('fnp.mean(rho, axis=1)', 'fnp.mean(_e2_H(rho * rho), axis=1)', 'fnp.mean(_e2_H(fnp.abs(rho)), axis=1)',
                 'fnp.sum(skplane_square, axis=1)', 'fnp.sum(skplane_square, axis=0)', 'fnp.sum(skplane, axis=1)',
                 'fnp.sum(skplane, axis=0)', 'fnp.mean(_e2_H(postrho * postrho), axis=1)', 'fnp.mean(postrho, axis=1)'):
        assert term in fs
        feature_pairs.append((term, term[:-1] + ', dtype=fnp.float64)'))
    patch('features', feature_pairs)

    patch('_gn_dcov', [
        ("    dK21 = _e2_H(_g5_rowmul(_e2_H(_e2_H(_g5_schain(th['iso'], kink, k3)) / v), Cpre)) + _e2_H(_g5_rowmul(_e2_H(_e2_H(th['k21n']) * _e2_H(1 - kink)), K21))",
         "    dK21 = _e2_H(_g5_rowmul(_ps32(_e2_H(_g5_schain(th['iso'], kink, k3)) / v), _ps_pre32(Cpre))) + _e2_H(_g5_rowmul(_ps32(_e2_H(th['k21n']) * _e2_H(1 - kink)), _ps_k2132(K21)))"),
        ("    dT21 = _e2_H(dK21 / _e2_H(_g5_outer(v, s))).astype(fnp.float32)",
         "    dT21 = dK21 / _e2_H(_g5_outer(_ps32(v), _ps32(s)))"),
        ("    dcov = _e2_H(_e2_H(X + _e2_H(X.T)).astype(fnp.float64)) * ss",
         "    dcov = _e2_H(_e2_H(X + _e2_H(X.T)) * pr['ss32']).astype(fnp.float64)"),
    ], anchor="_g5st =")

    patch('_g5_message_edge_base', [
        ("fnp.divide(aux['pre_cov'],pr['ss'] if pr is not None and pr['aux'] is aux else fnp.outer(s,s),out=R)",
         "fnp.divide(_ps_pre32(aux['pre_cov']),pr['ss32'] if pr is not None and pr['aux'] is aux else fnp.outer(_ps32(s),_ps32(s)),out=R)"),
        ("fnp.divide(K21,fnp.outer(s**2,s),out=S)", "fnp.divide(_ps_k2132(K21),fnp.outer(_ps32(s**2),_ps32(s)),out=S)"),
        (" * t['cov_factor']).astype(fnp.float64)", " * t['cov_factor'])"),
    ])
    patch('_message_edge_source', [
        ('    return out if not t else out * (1 + t)',
         '    _ps_diag = fnp.diag(out).astype(fnp.float64)\n    if t:\n        _ps_diag = _ps_diag * (1 + t)\n        out = out * _ps32(1 + t)\n    out = out.astype(fnp.float64)\n    fnp.fill_diagonal(out, _ps_diag)\n    return out'),
    ], anchor='t = _GN_EDGE.get(layer)')

    patch('_predict_binary', [
        ("            K21 = _e2_H(_e2_H(cp_pre + star_pre) + _e2_H(fnp.outer(vv if Bpre is Cpre else fnp.diag(Bpre), bpre))) + _e2_H(fnp.einsum('ij,i->ij', Bpre, _e2_H(2 * bpre), optimize=False))",
         "            _ps_bv = vv if Bpre is Cpre else fnp.diag(Bpre)\n            _ps_kdiag = (fnp.diag(cp_pre) + fnp.diag(star_pre) + _ps_bv * bpre) + fnp.diag(Bpre) * (2 * bpre)\n            _ps_cp32 = _ps32(cp_pre); _ps_bpre32 = _ps32(bpre)\n            _ps_K2132 = (_ps_cp32 + _ps32(star_pre) + fnp.outer(_ps32(_ps_bv), _ps_bpre32)) + fnp.einsum('ij,i->ij', _ps_pre32(Bpre), _ps32(2 * bpre), optimize=False)\n            K21 = _ps_K2132.astype(fnp.float64)\n            fnp.fill_diagonal(K21, _ps_kdiag)\n            fnp.fill_diagonal(_ps_K2132, _ps32(_ps_kdiag))\n            _E2_LAYER.update(ps_K21=K21, ps_K2132=_ps_K2132)"),
        ("            k22 = _e2_H(radial * _e2_H(_e2_VV + _e2_H(2 * _e2_CC))) / 3",
         "            _ps_k22diag = (radial * (fnp.diag(_e2_VV) + 2 * fnp.diag(_e2_CC))) / 3\n            _ps_vv32 = _ps32(_e2_VV)\n            _E2_LAYER['ps_vv32'] = _ps_vv32\n            k22 = _ps32(radial) * (_ps_vv32 + 2 * _ps32(_e2_CC)) / 3\n            fnp.fill_diagonal(k22, _ps32(_ps_k22diag))"),
        ("                _e2_mdm = fnp.outer(m, dm)\n                C = _e2_H(_e2_H(C - _e2_mdm) - _e2_H(_e2_mdm.T)) - _e2_H(fnp.outer(dm, dm))\n                if _GN_RC.get(l):\n                    C = C - _e2_H(_e2_H(_GN_RC[l]) * _e2_H(_gn_C0 - C))",
         "                _ps_Cdiag0 = fnp.diag(C).copy()\n                _ps_mdm_diag = m * dm\n                _ps_Cdiag = ((_ps_Cdiag0 - _ps_mdm_diag) - _ps_mdm_diag) - dm * dm\n                _ps_m32 = _ps32(m); _ps_dm32 = _ps32(dm)\n                _e2_mdm = fnp.outer(_ps_m32, _ps_dm32)\n                _ps_C0 = _ps32(C)\n                C = ((_ps_C0 - _e2_mdm) - _e2_mdm.T) - fnp.outer(_ps_dm32, _ps_dm32)\n                if _GN_RC.get(l):\n                    _ps_Cdiag = _ps_Cdiag - _GN_RC[l] * (_ps_Cdiag0 - _ps_Cdiag)\n                    C = C - _ps32(_GN_RC[l]) * (_ps_C0 - C)\n                C = C.astype(fnp.float64)\n                fnp.fill_diagonal(C, _ps_Cdiag)"),
        ("            inherited = fnp.einsum('i,j,ij->ij', _e2_H(g ** 2), g, cp_pre, optimize=False)",
         "            _ps_g2 = g ** 2\n            _ps_inherited_diag = fnp.einsum('i,i,i->i', _ps_g2, g, fnp.diag(cp_pre), optimize=False)\n            inherited = fnp.einsum('i,j,ij->ij', _ps32(_ps_g2), _ps32(g), _ps_cp32, optimize=False)"),
        ("            trace_post = _e2_H(fnp.outer(_e2_H(fnp.diag(Bpost)), b)) + _e2_H(fnp.einsum('ij,i->ij', Bpost, _e2_H(2 * b), optimize=False))",
         "            _ps_Bdiag = fnp.diag(Bpost)\n            _ps_trace_diag = _ps_Bdiag * b + _ps_Bdiag * (2 * b)\n            trace_post = fnp.outer(_ps32(_ps_Bdiag), _ps32(b)) + fnp.einsum('ij,i->ij', _ps32(Bpost), _ps32(2 * b), optimize=False)"),
        ("            innovation = _e2_H(_e2_H(fnp.diag(target)) - _e2_H(fnp.diag(inherited))) - _e2_H(fnp.diag(trace_post))",
         "            innovation = _e2_H(_e2_H(fnp.diag(target)) - _ps_inherited_diag) - _ps_trace_diag"),
        ("            born = _e2_H(fnp.einsum('ij,ij,j->ij', bp, bp, _e2_bb, optimize=False)) + _e2_H(fnp.einsum('i,ji->ij', _e2_H(beta * births), bp, optimize=False))",
         "            _ps_bp32 = _ps32(bp)\n            born = fnp.einsum('ij,ij,j->ij', _ps_bp32, _ps_bp32, _ps32(_e2_bb), optimize=False) + fnp.einsum('i,ji->ij', _ps32(beta * births), _ps_bp32, optimize=False)"),
        ("            fnp.fill_diagonal(born, _e2_H(_e2_H(_e2_bb * _e2_H(_e2_H(_e2_H(3 * u) * u) + _e2_H(delta * delta))) / 4))",
         "            _ps_born_diag = _e2_H(_e2_H(_e2_bb * _e2_H(_e2_H(_e2_H(3 * u) * u) + _e2_H(delta * delta))) / 4)"),
        ("            born = born + _bd_plane_delta",
         "            born = born + _bd_plane_delta\n            _ps_born_diag = _ps_born_diag + fnp.diag(_bd_plane_delta)\n            fnp.fill_diagonal(born, _ps_born_diag)"),
        ("            cp_post = _e2_H(inherited + born) * live",
         "            _ps_cpdiag = (_ps_inherited_diag + _ps_born_diag) * fnp.diag(live)\n            cp_post = (inherited + _ps32(born)) * live"),
        ("            rest = _e2_H(target - cp_post) - trace_post\n            star = _e2_H(_e2_H(3 * _e2_H(rest.T)) - _e2_H(2 * _e2_H(fnp.diag(_e2_H(fnp.diag(rest)))))).astype(fnp.float32)",
         "            _ps_rest_diag = (fnp.diag(target) - _ps_cpdiag) - _ps_trace_diag\n            _ps_star_diag = (3 * _ps_rest_diag - 2 * _ps_rest_diag).astype(fnp.float32)\n            rest = (_ps_target32(target) - cp_post) - trace_post\n            star = 3 * rest.T - 2 * fnp.diag(fnp.diag(rest))\n            fnp.fill_diagonal(star, _ps_star_diag)"),
    ], anchor='            cp_post = _e2_H(inherited + born) * live')

    result = source
    edits = []
    for node, before, after in sorted(changes, key=lambda x: x[0].lineno, reverse=True):
        assert result.count(before) == 1
        result = result.replace(before, after, 1)
        edits.append(dict(label='precision_' + node.name, function=node.name, original_line=node.lineno, before=before, after=after))
    result += HELPERS
    edits.append(dict(label='precision_helpers', before='', after=HELPERS))
    ast.parse(result)
    assert inverse(result, edits) == source
    return result, edits


def inverse(source, edits):
    for e in reversed(edits):
        assert source.count(e['after']) == 1
        source = source.replace(e['after'], e['before'], 1)
    return source
