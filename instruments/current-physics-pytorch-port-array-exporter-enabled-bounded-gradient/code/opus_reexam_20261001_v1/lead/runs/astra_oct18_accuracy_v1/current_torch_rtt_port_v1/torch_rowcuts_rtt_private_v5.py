"""S46 row-cut simulator v15, injectable torch-only research module (2026-10-10).

Install AFTER PL.install and native-push simulation, BEFORE M.install:
    rowcuts.install(cd, torch, log=rows)
This module imports no torch, runs no network, and is NOT submission code.

Differences from stale torch_cutsim_v2:
* current S46 dead threshold 3.0902;
* diagonal repair occurs before k21 is formed, without in-place tensor mutation;
* omitted-radial correction occurs before b-fix/births/star, not state_out;
* supports the native push simulator's blockwise old-tail factors;
* all-off is a strict no-op; source replacements each assert a single match.

NOT YET CERTIFIED: prediction and gradient checks are mandatory. Native K4L's
projection must use loc!=0 rows: call install_m(cd, original_m_install)
INSTEAD OF original_m_install(cd) after install(cd). Covariance/edge cuts and float32/tree rounding remain unmodeled.
"""
import inspect
import math
import sys
import textwrap
import types

DEFAULTS = dict(dead=3.0902, ott=2.5, neg=2.5, ynt=2.5, orep=1.0, yrep=1.0, rtail=1.0, l15=3.5)
CUTS = ('dead', 'ott', 'neg', 'ynt', 'l15')


def install(cd, t=None, opts=None, log=None):
    t = cd.t if t is None else t
    o = dict(DEFAULTS, **(opts or {}))
    if any(k not in DEFAULTS for k in o):
        raise ValueError('unknown row-cut option')
    if not any(o[k] < 90 for k in CUTS):
        cd._cutsim_finish_step = lambda: None   # native-push-only diagnostic still installs M
        return cd
    assert hasattr(cd, 'pl_kappa') and hasattr(cd, 'pl_bsrc'), 'install after PL.install'
    assert 'state_in' not in cd.hooks, 'install before M.install'
    log = [] if log is None else log
    st = {}
    cd.cutsim_rows = st

    def top(score, multiple):
        count = min(score.numel(), ((int((score.detach() >= 0).sum()) + multiple - 1) // multiple) * multiple)
        order = t.argsort(-score.detach(), stable=True)
        return order[:count], order[count:]

    def begin(layer, mean, cov):
        st.clear()
        cd.cutsim_np_column = None
        cd.cutsim_law_pp = None
        st.update(layer=layer, mean=mean, cov=cov, repair=None, blocks=())
    cd._cutsim_begin = begin
    parent_push = cd.shared['push']

    def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
        result = parent_push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        if layer == 0 or not lam.numel():
            return result
        pp, U, Fn, coren, old = result
        cd.cutsim_law_pp = (int(layer), pp.detach(), int(old))  # one-step original right old factors for frozen law
        n = int(pp.shape[0])
        mu = cd.ws[layer].T @ st['mean']
        vv = (w.to(t.float64) * (st['cov'].to(t.float32) @ w).to(t.float64)).sum(0)
        alpha = mu / vv.clamp_min(1e-10).sqrt()
        if layer == 15:
            if o['l15'] >= 90:
                return result
            keep, gone = top(o['l15'] - alpha.abs(), 32)
            pp = pp.index_fill(0, gone, 0)
            st['keep'] = keep
            log.append(dict(layer=15, final_gone=int(gone.numel())))
            return pp, U, Fn, coren, old
        gate = t.erfc(-alpha / math.sqrt(2)) / 2
        alive = t.zeros(n, dtype=t.bool, device=pp.device)
        alive[top(gate - .01, 32)[0]] = True
        keep = t.arange(n, device=pp.device)
        rec = dict(layer=int(layer))
        if o['dead'] < 90:
            keep, gone = top(alpha + o['dead'], 64)
            pp = pp.index_fill(0, gone, 0)
            rec['dead'] = int(gone.numel())
        mask = t.zeros(n, dtype=t.bool, device=pp.device)
        mask[keep] = True
        assert not bool((alive & ~mask).any()), 'dead push cut removed an alive row'
        st['keep'] = keep
        st['mask'] = mask
        ak, alivek, m = alpha[keep], alive[keep], int(keep.numel())
        repair = t.zeros(n, dtype=t.float64, device=pp.device)
        blocks = []
        if 7 <= layer <= 14 and old and (o['ott'] < 90 or o['neg'] < 90):
            negative = (ak < -o['neg']) & ~alivek
            score = t.where(negative, ak + o['neg'], o['ott'] - ak)
            _, gone = top(score, 64)
            if gone.numel():
                tail = keep[gone]
                # Producer contract: [(U_full_rows, F_exact, column_start), ...].
                # Trace repairs always use FULL factors, never truncated Fu.
                push_context = getattr(cd, 'native_push_context', None)
                given = ([(b['U'], b['F'], b['lo']) for b in push_context['blocks']]
                         if push_context and push_context.get('blocks') else None)
                if hasattr(cd, 'native_push_context'):
                    assert given is not None, 'native-push block metadata missing'
                    assert sum(int(b[1].shape[1]) for b in given) == old
                    assert len(given) == (2 if layer >= 13 else 1), (layer, len(given))
                if given is None:
                    given = [(U, Fn, 0)]
                    rec['combined_old_fallback'] = True
                for Ub, Fb, start in given:
                    end = start + int(Fb.shape[1])
                    Uk = Ub[keep]
                    norm2 = (Fb * Fb).sum(0)
                    rho = (Uk * Uk).sum() / (Uk.shape[0] * Uk.shape[1])
                    vector = Fb @ (lam[start:end] * norm2)
                    part = ((Ub[tail] @ vector) * (3 * rho)).to(t.float64) * o['orep']
                    repair = repair.index_add(0, tail, part)
                    blocks.append((tail, Ub[tail], Fb, norm2, start, end))
                old_pp = pp[:, :old].index_fill(0, tail, 0)
                pp = t.cat((old_pp, pp[:, old:]), dim=1)
                rec['old_tail'] = int(tail.numel())
        if 7 <= layer <= 14 and o['ynt'] < 90 and pp.shape[1] > old:
            negative = (ak < -o['ynt']) & ~alivek
            count = min(m, ((int((~negative).sum()) + 63) // 64) * 64)
            if m - count >= 64:
                score = t.where(negative, ak + o['ynt'], t.zeros_like(ak))
                tail = keep[t.argsort(-score.detach(), stable=True)[count:]]
                assert not bool(alive[tail].any())
                young = p[:, old:]
                tau = young @ (lam[old:] * (young * young).sum(0))
                wt = w[:, tail]
                part = ((3 * (wt * wt).sum(0) / (n + 2)) * (wt.T @ tau)).to(t.float64)
                repair = repair.index_add(0, tail, part * o['yrep'])
                pp = t.cat((pp[:, :old], pp[:, old:].index_fill(0, tail, 0)), dim=1)
                rec['young_tail'] = int(tail.numel())
        st.update(repair=repair, blocks=tuple(blocks))
        log.append(rec)
        return pp, U, Fn, coren, old
    cd.shared['push'] = push

    def repair_diagonal(layer, diagonal):
        repair = st.get('repair')
        return diagonal if repair is None else diagonal + repair
    cd._cutsim_diagonal = repair_diagonal

    def final_star(layer, diagonal):
        if layer != 15 or 'keep' not in st:
            return diagonal
        keep = st['keep']
        mask = t.zeros_like(diagonal, dtype=t.bool).scatter(0, keep, t.ones_like(keep, dtype=t.bool))
        return t.where(mask, diagonal, t.zeros_like(diagonal))
    cd._cutsim_final_star = final_star

    def radial_columns(layer, energy, column, aa, beta):
        for tail, Ut, Fb, norm2, start, end in (st.get('blocks', ()) if o['rtail'] else ()):
            rowvar = ((Ut * Ut).sum(1) / Ut.shape[1]).to(t.float64)
            de = norm2.to(t.float64) * (aa[tail] * rowvar).sum()
            dc = ((Ut.T @ beta[tail].to(t.float32)) @ Fb).to(t.float64)
            energy = t.cat((energy[:start], energy[start:end] + de, energy[end:]))
            column = t.cat((column[:start], column[start:end] + dc, column[end:]))
        cd.cutsim_np_column = (int(layer), column.detach())  # frozen law state is detached by design
        return energy, column
    cd._cutsim_radial_columns = radial_columns

    # PL's two source edits are preserved explicitly. These assert the exact
    # pre-M class step and avoid mutating a tensor another autograd op saved.
    cls = type(cd)
    source = textwrap.dedent(inspect.getsource(cls.step))
    replacements = (
        ('birth = d * (1 + radial / 3)\n', 'birth = d * (1 + self.pl_kappa(layer) * radial / 3)\n'),
        ('b = g * bpre + radial * d * vv / 6\n', 'b = g * bpre + self.pl_bsrc(layer) * radial * d * vv / 6\n'),
        ('    w, w64 = self.w[layer], self.ws[layer]\n', '    self._cutsim_begin(layer, m, C)\n    w, w64 = self.w[layer], self.ws[layer]\n'),
        ("        k3s = f['_flms_star_diagonal'](w, star)\n", "        k3s = f['_flms_star_diagonal'](w, star)\n        k3s = self._cutsim_final_star(layer, k3s)\n"),
        ('        cp = self.diagonal(cp, cpdiag)\n', '        cpdiag = self._cutsim_diagonal(layer, cpdiag)\n        cp = self.diagonal(cp, cpdiag)\n'),
        ('        column = p64.T @ beta\n', '        column = p64.T @ beta\n        energy, column = self._cutsim_radial_columns(layer, energy, column, aa, beta)\n'),
    )
    for before, after in replacements:
        assert source.count(before) == 1, (before, source.count(before))
        source = source.replace(before, after)
    # Pure moment reductions retain only the float32 bank input and small
    # vector outputs. Large float64 products are recreated inside backward.
    import importlib.util
    from pathlib import Path
    moment_path = Path(__file__).resolve().with_name('core_moment_checkpoint_v2.py')
    moment_spec = importlib.util.spec_from_file_location('_o18_core_moments_v2', moment_path)
    moment_module = importlib.util.module_from_spec(moment_spec)
    moment_spec.loader.exec_module(moment_module)
    moment_module.install(cd, t)
    moment_replacements = (
        ('        p64 = pp.to(t.float64)\n        k3b = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)\n',
         '        k3b = self._o18_moment_diag(pp, lam)\n'),
        ('        p64 = pp.to(t.float64)\n        cpdiag = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)\n',
         '        cpdiag = self._o18_moment_diag(pp, lam)\n'),
        ('        energy = (p64 * p64).T @ aa\n        column = p64.T @ beta\n',
         '        energy, column = self._o18_moment_columns(pp, aa, beta)\n'),
    )
    for before, after in moment_replacements:
        assert source.count(before) == 1, (before, source.count(before))
        source = source.replace(before, after)
    assert 'p64' not in source, 'unexpected retained moment float64 cast'
    namespace = dict(vars(sys.modules[cls.__module__]))
    exec(compile(source, '<rowcuts-v15 with PL birth and radial repairs>', 'exec'), namespace)
    # Preserve the incoming PL step until pbase has applied its own changes.
    # The private pbase clone consumes the class source with ONLY our cut
    # hooks inserted, then makes its original PL/bsk/gate replacements itself.
    class_with_cuts = source
    for before, after in replacements[:2]:
        assert class_with_cuts.count(after) == 1
        class_with_cuts = class_with_cuts.replace(after, before)
    cd._cutsim_class_source = class_with_cuts
    cd._cutsim_step_applied = False
    compiled_step = namespace['step']
    def finish_step():
        if not cd._cutsim_step_applied:
            cd.step = types.MethodType(compiled_step, cd)
            cd._cutsim_step_applied = True
    cd._cutsim_finish_step = finish_step
    cd.cutsim_row_log = log
    return cd


def install_m(cd, original_m_install):
    """Install M with native K4L projection, preserving M's other hooks.

    Invoke instead of original M.install when cutsim is on. The sole source
    replacement is asserted; all row/data flow stays under M's checkpoint logic.
    """
    source = textwrap.dedent(inspect.getsource(original_m_install))
    before = """        if rows is None:
            beta = (w4 * loc * iso).sum() / (w4 * iso * iso).sum()
        else:
            beta = (w4 * loc * iso)[rows].sum() / (w4 * iso * iso)[rows].sum()
        return loc - beta * iso
"""
    after = """        pushed = (loc != 0).to(loc.dtype)
        if rows is not None:
            if rows.dtype == t.bool:
                assert rows.shape == loc.shape, 'K4L boolean row mask shape'
                selected = rows.to(loc.dtype)
            else:
                assert rows.dtype == t.int64 and rows.ndim == 1, 'K4L row indices must be int64'
                selected = t.zeros_like(loc).scatter(0, rows, t.ones_like(loc[rows]))
            pushed = pushed * selected
        wi = w4 * pushed * iso
        beta = (wi * loc).sum() / (wi * iso).sum().clamp_min(1e-300)
        return (loc - beta * iso) * pushed
"""
    assert source.count(before) == 1, ('M projection source changed', source.count(before))
    source = source.replace(before, after)
    roster_before = '                    kept = mu15.abs() <= L15_ROWCUT_Z * vv.sqrt()\n'
    roster_after = roster_before + ("                    final_rows = getattr(cd, 'cutsim_rows', {})\n"
        "                    if final_rows.get('layer') == 15 and 'keep' in final_rows:\n"
        "                        row_ids = final_rows['keep']\n"
        "                        assert row_ids.dtype == t.int64 and row_ids.ndim == 1\n"
        "                        kept = t.zeros_like(mu15, dtype=t.bool).scatter(0, row_ids, t.ones_like(row_ids, dtype=t.bool))\n")
    assert source.count(roster_before) == 1, 'M final head roster source changed'
    source = source.replace(roster_before, roster_after)
    namespace = dict(original_m_install.__globals__)
    parent_loader = namespace['_ncar']
    import importlib.util
    from pathlib import Path
    path = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py')
    spec = importlib.util.spec_from_file_location('_s46_private_pbase', path)
    private_pbase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(private_pbase)
    parent_loader('n6y')  # initialize frozen-module search paths before private clone imports specific
    lawpath = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1/n6y_rtt_private_v2.py')
    lawspec = importlib.util.spec_from_file_location('_s46_private_n6y', lawpath)
    private_law = importlib.util.module_from_spec(lawspec)
    lawspec.loader.exec_module(private_law)
    namespace['_ncar'] = lambda name: (private_pbase if name == 'pbase' else private_law if name == 'n6y' else parent_loader(name))
    # The M replay closure must capture the COMPOSED pbase+PL+rowcut step.
    site = '        step0 = cd.step\n'
    assert source.count(site) == 1, 'M replay step binding changed'
    source = source.replace(site, '        cd._cutsim_finish_step()\n' + site)
    exec(compile(source, '<M.install with native K4L rows and composed pbase cuts>', 'exec'), namespace)
    result = namespace[original_m_install.__name__](cd)
    cd._cutsim_finish_step()   # also covers requests with no carrier/replay wrapper
    return result

_O18_PARENT_INSTALL_M = install_m
_O18_RTT_MODULE = None
_O18_PUSH_MODULE = None
def install_m(cd, original_m_install):
    answer = _O18_PARENT_INSTALL_M(cd, original_m_install)
    assert _O18_RTT_MODULE is not None and _O18_PUSH_MODULE is not None
    assert answer is None or answer is cd
    return _O18_RTT_MODULE.install_after_m(cd, _O18_PUSH_MODULE)
