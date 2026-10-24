#!/usr/bin/env python3
"""TAG opus-physicist T98: the trainer's generated-seed worker (trainer/worker_generated_v4.py, unchanged, run through runpy) plus ONE arm for the
research gate #1 (physicist 10-11 11:06Z): the missing SCALE MODE in the K21 the pair map uses for its COVARIANCE output only.
A request whose cutsim opts carry 't98': {'gamma': g, 'lo': a, 'hi': b} (removed before the pinned simulator sees its options) gets, at layers
a..b, the pair map evaluated a second time with K21_C = K21 + kappa_C F, F_ij = v_i mu_j + 2 C_ij mu_i (diagonal 0),
kappa_C = max(g radial / 6 - (b_pre . mu) / (mu . mu), 0) (model_bank_bfix's k21off_pair recipe), and ONLY that call's post covariance is kept
(mean, third-order target, radial and aux from the card's own call): T78's 'K21c' channel split with the scale mode in place of the truth.
radial and b_pre = W' b come from the state the layer receives (read after M's own state_in). Without 't98' this is worker_generated_v4 exactly.
usage: as worker_generated_v4.py, run from O/trainer: python -B ../physicist/t98_worker_k21c.py <run dir> <worker index>"""
import runpy
import sys
from pathlib import Path

TRAINER = Path(__file__).resolve().parent.parent / 'trainer'
sys.path.insert(0, str(TRAINER))
import otr_cut_runtime_v2 as RT

enable0, CUR = RT.enable, {}


def enable(engine):
    configure0 = enable0(engine)
    install_runtime = engine.M.install
    torch = engine.torch
    f64 = torch.float64

    def configure(spec):
        CUR.clear()
        if spec and 't98' in (spec.get('opts') or {}):
            opts = dict(spec['opts'])
            CUR['arm'] = dict(opts.pop('t98'))
            spec = dict(spec, opts=opts)
        configure0(spec)

    def install(cd):
        out = install_runtime(cd)
        inner = cd.hooks.get('state_in')
        pmap0 = cd.env['_nonlinear_pair_batch']
        side = lambda: getattr(cd, 'm_side', {})

        def state_in(l, s, ctx):
            s = s if inner is None else inner(l, s, ctx)
            a = CUR.get('arm')
            if a and a['lo'] <= l <= a['hi'] and not side().get('again'):
                with torch.no_grad():
                    CUR['radial'] = float(torch.as_tensor(s[7]).detach().to(f64))
                    CUR['bpre'] = cd.ws[l].to(f64).T @ s[5].detach().to(f64)
                    CUR['layer'] = l
            return s

        def pmap(*args, **kwargs):
            res = pmap0(*args, **kwargs)
            a, l = CUR.get('arm'), cd.call_layer
            if not a or not (a['lo'] <= l <= a['hi']) or CUR.get('layer') != l or side().get('again'):
                return res
            with torch.no_grad():
                mu, C, K21 = (z.detach().to(f64) for z in args[:3])
                v = torch.diagonal(C); mm = float(mu @ mu)
                kappa = max(float(a['gamma']) * CUR['radial'] / 6 - float(CUR['bpre'] @ mu) / max(mm, 1e-300), 0.0)
                F = v[:, None] * mu[None, :] + 2 * C * mu[:, None]
                F = F - torch.diag(torch.diagonal(F))
                res2 = pmap0(args[0], args[1], (K21 + kappa * F).to(args[2].dtype), *args[3:], **kwargs)
                CUR.setdefault('kappa', {})[l] = kappa
            return (res[0], res2[1]) + tuple(res[2:])
        cd.hooks['state_in'] = state_in
        cd.env['_nonlinear_pair_batch'] = pmap
        return out
    engine.M.install = install
    return configure


RT.enable = enable
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
