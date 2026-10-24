#!/usr/bin/env python3
"""TAG opus-physicist T110: the trainer's generated-seed worker v4 (unchanged, through runpy) on the PINNED otr_m.py copy, plus a read-only DUMP for the lead's
star pair-closure gate (10-11): the star plane the state carries INTO layers 11-13 (= the star made at 10-12), and at layers 11-14 what the pair map receives
(mu, C, K21) with the gate the step formed, and the layer's float64 weights. A request whose cutsim opts carry 't110tag': <name> (removed before the
pinned simulator sees its options) writes <run>/dumps/<name>.npz. Arithmetic = worker_generated_v4 exactly (captures under no_grad).
usage: as worker_generated_v4.py, run from O/trainer: python -B ../physicist/t110_worker_stardump.py <run dir> <worker index>"""
import hashlib
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINER, FROZEN_M = HERE.parent / 'trainer', HERE / 'otr_frozen_bba0dea0'
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(FROZEN_M))
import otr_m as _M_FROZEN
assert Path(_M_FROZEN.__file__).resolve().parent == FROZEN_M and hashlib.sha256(Path(_M_FROZEN.__file__).read_bytes()).hexdigest().startswith('bba0dea0'), _M_FROZEN.__file__
import otr_cut_runtime_v2 as RT

enable0, CUR = RT.enable, {}


def enable(engine):
    configure0 = enable0(engine)
    install_runtime = engine.M.install
    torch = engine.torch
    f32 = lambda z: z.detach().to(torch.float32).cpu().numpy()

    def configure(spec):
        CUR.clear()
        if spec and 't110tag' in (spec.get('opts') or {}):
            opts = dict(spec['opts']); CUR['tag'] = opts.pop('t110tag'); spec = dict(spec, opts=opts)
        configure0(spec)

    def install(cd):
        out = install_runtime(cd)
        si0, pin0 = cd.hooks.get('state_in'), cd.hooks['pair_in']
        again = lambda: getattr(cd, 'm_side', {}).get('again')

        def state_in(l, s, ctx):
            s = s if si0 is None else si0(l, s, ctx)
            if CUR.get('tag') and 11 <= l <= 13 and not again():
                with torch.no_grad():
                    CUR.setdefault('d', {})['star_into_%d' % l] = f32(s[4])
            return s

        def pin_hook(l, pin, ctx):
            pin = pin0(l, pin, ctx)
            if CUR.get('tag') and 11 <= l <= 14 and not again():
                with torch.no_grad():
                    d = CUR.setdefault('d', {})
                    d.update({'mu_%d' % l: f32(pin[0]), 'C_%d' % l: f32(pin[1]), 'K21_%d' % l: f32(pin[2]), 'gate_%d' % l: f32(ctx['gate']),
                              'W_%d' % l: f32(cd.ws[l])})
                    if l == 14:
                        (RUN / 'dumps').mkdir(exist_ok=True)
                        np.savez(RUN / 'dumps' / ('%s.npz' % CUR['tag']), **d)
            return pin
        cd.hooks['state_in'] = state_in
        cd.hooks['pair_in'] = pin_hook
        return out
    engine.M.install = install
    return configure


RT.enable = enable
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
