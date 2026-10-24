"""TAG scout-card K3 v7 (2026-10-12, idea sprint v2 #1): = v6 + atoms k4lf34 / k4lf34h (K4L ages 3-4 at the FINAL layer only, strength 1 / 0.5:
pl k4la3 = k4la4 = -1 on layers 0-14 and c - 1 at layer 15; the pair map and births are untouched, only the final mean's k4 changes). v6's text follows.
TAG scout-card K3 v6 (2026-10-12): = v5 + atom a4k31b1214 (the age-4 cohort at 0.6 only at read layers 12-14; scout-native: 0.023% bill vs 0.120% for 4-14). v5's text follows.
TAG scout-card K3 v5 (2026-10-12): = v4 + atom a4k31b1114 (K31L4 age-4 cohort at 0.6 ONLY at read layers 11-14: a new k31ly age-4 instance on the
unused leaf family nkbk with base 0 and request pl nkbk = 0.6 at L11-14, and pl nkl3 -0.6 at L11-14 added to the stored shift, so ages 1-3 keep their
strengths at every layer; lead 22:1xZ, natively a zero age-4 entry is the 3-age read bit for bit). v4's text follows.
TAG scout-card K3 v4 (2026-10-12): = v3 + atom a4k31s10 (K31L4 age-4 cohort at strength 1.0: natively bill-free, lead 22:1xZ). v3's text follows.
TAG scout-card K3 v3 (2026-10-12): = v2 + atom a3k4l7e (K4L age 3 at 0.7 on layers 1-9 only, off at 0 and 10-15). v2's text follows.
TAG scout-card K3 v2 (2026-10-12): = v1 + atoms gk05 / gk15 (transport gate at strength 0.5 / 1.5: gsk = gk4 = s). v1's text follows.
TAG scout-card K3 v1 (2026-10-12): = make_k2_v2 for worker_sc5.py, every request's cut simulator re-pointed to the scout's private copy
O/scout_card/cutsim_sc1 (identical files except pbase_cutsim_v1.py, which adds the keywords gbp / gbeta, inert at 0), plus atoms gkp (gk + gbp), gkB
(gk + gbeta), gkpB (gk + gbp + gbeta). make_k2_v2's text follows.
TAG scout-card K2 v2 (2026-10-12): = v1 + atoms kb10 / kb25 / kb43 (request field kb: b <- b - kb (m - host), needs worker_sc4.py, which
is worker_sc3 + that field; run every K2 arm on worker_sc4). v1's text follows.
TAG scout-card K2 (2026-10-12): forward arms at the S50 point for worker_sc3.py (card copy otr_m_sc3 = otr_m_v5 + K4L ages 3 and 4 under the
families k4la3 / k4la4 + the b-fix gamma field 'bfixg'), paired against the S51 round's base evals on fresh A41. Each request = the trainer's own
a_RS51_base request for that seed + 'pl': {'k4la3': -1 x 16, 'k4la4': -1 x 16} (ages 3-4 of K4L off = otr_m_v5 bit for bit) + the arm's changes.
An arm is a '+'-joined list of the atoms in ATOMS (e.g. 'gk+a3k4l7' = both), so combined arms need no new code.
usage: python make_k2_v1.py <run dir> <tag> <first seed> <count> <arm> [<arm> ...]   (ids sc_<tag>_r<seed>_<arm>; arm 'ctl' gets tag k0)"""
import glob
import json
import sys
from pathlib import Path

O = Path(__file__).resolve().parents[1]
BASE = O / 'trainer' / 'run_RAD_S51_gen'
NKL3 = 0.727736          # S50 point's stored nkl3 shift at layers 1-14 (read_pt.py on cert/RAD/S50/point_v1/params.pt)


def pbase(**kw):
    return lambda req: next(m for m in req['ncar'] if m.get('module') == 'pbase').update(kw)


def k31l_age4(s):
    """K31L4 strength s on the pairs aged 4 only (new instance, leaf nkl3, base s - NKL3; the age-3 instance's base lowered by s)."""
    def edit(req):
        a3 = [m for m in req['ncar'] if m.get('module') == 'k31ly' and m.get('age') == 3]
        assert len(a3) == 1 and a3[0]['k31'] == 0.0 and a3[0]['leaf'] == 'nkl3', a3
        a3[0]['k31'] = -s
        req['ncar'].insert(req['ncar'].index(a3[0]) + 1, {'module': 'k31ly', 'age': 4, 'k31': s - NKL3, 'leaf': 'nkl3'})
    return edit


def k31l_age4_band(s, lo, hi):
    """K31L4 strength s on the pairs aged 4 only, and only at layers lo..hi (see the module docstring)."""
    def edit(req):
        a3 = [m for m in req['ncar'] if m.get('module') == 'k31ly' and m.get('age') == 3]
        assert len(a3) == 1 and a3[0]['k31'] == 0.0 and a3[0]['leaf'] == 'nkl3', a3
        req['ncar'].insert(req['ncar'].index(a3[0]) + 1, {'module': 'k31ly', 'age': 4, 'k31': 0.0, 'leaf': 'nkbk'})
        band = [s if lo <= l <= hi else 0.0 for l in range(16)]
        req['pl'].update(nkbk=band, nkl3=[-x for x in band])
    return edit


def k4l(age, c):
    return lambda req: req['pl'].update({'k4la%d' % age: [c - 1.0] * 16})     # K4L strength of that age's pairs = 1 + shift


ATOMS = dict(ctl=lambda req: None,
             gk=pbase(gsk=1.0, gk4=1.0), gsk=pbase(gsk=1.0), gk4=pbase(gk4=1.0), gkh=pbase(gsk=0.5, gk4=0.5),
             gb75=pbase(gbk=0.75), gb15=pbase(gbk=1.5),
             xh=pbase(x1=1.6, x2=5.0), xq=pbase(x1=0.8, x2=2.5),
             a4k31s4=k31l_age4(0.4), a4k31s6=k31l_age4(0.6), a4k31s8=k31l_age4(0.8), a4k31s10=k31l_age4(1.0), a4k31b1114=k31l_age4_band(0.6, 11, 14), a4k31b1214=k31l_age4_band(0.6, 12, 14),
             k4lf34=lambda req: req['pl'].update(k4la3=[-1.0] * 15 + [0.0], k4la4=[-1.0] * 15 + [0.0]),
             k4lf34h=lambda req: req['pl'].update(k4la3=[-1.0] * 15 + [-0.5], k4la4=[-1.0] * 15 + [-0.5]),
             a3k4l5=k4l(3, 0.5), a3k4l7=k4l(3, 0.7), a3k4l10=k4l(3, 1.0), a4k4l5=k4l(4, 0.5), a4k4l7=k4l(4, 0.7),
             bsk6=pbase(bsk=0.6), bsk10=pbase(bsk=1.0),
             bfx15=lambda req: req.update(bfixg=0.15), bfx20=lambda req: req.update(bfixg=0.2),
             kb10=lambda req: req.update(kb=0.10), kb25=lambda req: req.update(kb=0.25), kb43=lambda req: req.update(kb=0.43),
             gkp=pbase(gsk=1.0, gk4=1.0, gbp=1), gkB=pbase(gsk=1.0, gk4=1.0, gbeta=1), gkpB=pbase(gsk=1.0, gk4=1.0, gbp=1, gbeta=1),
             gk05=pbase(gsk=0.5, gk4=0.5), gk15=pbase(gsk=1.5, gk4=1.5),
             a3k4l7e=lambda req: req['pl'].update(k4la3=[-1.0] + [-0.3] * 9 + [-1.0] * 6))


def private_cutsim(spec):
    """the trainer's cut-simulator spec re-pointed to O/scout_card/cutsim_sc1 (same file names; sources re-hashed; opts unchanged)."""
    import hashlib
    here = Path(__file__).resolve().parent / 'cutsim_sc1'
    assert Path(spec['module']).name == 'torch_cutsim_s46_v11.py', spec['module']
    sha = lambda f: hashlib.sha256((here / f).read_bytes()).hexdigest()
    new = dict(spec, module=str(here / 'torch_cutsim_s46_v11.py'), sha256=sha('torch_cutsim_s46_v11.py'), sources={f: sha(f) for f in spec['sources']})
    changed = sorted(f for f in spec['sources'] if spec['sources'][f] != new['sources'][f])
    assert changed == ['pbase_cutsim_v1.py'], changed          # only the private pbase differs from the engineer's simulator
    return new


def main(run, tag, first, count, arms):
    run = Path(run)
    for d in ('queue', 'claimed', 'results'):
        (run / d).mkdir(parents=True, exist_ok=True)
    made = 0
    for seed in range(first, first + count):
        src = glob.glob(str(BASE / 'claimed' / f'a_RS51_base_r{seed}.w*.json'))
        assert len(src) == 1, (seed, src)
        assert (BASE / 'results' / f'a_RS51_base_r{seed}.json').exists(), seed
        base = json.loads(Path(src[0]).read_text())
        base['cutsim'] = private_cutsim(base['cutsim'])
        assert 'pl' not in base and 'bfixg' not in base, base.get('pl')
        pb = [m for m in base['ncar'] if m.get('module') == 'pbase']
        assert len(pb) == 1 and set(pb[0]) == {'module', 'bsk'}, pb
        for arm in arms:
            req = json.loads(json.dumps(base))
            req['pl'] = {'k4la3': [-1.0] * 16, 'k4la4': [-1.0] * 16}
            for atom in arm.split('+'):
                ATOMS[atom](req)
            req['id'] = f"sc_{'k0' if arm == 'ctl' else tag}_r{seed}_{arm}"
            (run / 'queue' / (req['id'] + '.json')).write_text(json.dumps(req))
            made += 1
    print('requests written:', made)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
