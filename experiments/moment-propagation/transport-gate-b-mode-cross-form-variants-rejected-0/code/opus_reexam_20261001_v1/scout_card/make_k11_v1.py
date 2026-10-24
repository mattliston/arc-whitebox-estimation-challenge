"""TAG scout-card K11 v1 (2026-10-12): = make_k10_v1 re-pointed to O/scout_card/cutsim_sc6 (+ pbase gbs), plus atom gbsm05 (gbs = -0.5: the
birth leg's beta moved half-way in the OPPOSITE direction of gbeta = 1). make_k10_v1's text follows.
TAG scout-card K10 v1 (2026-10-12): = make_k9_v1 re-pointed to O/scout_card/cutsim_sc5 (the physicist's certified law file + pbase with bk4l),
plus atom bk4l1 (the birth density's LOCAL kurtosis term at 1). make_k9_v1's text follows.
TAG scout-card K9 v1 (2026-10-12, physicist v2 #1 lq / #4 gbu): = make_k8_v1 re-pointed to O/scout_card/cutsim_sc4 (pbase + gbu, the law n6y + lq;
both inert at 0) for worker_sc8.py, plus atoms lq1 (the law's corrected gate / density, n6y lq = 1) and gbu1 (pbase gbu = 1). make_k8_v1's text follows.
TAG scout-card K8 v1 (2026-10-12, radlab 22:0xZ budget: L12-15 first): = make_k7_v1 re-pointed to O/scout_card/cutsim_sc3 (per-layer gkL / bskL) for
worker_sc7.py, plus atoms gks (gk at 1.0 on L0-10, 1.5 on L11-14), bsklate (bsk 0.34 on L0-12, 1.0 on L13-14), bsksplit (0.34 on L0-5, 0 on L6-12, 1.0 on L13-14).
make_k7_v1's text follows.
TAG scout-card K7 v1 (2026-10-12): = make_k3_v3 re-pointed to the private simulator O/scout_card/cutsim_sc2 (pbase with the physicist's 21:5xZ
consistency keywords gbd / gbeta (1 full, 2 half) / gomit / bk4; inert at 0) for worker_sc6.py, plus atoms gbd1 gbd2 gbeta1 gbeta2 gomit bk4a (0.34) bk4b (1).
make_k3_v3's text follows.
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


def k4l(age, c):
    return lambda req: req['pl'].update({'k4la%d' % age: [c - 1.0] * 16})     # K4L strength of that age's pairs = 1 + shift


ATOMS = dict(ctl=lambda req: None,
             gk=pbase(gsk=1.0, gk4=1.0), gsk=pbase(gsk=1.0), gk4=pbase(gk4=1.0), gkh=pbase(gsk=0.5, gk4=0.5),
             gb75=pbase(gbk=0.75), gb15=pbase(gbk=1.5),
             xh=pbase(x1=1.6, x2=5.0), xq=pbase(x1=0.8, x2=2.5),
             a4k31s4=k31l_age4(0.4), a4k31s6=k31l_age4(0.6), a4k31s8=k31l_age4(0.8),
             a3k4l5=k4l(3, 0.5), a3k4l7=k4l(3, 0.7), a3k4l10=k4l(3, 1.0), a4k4l5=k4l(4, 0.5), a4k4l7=k4l(4, 0.7),
             bsk6=pbase(bsk=0.6), bsk10=pbase(bsk=1.0),
             bfx15=lambda req: req.update(bfixg=0.15), bfx20=lambda req: req.update(bfixg=0.2),
             kb10=lambda req: req.update(kb=0.10), kb25=lambda req: req.update(kb=0.25), kb43=lambda req: req.update(kb=0.43),
             gkp=pbase(gsk=1.0, gk4=1.0, gbp=1), gkB=pbase(gsk=1.0, gk4=1.0, gbeta=1), gkpB=pbase(gsk=1.0, gk4=1.0, gbp=1, gbeta=1),
             gk05=pbase(gsk=0.5, gk4=0.5), gk15=pbase(gsk=1.5, gk4=1.5),
             a3k4l7e=lambda req: req['pl'].update(k4la3=[-1.0] + [-0.3] * 9 + [-1.0] * 6),
             gbd1=pbase(gbd=1), gbd2=pbase(gbd=2), gbeta1=pbase(gbeta=1), gbeta2=pbase(gbeta=2), gomit=pbase(gomit=1),
             bk4a=pbase(bk4=0.34), bk4b=pbase(bk4=1.0),
             gks=pbase(gsk=1.0, gk4=1.0, gkL=[1.0] * 11 + [1.5] * 5),
             bsklate=pbase(bskL=[0.3402257055212658] * 13 + [1.0] * 3),
             bsksplit=pbase(bskL=[0.3402257055212658] * 6 + [0.0] * 7 + [1.0] * 3),
             lq1=lambda req: next(x for x in req['ncar'] if x.get('module') == 'n6y').update(lq=1.0),
             gbu1=pbase(gbu=1.0), bk4l1=pbase(bk4l=1.0), gbsm05=pbase(gbs=-0.5))


def private_cutsim(spec):
    """the trainer's cut-simulator spec re-pointed to O/scout_card/cutsim_sc1 (same file names; sources re-hashed; opts unchanged)."""
    import hashlib
    here = Path(__file__).resolve().parent / 'cutsim_sc6'
    assert Path(spec['module']).name == 'torch_cutsim_s46_v11.py', spec['module']
    sha = lambda f: hashlib.sha256((here / f).read_bytes()).hexdigest()
    new = dict(spec, module=str(here / 'torch_cutsim_s46_v11.py'), sha256=sha('torch_cutsim_s46_v11.py'), sources={f: sha(f) for f in spec['sources']})
    changed = sorted(f for f in spec['sources'] if spec['sources'][f] != new['sources'][f])
    assert changed == ['n6y_cutsim_v2.py', 'pbase_cutsim_v1.py'], changed          # only the private pbase and law differ from the engineer's simulator
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
