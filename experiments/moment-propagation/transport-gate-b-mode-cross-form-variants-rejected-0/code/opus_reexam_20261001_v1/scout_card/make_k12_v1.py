"""TAG scout-card K12 v1 (2026-10-12, lead backlog v1: F4C age 3): = make_k2_v2 for worker_sc9.py (card copy otr_m_sc5: F4C ages 1-3 with the
families f4a3 / x1a3 / x2a3; the engineer's simulator as in the trainer's request). Every request also carries pl f4a3 = -0.6, x1a3 = x2a3 = 0
(age 3 of F4C OFF = otr_m_sc4); atom f4c3 switches age 3 on at the S50 age-2 strengths (f4a3 = f4a2's stored 0.0651778, x2a3 = x2a2's
stored 0.258157 at L0-14, x1a3 = 0). make_k2_v2's text follows.
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
             f4c3=lambda req: req['pl'].update(f4a3=[0.0651778] * 15 + [0.0], x2a3=[0.258157] * 15 + [0.0], x1a3=[0.0] * 16))


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
        assert 'pl' not in base and 'bfixg' not in base, base.get('pl')
        pb = [m for m in base['ncar'] if m.get('module') == 'pbase']
        assert len(pb) == 1 and set(pb[0]) == {'module', 'bsk'}, pb
        for arm in arms:
            req = json.loads(json.dumps(base))
            req['pl'] = {'k4la3': [-1.0] * 16, 'k4la4': [-1.0] * 16, 'f4a3': [-0.6] * 16, 'x1a3': [0.0] * 16, 'x2a3': [0.0] * 16}
            for atom in arm.split('+'):
                ATOMS[atom](req)
            req['id'] = f"sc_{'k0' if arm == 'ctl' else tag}_r{seed}_{arm}"
            (run / 'queue' / (req['id'] + '.json')).write_text(json.dumps(req))
            made += 1
    print('requests written:', made)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
