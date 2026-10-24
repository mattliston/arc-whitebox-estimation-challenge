"""TAG scout-card K1 v2 (2026-10-12): forward arms at the S50 point, paired against the S51 round's base evals, for worker_sc1.py (card copy
otr_m_sc1 = otr_m_v5 + K4L age 3 under the family k4la3). Each request = the trainer's own a_RS51_base request for that seed
(trainer/run_RAD_S51_gen/claimed/a_RS51_base_r<seed>.w*.json: S50 frozen point, f4c, ss -1, hf 2, b6, law n6y + k31ly ages 1-3 + pbase bsk 0.3402,
cutsim v11, ssb, rdc 2.0) plus 'pl': {'k4la3': -1 x 16} (age 3 of K4L off = otr_m_v5 bit for bit) and ONE change per arm (ARMS).
The pairing base is trainer/run_RAD_S51_gen/results/a_RS51_base_r<seed>.json (read-only); arm 'ctl' must reproduce it bit for bit.
usage: python make_k1_v2.py <run dir> <first seed> <count> <arm> [<arm> ...]"""
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
    """K31L4 strength s on the pairs aged 4 only: a new instance (age limit 4, leaf nkl3, base s - NKL3 -> total s at layers 1-14) and the age-3
    instance's base lowered by s, so the cohorts aged 1-3 keep their strengths exactly."""
    def edit(req):
        a3 = [m for m in req['ncar'] if m.get('module') == 'k31ly' and m.get('age') == 3]
        assert len(a3) == 1 and a3[0]['k31'] == 0.0 and a3[0]['leaf'] == 'nkl3', a3
        a3[0]['k31'] = -s
        req['ncar'].insert(req['ncar'].index(a3[0]) + 1, {'module': 'k31ly', 'age': 4, 'k31': s - NKL3, 'leaf': 'nkl3'})   # beside its siblings, before pbase
    return edit


def k4l_age3(c):
    return lambda req: req['pl'].update(k4la3=[c - 1.0] * 16)       # K4L strength of age-3 pairs = 1 + shift


ARMS = dict(ctl=lambda req: None,
            gk=pbase(gsk=1.0, gk4=1.0), gb75=pbase(gbk=0.75), xh=pbase(x1=1.6, x2=5.0),
            a4k31s4=k31l_age4(0.4), a4k31s6=k31l_age4(0.6), a4k31s7=k31l_age4(0.73),
            a3k4l5=k4l_age3(0.5), a3k4l7=k4l_age3(0.7), a3k4l10=k4l_age3(1.0))


def main(run, first, count, arms):
    run = Path(run)
    for d in ('queue', 'claimed', 'results'):
        (run / d).mkdir(parents=True, exist_ok=True)
    made = 0
    for seed in range(first, first + count):
        src = glob.glob(str(BASE / 'claimed' / f'a_RS51_base_r{seed}.w*.json'))
        assert len(src) == 1, (seed, src)
        assert (BASE / 'results' / f'a_RS51_base_r{seed}.json').exists(), seed
        base = json.loads(Path(src[0]).read_text())
        assert 'pl' not in base, base.get('pl')
        pb = [m for m in base['ncar'] if m.get('module') == 'pbase']
        assert len(pb) == 1 and set(pb[0]) == {'module', 'bsk'}, pb
        for arm in arms:
            req = json.loads(json.dumps(base))
            req['pl'] = {'k4la3': [-1.0] * 16}
            ARMS[arm](req)
            req['id'] = f"sc_k{0 if arm == 'ctl' else 1}_r{seed}_{arm}"   # name order = serving order: the controls first, then arms interleaved by seed
            (run / 'queue' / (req['id'] + '.json')).write_text(json.dumps(req))
            made += 1
    print('requests written:', made)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
