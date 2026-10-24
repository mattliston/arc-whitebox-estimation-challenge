"""TAG scout-card K18 v7 (2026-10-13 15:1xZ): = v6 + the overlap atoms c_k31l75 / c_k4b90 / c_f4c85 / c_k4l90 (see the block above base_request). v6's text follows.
TAG scout-card K18 v6 (2026-10-13 14:5xZ, lead 14:4xZ: age caps at L1-14, the next age at 0.5 on top of C): = v5 + atoms c_a4k31 (C + the K31L age-4
cohort at 0.5 at read layers 8-14; c_a4k31e = the same at 8-11, the physicist's ARM S2: the band atom, k31ly age-4 instance on leaf nkbk, nkl3 lowered by the same amount), and for worker_sc9 (otr_m_sc5,
F4_AGE 3) c9base (C with F4C age 3 OFF: f4a3 = -0.6, x1a3 = x2a3 = 0 on every layer; the control) and c_f4c3 (C + F4C age 3 at HALF the S51 age-2
strengths at shift layers 8-14: theta3 = 0.5 x (0.6 + 0.0651778), x2a3 = 0.5 x 0.258157, x1a3 = 0; age 3 off elsewhere). v5's text follows.
TAG scout-card K18 v5 (2026-10-13 10:5xZ, lead 10:3xZ: re-screens of 10-01..10-10 closures ON TOP OF C): = v4 + atoms on C (K4L ages 3-4 at 0.5 at
L8-15 = the physicist's T141 arm C, pl only): cbase (C itself, the control vs his ph141 C results), c_e33 (+ family c2k = 0.0625 at L8-14: the lead's
E33 free C^2 part of the norm coupling at half his smallest 10-02 step; needs worker_sc11 = otr_m_sc7), c_x1a (+ x1a1 = x1a2 = 0.25 at L10-14: the
F4C exact S1 term, OFF in the card). v4's text follows.
TAG scout-card K18 v4 (2026-10-13 10:0xZ, lead 09:5xZ: lend both workers to the physicist's T139 extension): = v3 + atoms ph139a (K4L ages 3-4 at
strength 0.5 at L12-15, -1 elsewhere = the physicist's T139 arm A, t139_make.py) and ph139c (the same at L8-15 = his combination arm C); both
= the s51k4lf34h request with only pl changed, as in t139_make.py. v3's text follows.
TAG scout-card K18 v3 (2026-10-13 08:4xZ, lead 08:3xZ idle-worker screens: OFF components at a late band at a fractional strength): = v2 + atoms,
all on top of k4lf34h (the fired card): s51e2m05 / s51e34m05 (the L15 Edgeworth terms at strength -0.5: K19 measured +0.43% / +0.35% at +1,
per network roughly additive, i.e. mostly linear), s51a4b1214 (the K31L age-4 cohort at strength 1.0 at read layers 12-14 only: a new k31ly
age-4 instance on the zero-shift leaf nkbk, nkl3 lowered by the same amount there, so ages 1-3 keep their strengths; native = a per-layer age-4
entry, about 0.023% bill). v2's text follows.
TAG scout-card K18 v2 (2026-10-13 06:4xZ): = v1 + atoms for the L15-only Edgeworth terms ON TOP of k4lf34h (the fired card): s51e2 (family e2 = 1 at
L15: + k3^2 He4 phi / (72 sigma^5)), s51e34 (family e34 = 1 at L15: - k3 k4 He5 phi / (144 sigma^6); needs worker_sc10 = otr_m_sc6), s51e2e34 (both).
v1's text follows.
TAG scout-card K18 v1 (2026-10-13 06:2xZ, lead: L15-only age extensions at the S51 + gk point, to avoid the S50 / S51 overlap question): requests
at the TRUE S51 + gk11 point = the physicist's T134 learned-ON control request (O/physicist/t134_refit_screen/claimed/t134_a_ctl_on_<seed>.w*.json,
read only; for 7100584-591 they differ only in id / rank, so another seed = 7100584's with those replaced), keep_final 1, plus pl k4la3 = k4la4 =
-1 on every layer (K4L ages 3-4 off: my card copies otr_m_sc3+ then equal the trainer's otr_m_v5 bit for bit; checked on 584 / 585 at 05:31Z),
then the arm's pl edits. For worker_sc8 (or worker_sc9: its extra F4C-age-3 families stay at their 0 defaults unless an arm sets them).
Arms: s51ctl (nothing more), s51k4lf34h (K4L ages 3-4 at L15 at strength 0.5), s51k4lf34h4 / s51k4lf34h6 (strength 0.4 / 0.6).
usage: python make_k18_v1.py <run dir> <tag> <first seed> <count> <arm> [<arm> ...]   (ids sc_<tag>_r<seed>_<arm>)"""
import glob
import json
import sys
from pathlib import Path

O = Path(__file__).resolve().parents[1]
T134 = O / 'physicist' / 't134_refit_screen' / 'claimed'


def k4lf(strength):
    return lambda req: req['pl'].update(k4la3=[-1.0] * 15 + [strength - 1.0], k4la4=[-1.0] * 15 + [strength - 1.0])


def both(*fs):
    return lambda req: [f(req) for f in fs]


L15 = lambda family: (lambda req: req['pl'].update({family: [0.0] * 15 + [1.0]}))
def k31l_age4_band(s, lo, hi):
    """K31L4 strength s on the pairs aged 4 only, and only at layers lo..hi (= make_k3_v8's atom)."""
    def edit(req):
        a3 = [m for m in req['ncar'] if m.get('module') == 'k31ly' and m.get('age') == 3]
        assert len(a3) == 1 and a3[0]['k31'] == 0.0 and a3[0]['leaf'] == 'nkl3', a3
        req['ncar'].insert(req['ncar'].index(a3[0]) + 1, {'module': 'k31ly', 'age': 4, 'k31': 0.0, 'leaf': 'nkbk'})
        band = [s if lo <= l <= hi else 0.0 for l in range(16)]
        req['pl'].update(nkbk=band, nkl3=[-x for x in band])
    return edit


LM = lambda family, x: (lambda req: req['pl'].update({family: [0.0] * 15 + [x]}))
ATOMS = dict(s51ctl=lambda req: None, s51k4lf34h=k4lf(0.5), s51k4lf34h4=k4lf(0.4), s51k4lf34h6=k4lf(0.6),
             s51e2=both(k4lf(0.5), L15('e2')), s51e34=both(k4lf(0.5), L15('e34')), s51e2e34=both(k4lf(0.5), L15('e2'), L15('e34')),
             s51e2m05=both(k4lf(0.5), LM('e2', -0.5)), s51e34m05=both(k4lf(0.5), LM('e34', -0.5)), s51a4b1214=both(k4lf(0.5), k31l_age4_band(1.0, 12, 14)),
             ph139a=lambda req: req['pl'].update(k4la3=[-1.0] * 12 + [-0.5] * 4, k4la4=[-1.0] * 12 + [-0.5] * 4),
             ph139c=lambda req: req['pl'].update(k4la3=[-1.0] * 8 + [-0.5] * 8, k4la4=[-1.0] * 8 + [-0.5] * 8))
F4C3_OFF = lambda req: req['pl'].update(f4a3=[-0.6] * 16, x1a3=[0.0] * 16, x2a3=[0.0] * 16)
F4C3_HALF = lambda req: req['pl'].update(f4a3=[-0.6] * 8 + [0.5 * (0.6 + 0.0651778) - 0.6] * 7 + [-0.6], x1a3=[0.0] * 16, x2a3=[0.0] * 8 + [0.5 * 0.258157] * 7 + [0.0])
ATOMS.update(cbase=ATOMS['ph139c'],
             c_e33=both(ATOMS['ph139c'], lambda req: req['pl'].update(c2k=[0.0] * 8 + [0.0625] * 7 + [0.0])),
             c_x1a=both(ATOMS['ph139c'], lambda req: req['pl'].update(x1a1=[0.0] * 10 + [0.25] * 5 + [0.0], x1a2=[0.0] * 10 + [0.25] * 5 + [0.0])),
             c_a4k31=both(ATOMS['ph139c'], k31l_age4_band(0.5, 8, 14)),
             c_a4k31e=both(ATOMS['ph139c'], k31l_age4_band(0.5, 8, 11)),      # the physicist's ARM S2 (14:41Z): read layers 8-11 only
             c9base=both(ATOMS['ph139c'], F4C3_OFF), c_f4c3=both(ATOMS['ph139c'], F4C3_HALF))

# K18 v7 (2026-10-13 15:1xZ, lead 14:5xZ: with C on, do other fourth-order carriers now overlap C's per-unit k4 content?): scale-downs on top of C,
# as request pl shifts that ADD to the S51 point's stored values (read with engineer/a54_jmiv/read_pt.py from cert/RAD/S51/point_v1/params.pt).
S51 = dict(nkl1=-0.0126186, nkl2_8_10=-0.858385, nkl2_11_14=-0.733111, nkl3=0.727736, k4c=0.9595022755753744, k4=0.00831965,
           f4a1=0.104467, f4a2=0.0651778, x2a1=0.459128, x2a2=0.258157, k4la1_8_9=-0.126824, k4la1_10_14=-0.248603, k4la2=-0.274617)
band = lambda vals: [0.0] * 8 + list(vals) + [0.0]          # L8-14 (indices 8..14), 0 elsewhere


def k31l_scale(f):      # every k31ly instance's strength (base + stored) x f at read layers 8-14 (instances are cumulative, so every age x f)
    return lambda req: req['pl'].update(nkl1=band([(f - 1) * (0.0 + S51['nkl1'])] * 7),
                                        nkl2=band([(f - 1) * (1.0 + S51['nkl2_8_10'])] * 3 + [(f - 1) * (1.0 + S51['nkl2_11_14'])] * 4),
                                        nkl3=band([(f - 1) * (0.0 + S51['nkl3'])] * 7))


def k4band_scale(f):    # the per-unit k4 band dial (otr_pl 'k4': ctx['k4'] x (band + eps)) x f at L12-14 (band k4c) and L15 (1 + eps)
    return lambda req: req['pl'].update(k4=[0.0] * 12 + [(f - 1) * (S51['k4c'] + S51['k4'])] * 3 + [(f - 1) * (1.0 + S51['k4'])])


def f4c_scale(f):       # F4C ages 1-2: the rule's theta (0.6 + f4a) and the exact S2 strength x2a, x f at shift layers 8-14 (x1a stays 0)
    return lambda req: req['pl'].update(f4a1=band([(f - 1) * (0.6 + S51['f4a1'])] * 7), f4a2=band([(f - 1) * (0.6 + S51['f4a2'])] * 7),
                                        x2a1=band([(f - 1) * S51['x2a1']] * 7), x2a2=band([(f - 1) * S51['x2a2']] * 7))


def k4l12_scale(f):     # K4L ages 1-2 (strength 1 + k4la) x f at L8-14
    return lambda req: req['pl'].update(k4la1=band([(f - 1) * (1.0 + S51['k4la1_8_9'])] * 2 + [(f - 1) * (1.0 + S51['k4la1_10_14'])] * 5),
                                        k4la2=band([(f - 1) * (1.0 + S51['k4la2'])] * 7))


ATOMS.update(c_k31l75=both(ATOMS['ph139c'], k31l_scale(0.75)), c_k4b90=both(ATOMS['ph139c'], k4band_scale(0.9)),
             c_f4c85=both(ATOMS['ph139c'], f4c_scale(0.85)), c_k4l90=both(ATOMS['ph139c'], k4l12_scale(0.9)))


def base_request(seed):
    src = glob.glob(str(T134 / f't134_a_ctl_on_{seed}.w*.json'))
    if src:
        assert len(src) == 1, src
        return json.loads(Path(src[0]).read_text())
    ref = glob.glob(str(T134 / 't134_a_ctl_on_7100584.w*.json')); assert len(ref) == 1, ref
    return dict(json.loads(Path(ref[0]).read_text()), id=f't134_a_ctl_on_{seed}', rank=seed)


def main(run, tag, first, count, arms):
    run = Path(run)
    for d in ('queue', 'claimed', 'results'):
        (run / d).mkdir(parents=True, exist_ok=True)
    made = 0
    for seed in range(first, first + count):
        base = base_request(seed)
        assert base['rank'] == seed and 'pl' not in base and base['params'].endswith('S51/point_v1/params.pt'), (seed, base.get('pl'))
        for arm in arms:
            req = json.loads(json.dumps(base)); req['keep_final'] = 1
            req['pl'] = {'k4la3': [-1.0] * 16, 'k4la4': [-1.0] * 16}
            ATOMS[arm](req)
            req['id'] = f'sc_{tag}_r{seed}_{arm}'
            (run / 'queue' / (req['id'] + '.json')).write_text(json.dumps(req))
            made += 1
    print('requests written:', made)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
