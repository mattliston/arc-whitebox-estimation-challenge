"""TAG scout-card K18 v1 (2026-10-13 06:2xZ, lead: L15-only age extensions at the S51 + gk point, to avoid the S50 / S51 overlap question): requests
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


ATOMS = dict(s51ctl=lambda req: None, s51k4lf34h=k4lf(0.5), s51k4lf34h4=k4lf(0.4), s51k4lf34h6=k4lf(0.6))


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
