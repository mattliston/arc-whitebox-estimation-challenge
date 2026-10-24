"""TAG scout-card RB (2026-10-12, PREPARED for the lead's possible re-base, not used yet): requests at the S51 point + gk11 (= S53's config: the
trainer's run_RAD_S53_gen gradient request a_RS53g_r7100264, mode grad -> eval with keep_final, S51 point_v1 params + hu, pbase bsk / gsk / gk4)
on A41 seeds, with pl k4la3 = k4la4 = -1 (my card copies' extra K4L ages off), the atoms of make_k3_v7 (ctl = the re-based base itself), and the
cut simulator either the trainer's (default) or a scout private copy (--sim cutsim_sc<k>, its sources re-hashed).
usage: python make_rb_v1.py <run dir> <tag> <first seed> <count> [--sim cutsim_sc<k>] <arm> [<arm> ...]"""
import glob
import hashlib
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import make_k3_v7 as K                      # its ATOMS (pbase / k31ly / pl edits) apply to any request of this card

src = glob.glob(str(H.parent / 'trainer' / 'run_RAD_S53_gen' / 'claimed' / 'a_RS53g_r7100264.w*.json'))
assert len(src) == 1, src
TMPL = json.loads(Path(src[0]).read_text())
assert TMPL['params'].endswith('cert/RAD/S51/point_v1/params.pt')
pb = [m for m in TMPL['ncar'] if m.get('module') == 'pbase']
assert len(pb) == 1 and pb[0].get('gsk') == 1.0 and pb[0].get('gk4') == 1.0, pb


def private(spec, name):
    here = H / name
    sha = lambda f: hashlib.sha256((here / f).read_bytes()).hexdigest()
    return dict(spec, module=str(here / 'torch_cutsim_s46_v11.py'), sha256=sha('torch_cutsim_s46_v11.py'), sources={f: sha(f) for f in spec['sources']})


def main(argv):
    run, tag, first, count = Path(argv[0]), argv[1], int(argv[2]), int(argv[3]); rest = argv[4:]
    sim = None
    if rest and rest[0] == '--sim':
        sim, rest = rest[1], rest[2:]
    for d in ('queue', 'claimed', 'results'):
        (run / d).mkdir(parents=True, exist_ok=True)
    n = 0
    for seed in range(first, first + count):
        for arm in rest:
            req = json.loads(json.dumps(TMPL))
            req.update(rank=seed, mode='eval', keep_final=1); req.pop('scale', None)
            req['pl'] = {'k4la3': [-1.0] * 16, 'k4la4': [-1.0] * 16}
            if sim:
                req['cutsim'] = private(req['cutsim'], sim)
            for atom in arm.split('+'):
                if atom != 'gk':                 # gk is already in the S53 config
                    K.ATOMS[atom](req)
            req['id'] = f"sc_{tag}_r{seed}_{arm}"
            (run / 'queue' / (req['id'] + '.json')).write_text(json.dumps(req)); n += 1
    print('requests written:', n)


if __name__ == '__main__':
    main(sys.argv[1:])
