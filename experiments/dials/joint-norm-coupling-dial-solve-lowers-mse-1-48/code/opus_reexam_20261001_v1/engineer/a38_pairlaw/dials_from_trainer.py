"""TAG opus_engineer a38: the trainer's hand-over of the pair law's dials (O/trainer/candidates/<name>/physics_dials.json, entry
pair_law_dials) as a dials file of this lane's lever (a38_pairlaw/dials/<name>.json, literal names of fragment_np_v8.py):
  STRENGTH  the fold strength per layer 0..15 (0 at layers 0 and 15: the fold acts at 1-14)
  DSCALE    the factor on the removed multiple of C (one number when it is the same at layers 1-14, else 16 values)
  BIRTHS, ROWS, DIAG   the three source strengths
  K3L       the young pairs' plane's strength by cohort age (age 1 first; trailing zeros dropped)
  FORMS 3, BETA true, DFIT = the third argument (2: the table of multiples, 3: the constant regressor; default 3)
usage: dials_from_trainer.py <physics_dials.json> <name> [dfit]"""
import json
import sys
from pathlib import Path
E = Path(__file__).resolve().parents[1]
src = json.loads(Path(sys.argv[1]).read_text())
d = src['pair_law_dials']
name = sys.argv[2]; dfit = int(sys.argv[3]) if len(sys.argv) > 3 else 3
strength = [float(x) for x in d['fold_strength_per_layer']]
factor = [float(x) for x in d['factor_on_removed_multiple_of_C_per_layer']]
assert len(strength) == 16 and len(factor) == 16
ages = d.get('k31l_strength_per_cohort_age', {})
k3l = [float(ages[str(a)]) for a in range(1, 1 + max((int(a) for a in ages), default=0))]
while k3l and not k3l[-1]:
    k3l.pop()
out = dict(STRENGTH=[0.0] + strength[1:15] + [0.0], DSCALE=(factor[1] if len(set(factor[1:15])) == 1 else factor),
           BIRTHS=float(d['births_source_strength']), ROWS=float(d['rows_source_strength']), DIAG=float(d['diag_source_strength']),
           FORMS=3, DFIT=dfit, BETA=True, K3L=(k3l if k3l else 0.0))
path = E/'a38_pairlaw/dials'/(name + '.json')
assert not path.exists(), path
path.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(dict(written=str(path), parameters=src.get('parameters'), card=src.get('candidate'), card_sha256=src.get('sha256'), levers=src.get('builder_levers'), dials=out)))
