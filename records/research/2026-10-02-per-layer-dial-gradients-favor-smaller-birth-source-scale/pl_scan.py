"""TAG opus-trainer (2026-10-02): forward scan of ONE physics dial family through the workers of a run (TRAINER_PL=1 workers).
For each shift d given, every layer of the family gets today's value + d (a global change of that dial); the final-layer MSE of
the given networks is compared with shift 0 on the same networks and the same parameters. Requests go ahead of the master's.
Prints pooled change, standard error of the mean per-network change and the number of networks that improved; writes
<run>/pl_scan_<name>.json.
usage: python -B pl_scan.py <run dir> <params.pt> <name (new)> <file of comma-separated ranks> <family> <shift> [<shift> ...]"""
import json
import sys
from pathlib import Path

import numpy as np

from otr_common import enqueue, publish, result_path, wait_results
from otr_pl import FAMILIES

RUN, PARAMS, NAME, FAMILY = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], sys.argv[5]
ranks = [int(x) for x in Path(sys.argv[4]).read_text().split(',')]
shifts = [0.0] + [float(x) for x in sys.argv[6:]]
assert FAMILY in FAMILIES and not (RUN / ('pl_scan_%s.json' % NAME)).exists()
requests = {d: [dict(id='a_pl_%s_%s_%g_r%04d' % (NAME, FAMILY, d, rank), mode='eval', rank=rank, params=PARAMS,
                     pl=({FAMILY: [d] * 16} if d else {})) for rank in ranks] for d in shifts}
for request in sum(requests.values(), []):
    enqueue(RUN, request)
wait_results(RUN, sum(requests.values(), []), lambda text: print(text, flush=True))
mse = {d: np.array([json.loads(result_path(RUN, request).read_text())['mse'] for request in items]) for d, items in requests.items()}
record = dict(name=NAME, family=FAMILY, params=PARAMS, ranks=ranks, points=[])
print('%s shifted on every layer, %d networks, parameters %s' % (FAMILY, len(ranks), PARAMS))
for d in shifts[1:]:
    change = 100 * (mse[d] / mse[0.0] - 1)
    record['points'].append(dict(shift=d, pooled_change_pct=100 * float(mse[d].mean() / mse[0.0].mean() - 1),
                                 standard_error_pct=float(change.std(ddof=1) / len(change) ** 0.5),
                                 better_on=int((mse[d] < mse[0.0]).sum()), mse=mse[d].tolist()))
    print('shift %+.3f: %+.3f%% (standard error %.3f%%), better on %d/%d' % (
        d, record['points'][-1]['pooled_change_pct'], record['points'][-1]['standard_error_pct'], record['points'][-1]['better_on'], len(ranks)))
record['base_mse'] = mse[0.0].tolist()
publish(RUN / ('pl_scan_%s.json' % NAME), lambda stream: json.dump(record, stream, indent=1))
