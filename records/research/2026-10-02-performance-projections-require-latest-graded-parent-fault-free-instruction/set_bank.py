"""TAG opus-release: make a graded file THE BANK of the release stages (common.py reads calibration/bank.json): the file every
stage pairs a candidate with and whose platform record every projection scales. Use it when a new file has been graded with no
network lost: the nearest graded parent extrapolates least (three fires projected from S35 read -1.1, +0.2 and -2.1 s in mean
wall; the overhead per round trip of S36, S37 and S38 is 0.177-0.179 ms against S35's 0.183).
Writes calibration/bank_<name>_<sha12>_<sid>.json (never overwritten) and copies it to calibration/bank.json (the previous
bank stays in its own versioned file). The share of memory-bound work in the new bank's platform kernel time is carried over
from the previous bank's through the stage C record of the new file against it, when that record is given:
  w_new = w_old r_mem / (w_old r_mem + (1 - w_old) r_cpu).
Afterwards: store its quiet legs for quick.sh (a stage C run directory with P legs of this file, named in a new
calibration/bank_stage_c_paced_<UTC stamp>.txt) and its worker reference for stage D (make_reference.py <worker> <eval json>,
copied to calibration/bank_stage_d_paced.json).
usage: set_bank.py <name> <source.py> <eval_<id>.json> [<stage_c.json of this file against the bank in force>]"""
import json
from pathlib import Path
import shutil
import sys

from common import BANK, BANK_NAME, PLATFORM, R, sha, write_json
import project

name, source, record = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
evaluation = json.loads(record.read_text())['evaluation']
lost = [m['mlp_index'] for m in evaluation['results']['per_mlp'] if m.get('error_code') not in (None, 'SUCCESS')]
assert evaluation['state'] == 'completed' and not lost, f'not a clean graded run: state {evaluation["state"]}, lost {lost}'
w_mem = tuple(PLATFORM.get('w_mem') or project.W_MEM)
note = f'memory-bound share carried over unchanged from {BANK_NAME}: {w_mem}'
if len(sys.argv) > 4:
    stage = json.loads(Path(sys.argv[4]).read_text())
    assert stage['candidate_sha256'] == sha(source) and stage['parent_sha256'] == sha(BANK), 'not this file against the bank in force'
    r_mem, r_cpu = stage['projection']['ratio_proxy_memory'], stage['projection']['ratio_proxy_cpu']
    old, w_mem = w_mem, tuple(round(w*r_mem/(w*r_mem + (1 - w)*r_cpu), 3) for w in w_mem)
    note = (f'memory-bound share {old} of {BANK_NAME} carried through the local ratios of {Path(sys.argv[4]).parent.name} '
            f'(memory-bound x{r_mem:.4f}, CPU-bound x{r_cpu:.4f}): {w_mem}')
platform = project.platform_record(record, w_mem)
target = R/'calibration'/f"bank_{name}_{sha(source)[:12]}_{platform['sid']}.json"
write_json(target, dict(name=name, source=str(source), sha256=sha(source), evaluation=str(record), platform=platform,
                        previous_bank=dict(name=BANK_NAME, source=str(BANK), sid=PLATFORM['sid']), note=note))
shutil.copyfile(target, R/'calibration'/'bank.json')
print(f"the bank is now {name} (#{platform['sid']}, sha {sha(source)[:12]}): wall {platform['predict_mean']:.2f} s mean / "
      f"{platform['predict_max']:.2f} worst, kernel {platform['kernel_mean']:.2f}, overhead {platform['overhead_mean']:.3f}, residual "
      f"{platform['residual_mean']:.4f} / {platform['residual_max']:.4f} s; {note}; written {target.name} and calibration/bank.json")
