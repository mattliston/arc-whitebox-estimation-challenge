"""TAG opus-release: after a chain whose stage C ran the bank's legs beside the candidate's (--legs PCCP), compare those fresh
parent legs with the bank's STORED legs: every count must be equal (same file, platform clock), the kernel instructions without
page-fault work should agree to about 0.5%, and the candidate's projection must come out the same from either set. If it does,
the next chain can run stage C as two sessions (--legs CC --parent-from <stored legs>). No compute: reads finished sessions.
usage: compare_parent_legs.py <runs/c_<tag>> [<stored legs dir, default: the newest calibration/bank_stage_c_paced_*.txt>]"""
import json
from pathlib import Path
import sys

from common import PLATFORM, R
import project
from stage_c_boundary import load_session

run = Path(sys.argv[1]).resolve()
stored = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(
    sorted((R/'calibration').glob('bank_stage_c_paced_*.txt'), key=lambda p: p.stat().st_mtime)[-1].read_text().strip())
legs = lambda d, prefix: [load_session(x) for x in sorted(d.iterdir()) if x.is_dir() and x.name.startswith(prefix) and (x/'rows.jsonl').exists()]
fresh, old, cand = legs(run, 'P'), legs(stored, 'P'), legs(run, 'C')
assert fresh and old and cand, 'need P and C legs in the run and P legs in the stored directory'
free = lambda r: sum(project.fault_free(r).values())
print(f'fresh parent legs: {[Path(s["dir"]).name for s in fresh]} of {run.name}; stored: {[Path(s["dir"]).name for s in old]} of {stored.name}')
exact = ('requests', 'free_trips', 'freed_handles', 'peak_handles', 'flops', 'kernel_calls')
for a in fresh:
    for b in old[:1]:
        for x, y in zip(a['rows'], b['rows']):
            differ = [k for k in exact if x[k] != y[k]]
            print(f"  {Path(a['dir']).name} network {x['rank']}: counts {'all equal to the stored leg' if not differ else 'DIFFER: ' + ', '.join(differ)}; "
                  f"kernel instructions raw x{x['kernel_instructions']/y['kernel_instructions']:.4f}, without page-fault work x{free(x)/free(y):.4f} "
                  f"({x['server_minor_faults']:,} faults against {y['server_minor_faults']:,}); kernel seconds {x['kernel_s']:.1f} against {y['kernel_s']:.1f}")
c = project.summarize(cand)
for name, parent in (('the fresh parent legs of this window', fresh), ('the stored parent legs', old)):
    p = project.project(c, project.summarize(parent), PLATFORM)
    print(f"projection of the candidate from {name}: kernel {p['kernel_s']:.2f} s (proxy x{p['ratio_proxy']:.4f}), overhead {p['overhead_s']:.2f} s, "
          f"residual {p['residual_mean_s']:.4f} / {p['residual_max_s']:.4f} s, wall mean {p['mean_wall_s']:.2f} s, worst network {p['worst_wall_s']:.2f} s")
