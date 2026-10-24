"""[opus-hz7] reproduce the official120 stage-C projection of a normal boundary session against the stored S42 bank legs
(same math as stage_c_boundary: project.summarize + project.project with common.PLATFORM), plus P(loss) via ploss_v1.
usage (from O/release): project_vs_bank_v1.py <session C0 dir> [<session C0 dir> ...]"""
import json, sys
from pathlib import Path
R = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
sys.path.insert(0, str(R))
import project
from common import PLATFORM
from stage_c_boundary import load_session
sys.path.insert(0, str(R.parent/'lead/runs/opus_tree_v1'))
import ploss_v1
P = project.summarize([load_session(R/'runs/c_S42_bank_legs_heavy/P0')])
for d in sys.argv[1:]:
    s = load_session(Path(d)); c = project.summarize([s])
    pr = project.project(c, P, PLATFORM)
    rows = {r['rank']: r for r in s['rows']}
    first, warm = rows[35]['requests'], rows[100]['requests']
    print(json.dumps(dict(session=d, mean_wall_s=round(pr['mean_wall_s'], 2), worst_wall_s=round(pr['worst_wall_s'], 2),
                          worst_range=[round(pr['worst_wall_lo_s'], 2), round(pr['worst_wall_hi_s'], 2)], first=first, warm=warm,
                          p_loss_pct=round(100*ploss_v1.p(first, warm), 2), flops_35=rows[35]['flops'], flops_100=rows[100]['flops'],
                          residual_max_s=round(pr['residual_max_s'], 4), setup_last=s['setup']['pb_log'][-1])))
