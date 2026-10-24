"""Small read-only status; never reads target or prediction arrays."""
from pathlib import Path
import json
import shutil
HERE=Path(__file__).resolve().parent
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
guards=[v for f in HERE.glob('*_guard.json') if (v:=read(f)) is not None]
rows=sum(len((d/'rows.jsonl').read_text().splitlines()) for d in HERE.glob('capture_[0-9][0-9]') if (d/'rows.jsonl').exists())
out=dict(captured=rows,total=160,completed_capture_groups=len(list(HERE.glob('capture_*/complete.json'))),
         capture_complete=(HERE/'capture_complete.json').exists(),
         maximum_completed_guard_rss=max((g['max_rss_bytes'] for g in guards),default=0),
         maximum_added_swap=max((g['max_swap_growth_bytes'] for g in guards),default=0),
         failed_guards=[str(g['command']) for g in guards if not g['completed']],
         disk_free_gib=shutil.disk_usage(HERE).free/1024**3)
for mode in ('graph','control'):
    path=HERE/f'fit_{mode}'/'training.jsonl'
    if path.exists():
        lines=path.read_text().splitlines();out[f'{mode}_training_last']=json.loads(lines[-1]) if lines else None
    done=read(HERE/f'fit_{mode}'/'result.json')
    if done:out[f'{mode}_fit_complete']=dict(frozen_fit_mse=done['frozen_fit_mse'],baseline_fit_mse=done['baseline_fit_mse'])
result=read(HERE/'assessment.json')
if result:out['assessment']={k:result[k] for k in ('graph_vs_bank','control_vs_bank','graph_vs_control','passes_information_gate','decision')}
print(json.dumps(out,indent=2))
