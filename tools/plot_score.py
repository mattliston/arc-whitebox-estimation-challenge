#!/usr/bin/env python3
"""Render the README score figure from archived measurements; no estimator runs.

Requires matplotlib. Writes graph/score.svg and _build/score_preview.png.
"""
import datetime as dt
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
subs = json.loads((ROOT / 'graph/graph.json').read_text())['submissions']
clean = [s for s in subs if s['status'] == 'GRADED' and not s['failed_networks'] and s.get('score_public')]
excluded = [s for s in subs if s not in clean]
time = lambda s: dt.datetime.fromisoformat(s['date'].replace('Z', '+00:00'))
clean.sort(key=time)
chain = [s for s in clean if s['best_so_far']]
best = min(clean, key=lambda s: s['score_public'])
designated = [next(s for s in clean if str(s['submission_id']) == sid) for sid in ['337619', '337641']]
url = lambda s: f"https://github.com/mattliston/arc-whitebox-estimation-challenge/blob/main/submissions/{s['submission_id']}/README.md"
paper, ink, muted, accent = '#10151c', '#e9edf2', '#9aa8b8', '#e7ab91'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none', 'svg.hashsalt':'arc-phase2-score'})
fig = plt.figure(figsize=(12, 5.6), facecolor=paper)
ax = fig.add_axes([.085, .25, .65, .53], facecolor=paper)
ax.set_yscale('log')
ax.scatter([time(s) for s in clean], [s['score_public'] for s in clean], s=14, color='#8290a2', alpha=.8, zorder=3, linewidths=0, urls=[url(s) for s in clean])
ax.step([time(s) for s in chain] + [time(subs[-1])], [s['score_public'] for s in chain] + [chain[-1]['score_public']], where='post', color=accent, linewidth=1.4, zorder=4)
ax.scatter([time(s) for s in chain], [s['score_public'] for s in chain], s=13, color=accent, zorder=5, linewidths=0, urls=[url(s) for s in chain])
ax.scatter([time(best)], [best['score_public']], s=38, facecolors=paper, edgecolors=accent, linewidths=1.3, zorder=6, urls=[url(best)])
ax.set_ylim(.9e-9, 9e-7)
ax.set_xlim(dt.datetime(2026, 8, 22, tzinfo=dt.timezone.utc), dt.datetime(2026, 10, 24, tzinfo=dt.timezone.utc))
ax.yaxis.set_major_locator(FixedLocator([1e-9, 1e-8, 1e-7]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0e}'.replace('e-0','e-')))
ax.yaxis.set_minor_locator(NullLocator())
ax.tick_params(axis='both', colors=muted, length=0, pad=10, labelsize=10)
ticks=[dt.datetime(2026,8,22),dt.datetime(2026,9,1),dt.datetime(2026,9,15),dt.datetime(2026,10,1),dt.datetime(2026,10,15),dt.datetime(2026,10,22)]
ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(ticks)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%-d %b',tz=dt.timezone.utc))
ax.grid(axis='y',color='#303c4b',linewidth=.65,zorder=0)
for spine in ax.spines.values(): spine.set_visible(False)
first=clean[0]
ax.annotate(f"First clean result\n{first['score_public']:.3g}", (time(first),first['score_public']), xytext=(11,1), textcoords='offset points', va='top',color=muted,fontsize=9,linespacing=1.5)
ax.annotate('Best recorded', (time(best),best['score_public']), xytext=(-15,17), textcoords='offset points',ha='right',color=accent,fontsize=10)
fail_ax=fig.add_axes([.085,.135,.65,.037],sharex=ax,facecolor=paper)
fail_ax.scatter([time(s) for s in excluded],[0]*len(excluded),marker='|',s=35,color='#6e7a8c',linewidths=1,urls=[url(s) for s in excluded])
fail_ax.set_ylim(-1,1);fail_ax.axis('off')
fig.text(.085,.105,f'{len(excluded)} failed or incomplete evaluations · excluded from the score curve',color=muted,fontsize=9)
fig.text(.085,.925,'Public submission scores',color=ink,fontsize=19,weight='medium')
fig.text(.085,.867,f'{len(subs)} archived submissions · {len(clean)} evaluations with no failed networks',color=muted,fontsize=11)
fig.text(.735,.80,'Lower is better · log scale',ha='right',color=muted,fontsize=9)
fig.lines.append(plt.Line2D([.78,.78],[.25,.78],transform=fig.transFigure,color='#354251',linewidth=.7))
fig.text(.82,.765,'DESIGNATED ESTIMATORS',color=muted,fontsize=9)
for s,y,label in [(designated[0],.655,'Best archived score'),(designated[1],.415,'Second designated source')]:
    fig.text(.82,y,f"#{s['submission_id']}",color=ink,fontsize=13,url=url(s))
    fig.text(.82,y-.060,f"{s['score_public']:.7e}".replace('e-0','e-'),color=accent,fontsize=16)
    fig.text(.82,y-.105,label,color=muted,fontsize=9)
fig.text(.085,.035,'Dots: clean evaluations     Line: best score so far     Dates: UTC',color=muted,fontsize=9)
fig.text(.965,.035,'Archived public results',ha='right',color=muted,fontsize=9)
fig.savefig(ROOT/'graph/score.svg',facecolor=paper,metadata={'Date':None,'Title':'ARC Phase 2 archived public submission scores','Description':'Logarithmic score versus date; lower is better. Failed or incomplete evaluations are shown separately. Designated estimators 337619 and 337641 are labeled.'})
svg_path=ROOT/'graph/score.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
fig.savefig(ROOT/'_build/score_preview.png',dpi=150,facecolor=paper)
print(f'Rendered {len(clean)} scored evaluations and {len(excluded)} excluded evaluations; exact values remain in the submission records.')
