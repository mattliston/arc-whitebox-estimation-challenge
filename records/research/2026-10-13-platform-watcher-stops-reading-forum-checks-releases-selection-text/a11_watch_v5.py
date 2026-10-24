#!/usr/bin/env python3
"""TAG analyst A11 watch v5 (2026-10-13 02:1xZ: v4 WITHOUT the forum, Matt 10-12: nothing on the forum is worth reading, never fetch forum topics).
v4 text: (lead 2026-10-10 03:40Z): three public signals the team must hear about at once; HTTP only, a few requests, seconds.
  1. a new flopscope release: PyPI (flopscope, flopscope-client, flopscope-server, whestbench) and GitHub AIcrowd/flopscope releases; the grader ran
     flopscope 0.12.1 / whestbench 0.16.1 on the 10-02 and 10-09 pages (runtime_environment), which were also the newest releases on 10-10 03:45Z;
  2. (v5: the forum is NOT read any more);
  3. the challenge page: the selection of final submissions ('selection deadline', 'select your', 'nominate' with a date, a selection interface).
State in a11_watch_state.json; a change writes ONE line to O/TEAM_LOG.md ('WATCH: ...') and the line is printed; no change prints 'no change'.
usage: a11_watch_v5.py   (called by a11_waiter_v5.sh every hour)"""
import json, re, html, time, urllib.request, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent; STATE = HERE / 'a11_watch_state.json'
UA = {'User-Agent': 'analyst-watch/1.0', 'Accept': 'application/json'}


def get(url, accept_json=True, timeout=40):
    req = urllib.request.Request(url, headers=UA if accept_json else {'User-Agent': UA['User-Agent']})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode('utf-8', 'replace')
    return json.loads(data) if accept_json else data


now = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%MZ'], text=True).strip()
st = json.loads(STATE.read_text()) if STATE.exists() else {}
cur, news, quiet, errors = {}, [], [], []                     # quiet: participants' posts, printed to the waiter's log only
for pkg in ('flopscope', 'flopscope-client', 'flopscope-server', 'whestbench'):
    try:
        d = get(f'https://pypi.org/pypi/{pkg}/json'); v = d['info']['version']
        up = min((f['upload_time_iso_8601'] for f in d['releases'].get(v, [])), default='')
        cur[f'pypi_{pkg}'] = v
        if st.get(f'pypi_{pkg}') and st[f'pypi_{pkg}'] != v: news.append(f'NEW {pkg} {v} on PyPI (uploaded {up[:16]}; was {st[f"pypi_{pkg}"]})')
    except Exception as e: errors.append(f'pypi {pkg}: {e}')
try:
    rel = get('https://api.github.com/repos/AIcrowd/flopscope/releases?per_page=3'); tag = rel[0]['tag_name'] if rel else ''
    cur['gh_flopscope_tag'] = tag
    if st.get('gh_flopscope_tag') and st['gh_flopscope_tag'] != tag:
        body = re.sub(r'\s+', ' ', rel[0].get('body') or '')[:400]
        news.append(f'NEW flopscope release {tag} on GitHub ({rel[0].get("published_at", "")[:16]}): {body} (changelog: https://github.com/AIcrowd/flopscope/blob/main/CHANGELOG.md)')
except Exception as e: errors.append(f'github: {e}')
try:
    s = html.unescape(html.unescape(get('https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026', accept_json=False)))
    s = s.replace('\\x20', ' ').replace('\\x27', "'")
    hits = sorted(set(re.sub(r'\s+', ' ', m.group(0)) for m in re.finditer(r'[^.<>]{0,120}(?:[Ss]election deadline|select your|[Ss]elect (?:your )?final|[Nn]ominate)[^.<>]{0,160}', s)))
    cur['page_selection_text'] = hits
    if st.get('page_selection_text') is not None and hits != st['page_selection_text']:
        news.append('CHALLENGE PAGE selection text changed: ' + ' | '.join(h[:200] for h in hits)[:600])
except Exception as e: errors.append(f'page: {e}')
for k, v in st.items():
    cur.setdefault(k, v)                                   # keep what a failed request could not refresh
STATE.write_text(json.dumps(cur, indent=1))
for q in quiet: print(f'{now} {q}')
if news:
    line = f"[analyst] {now} WATCH (automatic, a11_watch_v4.py): " + ' ; '.join(news)
    with open(O / 'TEAM_LOG.md', 'a') as f: f.write(line + '\n')
    print(line)
else:
    print(f'{now} watch: no change' + (f' (errors: {"; ".join(errors)[:300]})' if errors else '') + f' | flopscope {cur.get("pypi_flopscope")} whestbench {cur.get("pypi_whestbench")} gh {cur.get("gh_flopscope_tag")}')
