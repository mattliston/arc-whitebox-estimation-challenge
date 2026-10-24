#!/usr/bin/env python3
"""TAG analyst A11 watch (lead 2026-10-10 03:40Z): three public signals the team must hear about at once; HTTP only, a few requests, seconds.
  1. a new flopscope release: PyPI (flopscope, flopscope-client, flopscope-server, whestbench) and GitHub AIcrowd/flopscope releases; the grader ran
     flopscope 0.12.1 / whestbench 0.16.1 on the 10-02 and 10-09 pages (runtime_environment), which were also the newest releases on 10-10 03:45Z;
  2. the forum (category 2991): any new topic, and any new post by staff (organizers) in any topic of the category (participants' posts: waiter log only);
  3. the challenge page: the selection of final submissions ('selection deadline', 'select your', 'nominate' with a date, a selection interface).
State in a11_watch_state.json; a change writes ONE line to O/TEAM_LOG.md ('WATCH: ...') and the line is printed; no change prints 'no change'.
usage: a11_watch.py   (called by a11_waiter_v3.sh every hour)"""
import json, re, html, time, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent; STATE = HERE / 'a11_watch_state.json'
UA = {'User-Agent': 'analyst-watch/1.0', 'Accept': 'application/json'}


def get(url, accept_json=True, timeout=40):
    req = urllib.request.Request(url, headers=UA if accept_json else {'User-Agent': UA['User-Agent']})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode('utf-8', 'replace')
    return json.loads(data) if accept_json else data


now = time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())
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
    d = get('https://discourse.aicrowd.com/c/white-box-estimation-challenge-2026/2991.json'); topics = d['topic_list']['topics']
    cur['forum_topics'] = {str(t['id']): t.get('last_posted_at') for t in topics}
    old = st.get('forum_topics') or {}
    for t in topics:
        tid = str(t['id'])
        if old and tid not in old: news.append(f'NEW forum topic {tid} "{t["title"][:100]}" (https://discourse.aicrowd.com/t/{tid})')
        elif old and t.get('last_posted_at') != old.get(tid):      # a new post: say who and whether staff
            try:
                td = get(f'https://discourse.aicrowd.com/t/{tid}.json'); p = td['post_stream']['posts'][-1]
                staff = p.get('staff') or p.get('admin') or p.get('moderator')
                txt = html.unescape(re.sub(r'\s+', ' ', re.sub('<[^>]+>', ' ', p['cooked']))).strip()[:300]
                (news if staff else quiet).append(f'{"ORGANIZER" if staff else "participant"} post in topic {tid} "{t["title"][:80]}" by {p["username"]} {p["created_at"][:16]}: {txt}')
            except Exception as e: errors.append(f'topic {tid}: {e}')
except Exception as e: errors.append(f'forum: {e}')
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
    line = f"[analyst] {now} WATCH (automatic, a11_watch.py): " + ' ; '.join(news)
    with open(O / 'TEAM_LOG.md', 'a') as f: f.write(line + '\n')
    print(line)
else:
    print(f'{now} watch: no change' + (f' (errors: {"; ".join(errors)[:300]})' if errors else '') + f' | flopscope {cur.get("pypi_flopscope")} whestbench {cur.get("pypi_whestbench")} gh {cur.get("gh_flopscope_tag")} | forum topics {len(cur.get("forum_topics") or {})}')
