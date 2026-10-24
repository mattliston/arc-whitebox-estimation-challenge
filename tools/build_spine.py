"""Build graph/spine.json and submissions/SUBMISSIONS.md from the AIcrowd scan plus the frozen repo's fire records.
Read-only on the frozen repo. Usage: python3 -I tools/build_spine.py <scan.jsonl>"""
import json, sys, os, glob, re, hashlib, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
R='/Users/matt/arc-white-box-estimation'
P2=R+'/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
O=P2+'/opus_reexam_20261001_v1'
scan=[json.loads(l) for fn in sys.argv[1:] for l in open(fn)]
_seen={}
for r in scan: _seen[r['id']]=r
scan=list(_seen.values())
ours={r['id']:r for r in scan if r.get('who')=='mliston'}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
src={}   # id -> dict(tag, source_origin, source_sha, archive_origin, archive_sha, note)
# October: fire receipts + stage_a
for rp in glob.glob(O+'/fire/*/receipt.json'):
    try: rec=json.load(open(rp))
    except Exception: continue
    sid=rec.get('submission_id'); tag=rec.get('tag')
    if not isinstance(sid,int): continue
    e=dict(tag=tag, archive_sha=rec.get('sha256'))
    run=O+'/release/runs/'+tag
    sa=run+'/stage_a.json'
    if os.path.isfile(sa):
        try:
            d=json.load(open(sa)); c=d.get('candidate')
            if c and os.path.isfile(c): e['source_origin']=os.path.relpath(c,R); e['source_sha']=d.get('candidate_sha256') or sha(c)
        except Exception: pass
    if 'source_origin' not in e and os.path.isfile(run+'/payload/estimator.py'):
        e['source_origin']=os.path.relpath(run+'/payload/estimator.py',R); e['source_sha']=sha(run+'/payload/estimator.py')
    if os.path.isfile(run+'/submission.tar.gz'): e['archive_origin']=os.path.relpath(run+'/submission.tar.gz',R)
    try: e['description']=json.load(open(os.path.dirname(rp)+'/attempt.json')).get('description')
    except Exception: pass
    src[sid]=e
# September: fable_fire_v1 submit_output.txt
for so in glob.glob(P2+'/fable_fire_v1/fire_cand*/submit_output.txt'):
    txt=open(so,errors='replace').read(); m=re.search(r'submission id (\d{6})', txt)
    if not m: continue
    sid=int(m.group(1)); d=os.path.dirname(so)
    cands=[p for p in glob.glob(d+'/*.py')]
    if cands:
        c=sorted(cands)[0]; src.setdefault(sid, {}).update(tag=os.path.basename(d), source_origin=os.path.relpath(c,R), source_sha=sha(c))
# Codex banks: official_bank_*.json (source sha; locate file by sha among candidate files in codex dirs)
cand_by_sha={}
for p in glob.glob(P2+'/codex_resume_2026092*/**/*.py', recursive=True):
    if os.path.getsize(p)<200000: continue
    cand_by_sha.setdefault(sha(p), p)
for bp in glob.glob(P2+'/codex_resume_2026092*/official_bank_*.json'):
    try: d=json.load(open(bp))
    except Exception: continue
    sid=d.get('submission_id'); s=d.get('source_sha256')
    if isinstance(sid,int) and s:
        e=src.setdefault(sid,{}); e.setdefault('tag', os.path.basename(bp).replace('.json',''))
        if s in cand_by_sha: e['source_origin']=os.path.relpath(cand_by_sha[s],R); e['source_sha']=s
        else: e['source_sha']=s
        e['bank_record']=os.path.relpath(bp,R)

# --- ledger-driven matching for fires without receipts ---
arch_by_sha={}; payload_by_sha={}
for run in glob.glob(O+'/release/runs/*/'):
    a=run+'submission.tar.gz'
    if os.path.isfile(a): arch_by_sha[sha(a)]=run
    pe=run+'payload/estimator.py'
    if os.path.isfile(pe): payload_by_sha[sha(pe)]=pe
def by_prefix(table, pre):
    pre=pre.lower(); hits=[k for k in table if k.startswith(pre)]
    return table[hits[0]] if len(hits)==1 else None
led=open(R+'/autoresearch/SUBMISSION_LEDGER.md', errors='replace').read().splitlines()
for line in led:
    for m in re.finditer(r'#(3(?:28|29|3[0-7])\d{3})\b', line):
        sid=int(m.group(1))
        if sid not in ours or src.get(sid,{}).get('source_origin'): continue
        seg=line[m.end():m.end()+400]
        am=re.search(r'archive(?: sha256)?[ :]+([0-9a-f]{8,64})', seg) or re.search(r'archive([0-9a-f]{8,64})', seg)
        sm=re.search(r'source(?: sha256)?[ :]+([0-9a-f]{8,64})', seg) or re.search(r'source([0-9a-f]{8,64})', seg) or re.search(r'\(([0-9a-f]{8})\b', seg)
        e=src.setdefault(sid,{})
        if am:
            run=by_prefix(arch_by_sha, am.group(1))
            if run:
                e['archive_origin']=os.path.relpath(run+'submission.tar.gz',R); e['archive_sha']=[k for k in arch_by_sha if arch_by_sha[k]==run][0]; e.setdefault('tag', os.path.basename(run.rstrip('/')))
                pe=run+'payload/estimator.py'
                if os.path.isfile(pe): e['source_origin']=os.path.relpath(pe,R); e['source_sha']=sha(pe)
        if 'source_origin' not in e and sm:
            pe=by_prefix(payload_by_sha, sm.group(1))
            if pe:
                e['source_origin']=os.path.relpath(pe,R); e['source_sha']=sha(pe); run=os.path.dirname(os.path.dirname(pe))
                e.setdefault('tag', os.path.basename(run)); a=run+'/submission.tar.gz'
                if os.path.isfile(a): e['archive_origin']=os.path.relpath(a,R); e['archive_sha']=sha(a)
# --- September fires by file name from the ledger ---
name_hits={}
for line in led:
    for m in re.finditer(r'#(3(?:28|29|30|31)\d{3})\b[^\n]{0,120}?([A-Za-z0-9_./-]+\.py)', line):
        sid=int(m.group(1)); fn=os.path.basename(m.group(2))
        if sid in ours and not src.get(sid,{}).get('source_origin') and sid not in name_hits: name_hits[sid]=fn
for sid,fn in name_hits.items():
    found=[p for p in glob.glob(P2+'/**/'+fn, recursive=True)]
    if not found: found=[p for p in glob.glob(R+'/pure_closure_writeup/**/'+fn, recursive=True)]
    if found:
        c=sorted(found, key=len)[0]; src.setdefault(sid,{}).update(source_origin=os.path.relpath(c,R), source_sha=sha(c), tag=src.get(sid,{}).get('tag') or fn)


# --- manual overrides resolved from the ledger text (paths verified) ---
MANUAL={
 333429: dict(tag='S36_candidate_T_C_B_F_D_Hfull_Phfull_v1_Rlvfuse_sized32_Rtsr_off', source_origin='pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py'),
 333435: dict(tag='S37_candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off', source_origin='pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py'),
 336570: dict(tag='stkvanob_20261016_opus_v1', source_origin='pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_stkvanob_20261016_opus_v1/payload/estimator.py', archive_origin='pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/runs/a_stkvanob_20261016_opus_v1/submission.tar.gz'),
}
for sid,m in MANUAL.items():
    e=src.setdefault(sid,{})
    for k,v in m.items():
        if k.endswith('_origin'):
            full=os.path.join(R,v)
            if os.path.isfile(full): e[k]=v; e[k.replace('_origin','_sha')]=sha(full)
        else: e[k]=v

# Reviewed source overrides take precedence over heuristic filename matches.
source_overrides = json.load(open(ROOT / 'tools/submission_sources.json'))
for sid_text, entry in source_overrides.items():
    sid = int(sid_text)
    e = src.setdefault(sid, {})
    for kind in ('source', 'archive'):
        origin = entry.get(kind + '_origin')
        if origin:
            full = os.path.join(R, origin)
            actual = sha(full)
            expected = entry.get(kind + '_sha256')
            if expected and actual != expected:
                raise ValueError(f'Reviewed {kind} changed for #{sid}: {origin}')
            e[kind + '_origin'] = origin
            e[kind + '_sha'] = actual
    e['source_evidence'] = entry.get('evidence')
# Mining recovers earlier sources and receipts that were absent from the scan.
for fn in sorted(glob.glob(str(ROOT / '_build/mine/*.jsonl'))):
    for line in open(fn):
        if not line.strip(): continue
        node = json.loads(line)
        if node.get('kind') != 'submission' or len(node.get('submission_ids', [])) != 1: continue
        sid = node['submission_ids'][0]
        if sid not in ours: continue
        e = src.setdefault(sid, {})
        for role in ('source', 'archive'):
            if e.get(role + '_origin'): continue
            for f in node.get('files', []):
                if f.get('role') != role: continue
                origin = f.get('origin', '')
                full = os.path.join(R, origin)
                if os.path.isfile(full):
                    e[role + '_origin'] = origin; e[role + '_sha'] = sha(full)
                    break
        e.setdefault('tag', node.get('title'))
# Every located source must agree with its recorded digest.
for sid, e in src.items():
    for kind in ('source', 'archive'):
        origin, expected = e.get(kind + '_origin'), e.get(kind + '_sha')
        if origin and expected and sha(os.path.join(R, origin)) != expected:
            if kind == 'archive' and expected in arch_by_sha:
                e['archive_origin'] = os.path.relpath(arch_by_sha[expected] + 'submission.tar.gz', R)
                e['archive_location_note'] = 'Located by receipt SHA256; original tag directory held different archive bytes.'
            else:
                raise ValueError(f'{kind} digest mismatch for #{sid}: {origin}')

# assemble spine
spine=[]; best=None
for sid in sorted(ours):
    r=ours[sid]; agg=r.get('agg') or {}; e=src.get(sid,{})
    entry=dict(submission_id=sid, date=r.get('created'), status=r.get('status'), score_public=r.get('score') if r.get('status')=='GRADED' else None,
               n=r.get('n'), failed_networks=len(r.get('failed') or []), utilization=agg.get('mean_compute_utilization'),
               wall_mean_s=(r.get('wall') or [None,None])[0], wall_max_s=(r.get('wall') or [None,None])[1],
               resid_mean_s=(r.get('resid') or [None,None])[0], resid_max_s=(r.get('resid') or [None,None])[1],
               tag=e.get('tag'), source_origin=e.get('source_origin'), source_sha256=e.get('source_sha'), archive_origin=e.get('archive_origin'), archive_sha256=e.get('archive_sha'), source_evidence=e.get('source_evidence'), archive_location_note=e.get('archive_location_note'),
               bank_record=e.get('bank_record'), description=e.get('description'), node=None, best_so_far=False)
    s=entry['score_public']
    if s is not None and not entry['failed_networks'] and (best is None or s<best): best=s; entry['best_so_far']=True
    entry['best_at_time']=best
    spine.append(entry)
os.makedirs('graph',exist_ok=True); os.makedirs('submissions',exist_ok=True)
json.dump(dict(built_at=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), count=len(spine), spine=spine), open('graph/spine.json','w'), indent=1)
with_src=sum(1 for e in spine if e['source_origin']); print('spine entries', len(spine), 'with source file', with_src, 'graded', sum(1 for e in spine if e['status']=='GRADED'), 'failed', sum(1 for e in spine if e['status']=='FAILED'))
print('missing source for ids:', [e['submission_id'] for e in spine if not e['source_origin']])

# --- SUBMISSIONS.md ---
lines=["# Phase 2 submissions", "", "Every Phase 2 submission of ours, oldest first, with the official public score as graded by AIcrowd (score = mean over the public networks of final-layer MSE x utilisation; lower is better). 'best' marks a new best at the time. Each directory holds the file exactly as submitted.", "", "| # | date (UTC) | id | status | public score | best | utilisation | wall max (s) | residual max (s) | failed networks | submitted file |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for i,e in enumerate(spine,1):
    sc=str(e['score_public']) if e['score_public'] else ('FAILED' if e['status']=='FAILED' else '-')
    srcp=('[estimator.py](%d/estimator.py)'%e['submission_id']) if e['source_origin'] else '(not located)'
    lines.append("| %d | %s | [#%d](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/%d) | %s | %s | %s | %s | %s | %s | %s | %s |"%(i, (e['date'] or '')[:16].replace('T',' '), e['submission_id'], e['submission_id'], e['status'], sc, '*' if e['best_so_far'] else '', ('%.4f'%e['utilization']) if e['utilization'] else '-', e['wall_max_s'] if e['wall_max_s'] is not None else '-', e['resid_max_s'] if e['resid_max_s'] is not None else '-', e['failed_networks'], srcp))
open('submissions/SUBMISSIONS.md','w').write("\n".join(lines)+"\n")
print('SUBMISSIONS.md rows', len(spine))
