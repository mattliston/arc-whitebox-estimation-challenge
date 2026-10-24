#!/usr/bin/env python3
"""Write concise entry points; experiment directories contain the actual programs."""
import collections
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def md(s):return str(s).replace('|','\\|').replace('[','\\[').replace(']','\\]').replace('\n',' ')
def target(n):return n['directory']+'/'+n.get('document','README.md')
def rel(directory,path):return os.path.relpath(ROOT/path,ROOT/directory)
def write(path,lines):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(lines)+'\n')
def main():
    g=json.loads((ROOT/'graph/graph.json').read_text());nodes={n['id']:n for n in g['nodes']};components={c['id']:c for c in g['components']}
    relations=collections.defaultdict(list)
    for e in g['edges']:
        relations[e['from']].append((e['kind'].replace('_',' '),e['to']))
        relations[e['to']].append(('← '+e['kind'].replace('_',' '),e['from']))
    subs={s['submission_id']:s for s in g['submissions']}
    inventory=[]
    for n in g['nodes']:
        d=n['directory'];programs=[f for f in n['files'] if f['path'].endswith(('.py','.sh'))];other=[f for f in n['files'] if f not in programs]
        L=['# '+n['title'],'',n['date']+' · '+n['kind']+' · '+n['outcome']+' · '+n['hedge'],'',n['claim'],'']
        if n.get('code_limitations'): L += ['Code availability: '+str(n['code_limitations']),'']
        if n.get('measurement'):L+=['Recorded measurement:','','    '+str(n['measurement']).replace('\n','\n    '),'']
        if n['kind']=='submission':
            s=subs[n['submission_ids'][0]];L += [f"Public score: {s.get('score_public')}. Status: {s['status']}. Failed networks: {s['failed_networks']}.",'',f"[Run or rebuild the designated estimators]({rel(d,'estimator/REPRODUCE.md')}).",'']
        elif n['kind']=='experiment':
            if n.get('code_root'):
                L += [f"Program source is in [`code/`](code/), preserving the relative layout below `{n['code_root']}`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators]({rel(d,'estimator/REPRODUCE.md')}); [research data availability]({rel(d,'data/DATA_INDEX.md')}).",'']
            else:L += ['The measured result is retained here. A separate executable program was not identified; this entry is indexed as a result record rather than a code directory.','']
        if programs:
            L += ['## Programs','']
            # Actual file links remain available even when a program bundle has many helpers.
            for f in programs[:12]:L.append(f"- [{md(rel(d,f['path']))}]({rel(d,f['path'])})")
            if len(programs)>12:
                L += ['','<details>',f'<summary>Additional programs and dependencies ({len(programs)-12})</summary>','']
                L += [f"- [{md(rel(d,f['path']))}]({rel(d,f['path'])})" for f in programs[12:]]
                L += ['','</details>']
            L.append('')
        if n.get('evidence'):
            L += ['## Evidence','']+[f"- [{md(Path(e.split('#')[0]).name + ('#'+e.split('#')[1] if '#' in e else ''))}]({rel(d,e.split('#')[0])}{'#'+e.split('#')[1] if '#' in e else ''})" for e in n['evidence']]+['']
        if other:
            L += ['<details>',f'<summary>All result files and records ({len(other)})</summary>','']
            L += [f"- [{md(Path(f['path']).name)}]({rel(d,f['path'])}) · {f['role']}" for f in other]
            L += ['','</details>','']
        if relations[n['id']]:L+=['## Related work','']+[f"- {kind}: [{md(nodes[t]['title'])}]({rel(d,target(nodes[t]))})" for kind,t in relations[n['id']]]+['']
        L += [f"[Research graph]({rel(d,'graph/index.html')}#node={n['id']}) · [Copy provenance]({rel(d,'PROVENANCE.json')})",'']
        write(target(n),L)
        if n['kind']=='experiment':inventory.append(dict(id=n['id'],title=n['title'],directory=d,programs=len(programs),python_files=sum(f['path'].endswith('.py') for f in programs),code_root=n.get('code_root'),status=n.get('code_status'),limitations=n.get('code_limitations')))
    top=['# Experiments','','Actual research programs, grouped by estimator component. Each code directory includes its Python files, recovered local dependencies, results and an evidence-linked README. Files retain their original bytes; historical programs can require machine paths and datasets listed in the [data index](../data/DATA_INDEX.md).','','For a verified runnable entry point, see [estimator reproduction](../estimator/REPRODUCE.md). Findings, lessons and result-only records are in the [research index](../records/README.md).','','| Component | Experiments with programs | Result records |','|---|---:|---:|']
    record_index=['# Research records','','Findings, decisions, lessons, incidents and measured results whose executable program was not identified. The [experiments index](../experiments/README.md) contains actual program bundles. The [interactive viewer](../graph/index.html) searches both.','','| Date | Kind | Record |','|---|---|---|']
    for c in g['components']:
        ns=[n for n in g['nodes'] if n['kind']=='experiment' and n['component']==c['id']]
        if not ns:
            old=ROOT/'experiments'/c['id']/'README.md'
            if old.is_file():
                dest=ROOT/'_build/packaging_before/retired'/old.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);old.replace(dest)
            try:old.parent.rmdir()
            except OSError:pass
            continue
        d='experiments/'+c['id'];code=[n for n in ns if n.get('code_root')];only=[n for n in ns if not n.get('code_root')]
        top.append(f"| [{c['title']}]({c['id']}/README.md) | {len(code)} | {len(only)} |")
        L=['# '+c['title'],'',c['what'],'','| Date | Experiment | Programs | Outcome |','|---|---|---:|---|']
        for n in ns:
            count=sum(f['path'].endswith(('.py','.sh')) for f in n['files'])
            L.append(f"| {n['date'][:10]} | [{md(n['title'])}]({rel(d,target(n))}) | {count if n.get('code_root') else 'result only'} | {n['outcome']} |")
        write(d+'/README.md',L)
    for n in g['nodes']:
        if n['directory'].startswith('records/research/'):
            record_index.append(f"| {n['date'][:10]} | {n['kind']} | [{md(n['title'])}]({rel('records',target(n))}) |")
    write('experiments/README.md',top);write('records/README.md',record_index)
    tools=[n for n in g['nodes'] if n['kind']=='tool' and n.get('code_root')]
    write('instruments/README.md',['# Research instruments','','Archived scorers, runners, builders and audit programs.','','| Date | Instrument |','|---|---|']+[f"| {n['date'][:10]} | [{md(n['title'])}]({rel('instruments',target(n))}) |" for n in tools])
    (ROOT/'graph/code_inventory.json').write_text(json.dumps(inventory,indent=1)+'\n')
    print('Research documents:',len(g['nodes']),'experiment program bundles:',sum(bool(n['code_root']) for n in inventory),'result-only:',sum(not n['code_root'] for n in inventory))

if __name__=='__main__':main()
