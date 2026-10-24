"""Source/JSON-only preparation; never imports NumPy, FNP, or kernel.py."""
from collections import Counter
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
CERT=P2/'astra_phase2_reexamination_v1/catalog_executable_cse_certificate_v1.json'
PLAN=H2/'teammate_s32_hk_residency_trace_v1/prospective_union_price_v1.json'
CERT_SHA='3bada45339d8adb6db19e2a68df41e65f5042985521738559a5d6645ece5a970'
PLAN_SHA='2e014b0fa100de3bcc622936f2def6c9f244a1ef79a72eab3a89a1a33a466ca6'


def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')


def scratch_peak(raw):
    """Independent lifetime count: allocate before retiring this node's inputs."""
    nodes=raw['nodes'];out={o['ref'] for o in raw['outputs']};live=set()
    def visit(i):
        if i in live:return
        live.add(i)
        for a in nodes[i]['args']:visit(a)
    for i in out:visit(i)
    uses=Counter(a for i in live for a in nodes[i]['args'])
    allocated=set();peak=0
    for i in sorted(live):
        if nodes[i]['op']!='input' and i not in out:allocated.add(i)
        peak=max(peak,len(allocated))
        for a in nodes[i]['args']:
            uses[a]-=1
            if not uses[a]:allocated.discard(a)
        if not uses[i]:allocated.discard(i)
    assert not allocated
    return peak


def main():
    assert pin(CERT)['sha256']==CERT_SHA and pin(PLAN)['sha256']==PLAN_SHA
    records=[r for r in json.loads(CERT.read_text())['records'] if r['rank']==15]
    assert len(records)==6 and all(r['passed'] for r in records)
    for r in records:
        assert r['name'].startswith('executable_catalog_2x3x3_')
        for c in r['circuits'].values():
            assert all(n['op'] in ('input','add','sub','neg') for n in c['nodes'])
            assert all(o['scale']=='1' for o in c['outputs'])
    records.sort(key=lambda r:r['name'])
    by_name={r['name']:r for r in records}
    peaks={r['name']:{s:scratch_peak(r['circuits'][s]) for s in ('L','R','P')} for r in records}
    plan=json.loads(PLAN.read_text());used={tuple(k) for r in plan['rows'] for k in r['used_positive_keys']}
    assert len(used)==84
    choices=[]
    for row in plan['table']:
        if tuple(row['key']) not in used:continue
        q=row['alternative'];old,m,k,n=row['key'];a,b,c=q['child'];M,K,N=q['padded'];p=peaks[q['name']]
        scratch=4*max(p['L']*a*b,p['R']*b*c,p['P']*a*c)
        padding=4*((M*K if (M,K)!=(m,k) else 0)+(K*N if (K,N)!=(k,n) else 0)+(M*N if (M,N)!=(m,n) else 0))
        workspace=q['tree_bytes']+q['packed_maps_bytes']+scratch+padding
        choices.append(dict(key=row['key'],alternative=q,scratch_bytes=scratch,padding_bytes=padding,workspace_bytes=workspace))
    assert len(choices)==len(used)
    choices.sort(key=lambda r:(-r['workspace_bytes'],r['key'],r['alternative']['name']))
    representative=choices[0]
    assert representative['key']==[6,960,960,4096]
    assert representative['alternative']['name']=='executable_catalog_2x3x3_cyclic1'
    assert representative['workspace_bytes']==2533228700
    assert representative['alternative']['flops']==3501135335
    programs=dict(passed=True,source=pin(CERT),records=records,scratch_blocks=peaks,
                  metadata_only=True,preparer=pin(Path(__file__)))
    write(HERE/'programs.json',programs)
    descriptor=dict(passed=True,selection='maximum complete compact workspace; ties lexicographic key then name',
                    prospective_plan=pin(PLAN),programs=pin(HERE/'programs.json'),used_keys=84,
                    representative=representative,ordered_choices=choices,metadata_only=True,preparer=pin(Path(__file__)))
    write(HERE/'representative.json',descriptor)
    print(json.dumps(dict(programs=pin(HERE/'programs.json'),representative=pin(HERE/'representative.json'))))


if __name__=='__main__':main()
