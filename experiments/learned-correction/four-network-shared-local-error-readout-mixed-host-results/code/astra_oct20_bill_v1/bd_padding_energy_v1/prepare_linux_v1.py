"""Freeze path-remapped manifests and full32 plan; import only, no native launch."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

def write(p,d):
    with p.open('x')as f:json.dump(d,f,indent=2);f.write('\n')

def remap(x):
    if isinstance(x,str):return x.replace('/Users/matt/arc-white-box-estimation/',str(ROOT)+'/')
    if isinstance(x,list):return [remap(y)for y in x]
    if isinstance(x,dict):return {k:remap(v)for k,v in x.items()}
    return x

def main():
    original=HERE/'CANDIDATES_v1.json'
    assert sha(original)=='479285ca094e2524efc1fcae58539cf5bea75af6dcec86259097c1830177488d'
    m=json.loads(original.read_text())
    for filename in ('TINY_v1.json','IMPORT_v1.json'):
        r=json.loads((HERE/filename).read_text())
        assert r['successful']and r['manifest_sha256']==sha(original)
    linux=remap(m);linux.update(path_remap_only=True,original_manifest_sha256=sha(original))
    mp=HERE/'CANDIDATES_LINUX_v1.json';write(mp,linux)
    for rec in linux['candidates']:
        for role in ('science','release'):
            assert sha(rec[role]['path'])==rec[role]['sha256']
            assert sha(rec[role]['parent_path'])==rec[role]['parent_sha256']
        assert sha(rec['append_tail'])==rec['append_tail_sha256']
    imported=HERE/'IMPORT_LINUX_v1.json'
    subprocess.run(['/home/arcworker/arc-envs/client/bin/python',str(HERE/'check_import_v1.py'),
                    '--manifest',str(mp),'--out',str(imported)],check=True)
    assert json.loads(imported.read_text())['successful']
    base=RUNS/'PLAN_CURRENT_MSG8_RANK16_24_v1.json'
    assert sha(base)=='352d985d064e6d9adafdeedc1dbb8808dc328730c83a46560d24b3af0a8277d5'
    plan=json.loads(base.read_text())
    plan.update(utc=utc(),kind='Same-count above-floor-preserving BD-corrected energy padding-only birth selector on current777a',
                reporting_reference='Exact same-host BLAS4 source777a VALID337422 control; full paired per-network ratios.',
                reference_submission=337422,
                bd_padding_energy_provenance=dict(manifest=str(mp),manifest_sha256=sha(mp),
                    tiny_sha256=sha(HERE/'TINY_v1.json'),import_sha256=sha(imported),
                    no_fitting=True,no_sweep=True,no_truth_fit=True,no_carrier_correction=True))
    plan.pop('fit_receipt',None)
    plan['candidates']={'bank337393':plan['candidates']['bank337393']}
    for rec in linux['candidates']:
        p=Path(rec['science']['path'])
        plan['candidates'][rec['name']]=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
            role='frozen_candidate',sha256=sha(p),effective_sha256=sha(p))
    pp=RUNS/'PLAN_MSG8_BD_PADDING_ENERGY_v1.json';write(pp,plan)
    receipt=dict(utc=utc(),status='BD_PADDING_ENERGY_QUALIFIED_READY_NO_NATIVE_LAUNCH',
                 plan=str(pp),plan_sha256=sha(pp),manifest=str(mp),manifest_sha256=sha(mp),
                 linux_import_sha256=sha(imported),local_tiny_sha256=sha(HERE/'TINY_v1.json'),
                 candidates=linux['candidates'],reference_submission=337422,
                 control='/home/arcworker/arc-runs/msg8_loaderfix337422_control32_markline_blas4_v1',
                 requested_jobs=12,shared_max_active=36,blas_threads=4,
                 cgroup_root='/sys/fs/cgroup/arc-estimator-v5',native_gain_unknown=True)
    write(HERE/'LINUX_READY_v1.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
