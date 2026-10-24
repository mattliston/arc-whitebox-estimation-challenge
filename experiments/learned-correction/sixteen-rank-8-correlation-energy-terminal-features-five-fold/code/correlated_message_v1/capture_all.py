"""Serial, target-free acquisition on the fixed twenty TRAIN networks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parent
RUNTIME=D.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NETS=(35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    q=read(HERE/'qualification/completed.json')
    qw=read(HERE/'qualification_watch.json')
    assert q['completed'] and q['repeat_byte_exact'] and qw['completed'] and qw['whole_group_retired']
    assert qw['max_swap_growth_bytes']==0
    assert q['feature_source_sha256']==sha(HERE/'feature.py')
    assert q['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    original=read(D.parent/'astra_phase2_reexamination_v1/current_c2_terminal_quadratic_information_v1/train_capture_v2/manifest.json')
    assert tuple(original['train'])==NETS
    root=HERE/'train_capture'
    root.mkdir(exist_ok=False)
    rows=[]
    for ordinal,rank in enumerate(NETS):
        if rank==35:
            out=HERE/'qualification'
            watch_path=HERE/'qualification_watch.json'
        else:
            out=root/f'rank{rank:04d}'
            watch_path=root/f'rank{rank:04d}_watch.json'
            command=[sys.executable,str(D/'run_guarded.py'),'--record',str(watch_path),'--seconds','100',
                     str(RUNTIME),'-W','ignore',str(HERE/'capture.py'),str(rank),str(out)]
            with (root/f'rank{rank:04d}.log').open('x') as log:
                done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            assert done.returncode==0,(rank,done.returncode)
        watch=read(watch_path)
        assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
        meta=read(out/'completed.json')
        assert meta['completed'] and not meta['truth_accessed']
        assert meta['feature_source_sha256']==q['feature_source_sha256']
        record=meta['records'][0]
        assert record['rank']==rank and record['includes_initialization']
        assert sha(record['capture'])==record['capture_sha256']
        rows.append(dict(ordinal=ordinal,rank=rank,record=record,watch=str(watch_path.resolve()),
                         completion=str((out/'completed.json').resolve()),completion_sha256=sha(out/'completed.json')))
        print(json.dumps(dict(rank=rank,parent_flops=record['parent_flops'],feature_flops=record['feature_flops'],
                              wall_s=record['parent_wall_s'],max_rss_bytes=watch['max_rss_bytes'])),flush=True)
    result=dict(completed=True,truth_accessed=False,train=list(NETS),rows=rows,
        source_sha256=sha(__file__),capture_source_sha256=sha(HERE/'capture.py'),
        feature_source_sha256=q['feature_source_sha256'],protocol_sha256=q['protocol_sha256'])
    with (root/'manifest.json').open('x') as f:
        json.dump(result,f,indent=2)


if __name__=='__main__':
    main()
