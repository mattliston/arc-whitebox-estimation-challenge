"""Finish the preexisting CHECK37/39/42/48 roster, without refitting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
ARMS={'baseline':(HERE.parent/'fable_wall_repair_v1/cand_c2_wallrepair_I95.py',
                  'dd64dfc367fbaf154fe956f335c21aa51c1589e7f19a731be74c29ca41bc29b3'),
      'recovered':(HERE/'recovered_p5_I95.py',
                   '41fd0cee138f929c7d8939009e8bbe720dce09c8124628033f0d2d29db01ac99')}


def main():
    out=HERE/'recovered_check4_v1'
    out.mkdir(exist_ok=False)
    (out/'PROTOCOL.md').write_text('''# Complete the recovered fit CHECK4\n\nThe earlier37 result is retained as a negative single-network observation.\nIt does not establish mean performance on the original CHECK37/39/42/48.\nComplete exactly the other three paired native evaluations, six fresh calls,\nwith unchanged P5 and I95 sources. No training, dose, parameter, interpolation,\ncheckpoint, or roster selection. Reuse the sealed37 cold calls. Open original\n1e9 FP32 truth files only after all six outputs and bills are sealed.\nThis completes the original holdout assessment; a positive finding would\nstill require an independent14 panel and separate composition validation.\n''')
    records=[]
    for rank in (37,39,42,48):
        for arm,(candidate,sha) in ARMS.items():
            assert hashlib.sha256(candidate.read_bytes()).hexdigest()==sha
            target=HERE/f'{arm}37' if rank==37 else out/f'{arm}_{rank}'
            watch=HERE/f'{arm}37_watch.json' if rank==37 else out/f'{arm}_{rank}_watch.json'
            if rank!=37:
                command=[sys.executable,str(HERE/'run_guarded.py'),'--record',str(watch),
                         '--seconds','90',str(RUNTIME),'-W','ignore',str(HERE/'native_screen.py'),
                         str(candidate),str(target),'--ranks',str(rank)]
                with (out/f'{arm}_{rank}.log').open('x') as log:
                    done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                assert done.returncode==0,(arm,rank,done.returncode)
            w=json.loads(watch.read_text())
            assert w['completed'] and w['whole_group_retired']
            result=json.loads((target/'result.json').read_text())
            assert result['candidate_sha256']==sha and not result['truth_accessed']
            c=result['records'][0]
            assert c['rank']==rank and c['includes_initialization']
            assert hashlib.sha256(Path(c['output']).read_bytes()).hexdigest()==c['output_sha256']
            records.append(dict(arm=arm,rank=rank,call=c,watch=str(watch)))
            print(json.dumps(dict(arm=arm,rank=rank,flops=c['flops'],wall_s=c['wall_s'])),flush=True)
    with (out/'completed.json').open('x') as f:
        json.dump(dict(completed=True,new_calls=6,records=records,truth_accessed=False),f,indent=2)


if __name__=='__main__':main()
