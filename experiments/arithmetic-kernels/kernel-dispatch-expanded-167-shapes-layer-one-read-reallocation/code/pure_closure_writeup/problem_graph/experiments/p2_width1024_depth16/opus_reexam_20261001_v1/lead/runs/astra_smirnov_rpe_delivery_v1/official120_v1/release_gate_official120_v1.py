from pathlib import Path
import hashlib,json,sys,runpy
R=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
POLICY=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json')
assert hashlib.sha256(POLICY.read_bytes()).hexdigest()=='f7ed781f660a20d2687b8d39289258e41338ccaf38a1bcd8d71c7c513f4ba132'
assert json.loads(POLICY.read_text())['approved'] is True
sys.path.insert(0,str(R))
S=R/'release_gate_v4.py'
assert hashlib.sha256(S.read_bytes()).hexdigest()=='99388ec83da42d5be2fa301f603407fdc986381af34f0817cb0ae99d2fe65c66'
code=S.read_text()
assert code.count("judge_source = R/'designation_judge.py'")==1
code=code.replace("judge_source = R/'designation_judge.py'", "judge_source = Path("+repr('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py')+")",1)
exec(compile(code,str(S),'exec'),dict(__name__='__main__',__file__=str(S)))
