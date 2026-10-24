from pathlib import Path
import hashlib,json,sys
_policy=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json')
assert hashlib.sha256(_policy.read_bytes()).hexdigest()=='f7ed781f660a20d2687b8d39289258e41338ccaf38a1bcd8d71c7c513f4ba132'
assert json.loads(_policy.read_text())['approved'] is True
_original=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py')
assert hashlib.sha256(_original.read_bytes()).hexdigest()=='c2c0fc318545f7ea74dce26c64c27236c34bf3375b56220d9d26f3b50f44c11e'
sys.path.insert(0,str(_original.parent))
_code=_original.read_text()
assert _code.count('wall_worst_s=100.0')==1
_code=_code.replace('wall_worst_s=100.0','wall_worst_s=120.0',1)
exec(compile(_code,str(_original),'exec'),dict(__name__='__main__',__file__=str(_original)))
