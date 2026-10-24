from pathlib import Path
import hashlib,json,sys,runpy
R=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
POLICY=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json')
assert hashlib.sha256(POLICY.read_bytes()).hexdigest()=='f7ed781f660a20d2687b8d39289258e41338ccaf38a1bcd8d71c7c513f4ba132'
assert json.loads(POLICY.read_text())['approved'] is True
sys.path.insert(0,str(R))
S=R/'stage_c_boundary.py'
assert hashlib.sha256(S.read_bytes()).hexdigest()=='5ddf8eb5be826939edfa3ac4a40d0a63ad601f7b7dac6f7ce0b2e70fd740bb04'
assert '--from-run' in sys.argv, 'metadata reassessment only'
import common
common.LIMITS=dict(common.LIMITS,worst_wall_s=120.0)
runpy.run_path(str(S),run_name='__main__')
