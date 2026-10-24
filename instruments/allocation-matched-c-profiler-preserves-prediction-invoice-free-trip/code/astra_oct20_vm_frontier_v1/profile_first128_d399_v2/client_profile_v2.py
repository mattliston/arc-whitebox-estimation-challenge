"""Frozen scope-correct, matched-state C-profile adapter over canonical client."""
import importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=HERE.parents[1];PORT=RUNS/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7';CANON=RUNS/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1';sys.path.insert(0,str(PORT))
from common_linux_v7 import sha,verified
BASE=CANON/'client_linux_markline_v1.py';assert sha(BASE)=='1beba822cd02cebdfe9f8b30d0e08fb498a870a85fecaeccbe2fa38782098455'
MODE=os.environ['ARC_D399_CPROFILE_MODE'];assert MODE in ('control','observe')
RECIPE_PATH=HERE/'RECIPE_v1.json';assert sha(RECIPE_PATH)==os.environ['ARC_D399_CPROFILE_RECIPE_SHA'];RECIPE=json.loads(RECIPE_PATH.read_text())
PROFILE_STATE=[None]
def install_candidate_profiler(candidate,root):
 assert sha(candidate)==RECIPE['candidate']['effective_sha256']
 helper_path=verified(root,RECIPE['observer']);spec=importlib.util.spec_from_file_location('candidate_cprofile_v1',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 return helper.CandidateProfile(str(candidate))
def adapted_source():
 source=BASE.read_text();patches=[
  ('assert sha(a.candidate) == a.candidate_sha',"assert sha(a.candidate) == a.candidate_sha == RECIPE['candidate']['effective_sha256']\nassert a.ranks == '35' and not a.warmup_each and a.setup == 'paced'\nassert sys.version_info[:2] == (3,10)"),
  ("out / 'rows.jsonl'","out / 'diagnostic_rows.jsonl'"),("f'pred_{rank}.bin'","f'diagnostic_pred_{rank}.bin'"),
  ('harness_sha256=sha(__file__),',"harness_sha256=sha(__file__), diagnostic_only=True, profile_mode=MODE, profile_recipe_sha256=sha(RECIPE_PATH),"),
  ('        u0, start = own_usage(), time.monotonic()\n',"        outside_profile = install_candidate_profiler(a.candidate, a.root)\n        PROFILE_STATE[0] = outside_profile\n        u0, start = own_usage(), time.monotonic()\n"),
  ('        prediction = est.predict(MLP(weights, rank), 2**41)\n',"        outside_profile.start(MODE == 'observe')\n        try:\n            prediction = est.predict(MLP(weights, rank), 2**41)\n        finally:\n            outside_profile.stop()\n")]
 for old,new in patches:assert source.count(old)==1,old;source=source.replace(old,new)
 return source
if __name__=='__main__':
 try:exec(compile(adapted_source(),str(BASE)+':d399_cprofile_'+MODE,'exec'),globals())
 finally:
  observer=PROFILE_STATE[0]
  if observer is not None:observer.write(out/'candidate_cprofile_v1.json')
