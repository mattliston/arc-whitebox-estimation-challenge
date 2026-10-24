"""Canonical complete-line group owner, only diagnostic client substituted."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;RUNS=HERE.parents[1];PORT=RUNS/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7';CANON=RUNS/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1';sys.path.insert(0,str(PORT))
from common_linux_v7 import sha
BASE=CANON/'run_group_linux_markline_v1.py';assert sha(BASE)=='e2004b88e39f801243cdf2778e549f80534ff3a12f95814f5d2778fc68cd3757'
source=BASE.read_text();old="str(here / 'client_linux_markline_v1.py')";assert source.count(old)==1;source=source.replace(old,repr(str(HERE/'client_profile_v1.py')));__file__=str(BASE);exec(compile(source,str(BASE)+':d399_outside_profile','exec'),globals())
