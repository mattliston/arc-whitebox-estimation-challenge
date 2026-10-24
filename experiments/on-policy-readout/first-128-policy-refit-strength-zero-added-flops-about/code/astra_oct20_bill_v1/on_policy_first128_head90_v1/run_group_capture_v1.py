"""Same capacity-v7 process-group ownership; substitute diagnostic client only."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PORT = HERE.parents[1] / 'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7'
sys.path.insert(0, str(PORT))
from common_linux_v7 import sha
BASE = PORT/'run_group_linux_v7.py'
assert sha(BASE) == '149e7af3aef527d5fea7e19af36fca2cf09ad72cc7362fe15906fe8b0edc884e'
source = BASE.read_text()
old = "str(here / 'client_linux_v7.py')"
assert source.count(old) == 1
source = source.replace(old, repr(str(HERE/'client_capture_v1.py')))
__file__ = str(BASE)
exec(compile(source, str(BASE)+':full_state_response_capture', 'exec'), globals())
