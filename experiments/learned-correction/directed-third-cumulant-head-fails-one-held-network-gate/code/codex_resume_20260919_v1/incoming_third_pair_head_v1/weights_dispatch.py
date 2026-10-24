from common import *
verify();verify('runtime_repair_sealed.json')
raise SystemExit(subprocess.run([str(P2/'data_gate/.venv/bin/python'),'-B','-W','ignore',str(HERE/'extract_weights.py'),sys.argv[1]]).returncode)
