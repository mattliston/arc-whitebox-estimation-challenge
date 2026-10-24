"""[opus-release] SAFE for S2t (#336367): exact S2t + two-strip covariance fragment + first-five full-D6 fragment (template
astra_qhfl_safe_source_v1/build_source_HELD_v1.py: parent + '\\n\\n' + two + '\\n\\n' + five). Prep only: no science, no fire."""
import hashlib, json, subprocess
from pathlib import Path
L = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs')
W = Path(__file__).resolve().parent; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent = L/'opus_release_v1/s2t/candidate_S2t_cblazy_v1.py'
two = L/'astra_gop2_distinct_safe_policy_v1/two_strip_v1/fragment_covariance_two_strip_HELD_v1.py'
five = L/'astra_rtt2_safe_two_strip_v1/request_margin_v2/fragment_safe_five_full_HELD_v2.py'
old_safe = L/'astra_rtt2_safe_two_strip_v1/request_margin_v2/candidate_rtt2_safe_HELD_v2.py'
assert sha(parent) == 'e0553d2e153694b5504c70da73cafb33b8c0649675f5d92648d2547372c79d46'
assert sha(five) == '38e63cd232fe08ce6d327f56474b5716a5c8831f8732976264830ba41d0c7432'
assert two.read_text() in old_safe.read_text() and five.read_text() in old_safe.read_text()   # the fired SAFE #336185's exact fragments
text = parent.read_text() + '\n\n' + two.read_text() + '\n\n' + five.read_text()
assert text[:len(parent.read_text())] == parent.read_text()
out = W/'candidate_S2t_safe_v1.py'; out.write_text(text); compile(text, str(out), 'exec')
m = dict(utc=subprocess.check_output(['/bin/date','-u','+%FT%TZ'], text=True).strip(), status='SAFE_SOURCE_PREP_NO_SCIENCE_NO_FIRE',
         candidate=dict(path=str(out), sha256=sha(out)), parent=dict(path=str(parent), sha256=sha(parent), fired='#336367'),
         fragments=[dict(path=str(two), sha256=sha(two)), dict(path=str(five), sha256=sha(five))], donor_public_safe=336185, donor_source_sha256=sha(old_safe))
with open(W/'SAFE_SOURCE_MANIFEST_v1.json', 'x') as f: json.dump(m, f, indent=2)
print(m['candidate']['sha256'])
