"""Read only the newly authorized Phase 2 bank329926; never upload/post."""
import json
from pathlib import Path
from whestbench.aicrowd_client import AIcrowdClient

root = Path('/Users/matt/arc-white-box-estimation')
client = AIcrowdClient(api_key=(root / 'key.txt').read_text().strip())
reply = client.get_submission_status(329926)
data = reply.get('data', reply)
fields = ('id', 'challenge_id', 'challenge_round_id', 'created_at',
          'grading_status_cd', 'grading_message', 'score', 'score_secondary')
result = {key:data[key] for key in fields if key in data}
assert result.get('id', 329926) == 329926
assert result.get('challenge_id', 1174) == 1174
assert result.get('challenge_round_id', 1429) == 1429
print(json.dumps(result), flush=True)
