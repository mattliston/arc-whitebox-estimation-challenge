"""Read one accepted submission; never print credentials."""
import argparse
import datetime
from whestbench.aicrowd_client import AIcrowdClient
from common import HERE, CLI, write

parser = argparse.ArgumentParser()
parser.add_argument('submission_id',type=int)
parser.add_argument('sequence',type=int)
args = parser.parse_args()
key = (CLI.parents[2]/'key.txt').read_text().strip()
response = AIcrowdClient(api_key=key).get_submission_status(args.submission_id)
data = response.get('data',response)
record = dict(submission_id=args.submission_id,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    **{k:data.get(k) for k in ['grading_status_cd','score','score_secondary','grading_message']})
write(HERE/f'sub{args.submission_id}_poll_{args.sequence:02d}.json',record)
print(record,flush=True)
