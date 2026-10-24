"""Read public evaluation state without executing page code or resubmitting."""
import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import requests

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.records = []
    def handle_starttag(self, tag, attrs):
        value = dict(attrs).get("data-react-props")
        if value:
            item = json.loads(value).get("props", {}).get("data", {}).get("submission")
            if item:
                self.records.append(item)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.submission_id > 0 and args.output_dir.is_dir()
    url = f"https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{args.submission_id}"
    response = requests.get(url, timeout=30, headers={"Cache-Control": "no-cache"})
    response.raise_for_status()
    page = Page()
    page.feed(response.content.decode("utf-8"))
    assert len(page.records) == 1, len(page.records)
    evaluation = page.records[0].get("evaluation")
    utc = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
    receipt = dict(submission_id=args.submission_id, url=url, observed_utc=utc, evaluation=evaluation)
    target = args.output_dir / f"eval_{args.submission_id}.json"
    target.write_text(json.dumps(receipt, indent=2))
    args.output_dir.joinpath(f"eval_{args.submission_id}_{utc.replace(':','').replace('-','')}.json").write_text(json.dumps(receipt, indent=2))
    ev = evaluation or {}
    results = ev.get("results") or {}
    rows = results.get("per_mlp") or []
    summary = dict(submission_id=args.submission_id, observed_utc=utc,
                   state=ev.get("state"), phase=ev.get("phase"),
                   updated_at=ev.get("updated_at"), progress=ev.get("progress"),
                   counts=dict(Counter(row.get("state") for row in rows)),
                   aggregates=results.get("aggregates"), error=ev.get("error"),
                   failures=[{k:row.get(k) for k in ("mlp_index","mlp_name","state","error_code")}
                             for row in rows if row.get("error_code") or row.get("state")=="failed"],
                   runtime_environment=ev.get("runtime_environment"))
    print(json.dumps(summary))
if __name__ == "__main__":
    main()

