from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
import json,requests,hashlib
D=Path(__file__).resolve().parent
class Page(HTMLParser):
 def __init__(self):super().__init__();self.props=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'data-react-props' in a:self.props.append(json.loads(a['data-react-props']))
def safe(x):
 if not isinstance(x,dict):return None
 allowed={'id','submission_id','score','score_secondary','grading_status_cd','username','name','team_name','display_name','slug','is_public','is_private','is_selected','selected','is_valid','is_disabled','is_flagged','is_eligible','is_leaderboard','submission_type','created_at','updated_at','isPublic','isPrivate','isSelected','createdAt','updatedAt','gradingStatus','gradingStatusCd'}
 z={k:v for k,v in x.items() if k in allowed and isinstance(v,(str,int,float,bool,type(None)))}
 z['available_keys']=list(x)
 for k in ('participant','user','team','submitter','owner','challengeRound','challenge_round'):
  if k in x:z[k]=safe(x[k])
 return z
records=[]
with requests.Session() as c:
 c.trust_env=False
 for sid in (333111,333050,332738):
  url=f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}'
  r=c.get(url,params={'_visibility_observed':datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')},timeout=20,headers={'Cache-Control':'no-cache'});r.raise_for_status();p=Page();p.feed(r.text)
  data=[x.get('props',{}).get('data',{}).get('submission') for x in p.props];data=[x for x in data if x]
  assert len(data)==1
  records.append({'id':sid,'url':r.url,'public_submission_metadata':safe(data[0]),'html_sha256':hashlib.sha256(r.content).hexdigest()})
result={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'records':records}
with (D/'public_metadata.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))

