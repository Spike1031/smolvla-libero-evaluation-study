"""Verify notebook-15 video index to packaged trace linkage; no video decoding."""
import csv,gzip,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
d=p/'evidence/later/finish_policy_eval.json.gz.parts'
i=json.loads((d/'index.json').read_text())
cases=json.loads(gzip.decompress(b''.join((d/x['file']).read_bytes() for x in i['parts'])))['cases']
rows=list(csv.DictReader((p/'videos/video_index.csv').open()))
for r in rows:
    matches=[c for c in cases if c['result']['model']==r['model'] and c['result']['state_id']==int(r['state']) and c['result']['condition']==r['condition']]
    if len(matches)!=1:raise ValueError('Nonunique video/trace key')
    scores=matches[0]['result']['scores']
    for column,key in [('goal_280','benchmark_goal_within_280'),('goal_end','goal_at_end'),('full_finish','complete_finish_automatic')]:
        if str(scores[key])!=r[column]:raise ValueError(f'Label mismatch: {r["model"]}/{r["state"]}/{column}')
print(json.dumps({'status':'passed','linked_rows':len(rows),'scope':'record keys and scores; binary video identity not verified'}))
