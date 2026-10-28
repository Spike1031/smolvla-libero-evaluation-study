"""Verify unique video inventory and stored labels; no video decoding."""
import csv,gzip,json
from pathlib import Path
p=Path(__file__).resolve().parents[1]
d=p/'evidence/later/finish_policy_eval.json.gz.parts'
i=json.loads((d/'index.json').read_text())
cases=json.loads(gzip.decompress(b''.join((d/x['file']).read_bytes() for x in i['parts'])))['cases']
rows=list(csv.DictReader((p/'videos/video_index.csv').open()))
if len({r['url'] for r in rows})!=len(rows) or len({r['video_id'] for r in rows})!=len(rows):raise ValueError('Duplicate video URL or ID')
for r in rows:
    if r['protocol']=='early_stop_on_goal_280_cap':
        result=json.loads((p/r['result_path']).read_text())
        if str(result['success'])!=r['goal_280']:raise ValueError('Early goal mismatch')
        if result['state_id']!=int(r['state']) or result['condition']!=r['condition']:raise ValueError('Early identity mismatch')
        if r['source_path'] not in [v['path'] for v in result['videos']]:raise ValueError('Attempt path mismatch')
        if r['full_finish']!='unknown':raise ValueError('Early finishing must remain unknown')
    else:
        matches=[c for c in cases if c['result']['model']==r['model'] and c['result']['state_id']==int(r['state']) and c['result']['condition']==r['condition']]
        if len(matches)!=1:raise ValueError('Nonunique video/trace key')
        for column,key in [('goal_280','benchmark_goal_within_280'),('goal_end','goal_at_end'),('full_finish','complete_finish_automatic')]:
            if str(matches[0]['result']['scores'][key])!=r[column]:raise ValueError('Extended label mismatch')
print(json.dumps({'status':'passed','unique_videos':len(rows),'early_records':sum(r['group']=='early_comparisons' for r in rows),'extended_records':sum(r['group']=='extended_finishing' for r in rows),'scope':'record identity and labels; binary video not reviewed'}))
