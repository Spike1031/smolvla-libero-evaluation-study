"""Recalculate recorded finishing scores; no inference or simulation."""
import json
import gzip
import hashlib
from pathlib import Path
from finish_scoring import score_trace

root = Path(__file__).resolve().parents[1]
def load_trace(name):
    directory = root / 'evidence/later' / (name + '.json.gz.parts')
    index = json.loads((directory / 'index.json').read_text())
    chunks = []
    for part in index['parts']:
        data = (directory / part['file']).read_bytes()
        if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
            raise ValueError(f"Trace part checksum mismatch: {part['file']}")
        chunks.append(data)
    compressed = b''.join(chunks)
    if hashlib.sha256(compressed).hexdigest() != index['compressed_sha256']:
        raise ValueError('Compressed trace checksum mismatch')
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != index['original_sha256']:
        raise ValueError('Original trace checksum mismatch')
    return json.loads(raw)

first = load_trace('finish_policy_eval')
last = load_trace('model19_four_cases')
keys = ['benchmark_goal_within_280', 'first_goal_step', 'goal_at_end',
        'goal_steps_total', 'first_complete_40step_window_end',
        'later_failed_window_end_steps', 'recontact_steps_after_first_complete_window',
        'complete_finish_automatic']
count = 0
for case in first['cases'] + last['cases'] + last['reused_original_cases']:
    actual = score_trace(case['step_trace'])
    saved = case['result']['scores']
    for key in keys:
        if actual[key] != saved[key]:
            raise ValueError(f'Score mismatch: {case["result"]["model"]}, {key}')
    if actual['last_40_steps']['checks'] != saved['last_40_steps']['checks']:
        raise ValueError('Final-window check mismatch')
    count += 1
print(json.dumps({'status': 'passed', 'trace_records': count,
                  'unique_rollouts': 34, 'reused_baseline_records': 4}))
