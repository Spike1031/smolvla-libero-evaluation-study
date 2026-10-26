"""Recalculate committed historical results. Standard library only; no model execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SERIES = {
    'rq1': ('paired_task0_states0_19_v1', ('1', '5')),
    'original': ('occlusion_center_task0_states0_19_v1', ('clean', 'occluded')),
    'round1': ('finetuned_task0_clean_occluded_v1', ('clean', 'occluded')),
    'round2': ('low_lr_fixed150_task0_clean_occluded_v1', ('clean', 'occluded')),
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def audit_series(project, label):
    folder, conditions = SERIES[label]
    root = project / 'results' / folder
    plan = read(root / 'plan.json')
    summary = read(root / 'summary.json')
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    require(plan_hash == summary['plan_hash'], f'{label}: plan hash mismatch')
    require(plan['state_ids'] == list(range(20)), f'{label}: unexpected state list')
    require(plan['suite'] == 'libero_spatial' and plan['task_id'] == 0,
            f'{label}: unexpected task')
    require(plan['seed_base'] == 42, f'{label}: unexpected base seed')
    paths = sorted(root.glob('state_*/*/result.json'))
    require(len(paths) == 40, f'{label}: expected 40 committed records, found {len(paths)}')
    rows, index = [], {}
    for path in paths:
        data = read(path)
        state = data['state_id']
        condition = str(data['n_action_steps']) if label == 'rq1' else data['condition']
        key = (state, condition)
        require(key not in index, f'{label}: duplicate {key}')
        require(type(state) is int and 0 <= state < 20 and condition in conditions,
                f'{label}: invalid state/condition {key}')
        expected_folder = 'k' + condition if label == 'rq1' else condition
        require(path.relative_to(root).as_posix() == f'state_{state:03d}/{expected_folder}/result.json',
                f'{label}: filename disagrees with record {path}')
        require(data['status'] == 'completed' and data['plan_hash'] == plan_hash,
                f'{label}: uncommitted or wrong-plan result {key}')
        require(data['seed'] == 42 + state, f'{label}: seed mismatch {key}')
        require(type(data['success']) is bool, f'{label}: non-Boolean success {key}')
        require(type(data['control_steps']) is int and 1 <= data['control_steps'] <= 280,
                f'{label}: invalid action count {key}')
        require(isinstance(data['eval_seconds'], (int, float)) and
                math.isfinite(data['eval_seconds']) and data['eval_seconds'] >= 0,
                f'{label}: invalid elapsed time {key}')
        require(data['n_action_steps'] == (int(condition) if label == 'rq1' else 1),
                f'{label}: action-step setting mismatch {key}')
        index[key] = data
        rows.append(dict(series=label, state=state, condition=condition,
                         goal_success=data['success'], control_steps=data['control_steps'],
                         eval_seconds=data['eval_seconds'], result_path=str(path.relative_to(project)),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    require(set(index) == {(s, c) for s in range(20) for c in conditions},
            f'{label}: incomplete pairing')
    aggregates = []
    for condition in conditions:
        selected = [index[s, condition] for s in range(20)]
        wins = sum(x['success'] for x in selected)
        seconds = sum(x['eval_seconds'] for x in selected)
        saved = summary['groups'][condition]
        require(saved['successes'] == wins and saved['completed'] == 20 and
                saved['paired_n'] == 20 and abs(saved['paired_success_rate'] - wins/20) < 1e-12 and
                abs(saved['paired_eval_seconds'] - seconds) < 1e-6,
                f'{label}/{condition}: saved summary mismatch')
        aggregates.append(dict(series=label, condition=condition, episodes=20,
                               goal_successes=wins, goal_rate=wins/20,
                               control_steps=sum(x['control_steps'] for x in selected),
                               eval_seconds=seconds))
    require(summary['total_completed'] == summary['total_requested'] == 40 and
            summary['paired_state_ids'] == list(range(20)), f'{label}: summary coverage mismatch')
    return index, rows, aggregates

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='Existing embodied-smolvla folder')
    parser.add_argument('--output', type=Path, required=True, help='New, empty output directory')
    parser.add_argument('--series', nargs='+', choices=list(SERIES), default=list(SERIES))
    args = parser.parse_args(argv)
    require(len(args.series) == len(set(args.series)), 'Duplicate series selection')
    project, output = args.project.resolve(), args.output.resolve()
    require(not output.is_relative_to(project), 'Output must be outside the source project')
    require(not output.exists() or not any(output.iterdir()), 'Output must be empty or absent')
    indices, episodes, aggregates = {}, [], []
    for label in args.series:
        index, rows, totals = audit_series(project, label)
        indices[label] = index
        episodes.extend(rows)
        aggregates.extend(totals)
    transitions = []
    if 'original' in indices:
        for candidate in ('round1', 'round2'):
            if candidate not in indices:
                continue
            for condition in ('clean', 'occluded'):
                for state in range(20):
                    before = indices['original'][state, condition]['success']
                    after = indices[candidate][state, condition]['success']
                    transitions.append(dict(candidate=candidate, state=state, condition=condition,
                        original_goal=before, candidate_goal=after,
                        change='unchanged' if before == after else ('repair' if after else 'regression')))
    if all(k in indices for k in ('original', 'round1', 'round2')):
        for state in (5, 14):
            require([indices[k][state, 'clean']['success'] for k in ('original', 'round1', 'round2')]
                    == [True, True, False], f'Corrected case {state:03d} no longer agrees with sources')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'episode_ledger.csv', episodes)
    write_csv(output/'summary.csv', aggregates)
    write_csv(output/'transitions.csv', transitions)
    result = dict(status='passed', scope='recorded-result recalculation; no inference or training',
                  series=args.series, committed_episodes=len(episodes), aggregates=aggregates)
    (output/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    main()
