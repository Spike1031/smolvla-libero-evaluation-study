"""Historical finish scoring functions. CPU only; requires NumPy."""
import numpy as np
WINDOW_STEPS=40
MAX_STEPS=440

def hold_window(rows):
    if len(rows) != WINDOW_STEPS:
        return {'passed': False, 'reason': 'Fewer than 40 steps'}

    def diameter(points):
        a = np.asarray(points, float)
        return float(np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1).max())
    matrices = np.asarray([r['eef_matrix'] for r in rows]).reshape(-1, 3, 3)
    cosine = np.clip((np.einsum('aij,bij->ab', matrices, matrices) - 1) / 2, -1, 1)
    eef_angle = float(np.degrees(np.arccos(cosine)).max())
    q = np.asarray([r['bowl_quaternion'] for r in rows], float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    bowl_angle = float(np.degrees(2 * np.arccos(np.clip(np.abs(q @ q.T), 0, 1))).max())
    eef_d = diameter([r['eef_position'] for r in rows])
    bowl_d = diameter([r['bowl_position'] for r in rows])
    checks = {'goal_every_step': all((r['goal'] for r in rows)), 'no_gripper_bowl_contact': all((not r['gripper_bowl_contact'] for r in rows)), 'whole_gripper_clearance_5cm': all((r['clearance_lower_bound_m'] >= 0.05 for r in rows)), 'eef_position_stable': eef_d <= 0.002, 'bowl_position_stable': bowl_d <= 0.002, 'eef_orientation_stable': eef_angle <= 2.0, 'bowl_orientation_stable': bowl_angle <= 2.0}
    fingers = np.asarray([r['gripper_joint_positions'] for r in rows], float)
    return {'passed': all(checks.values()), 'checks': checks, 'eef_position_diameter_m': eef_d, 'bowl_position_diameter_m': bowl_d, 'eef_orientation_range_deg': eef_angle, 'bowl_orientation_range_deg': bowl_angle, 'minimum_clearance_lower_bound_m': min((r['clearance_lower_bound_m'] for r in rows)), 'gripper_joint_max_range_m_diagnostic': float(np.ptp(fingers, axis=0).max())}

def score_trace(trace):
    assert len(trace) == MAX_STEPS and [r['step'] for r in trace] == list(range(1, MAX_STEPS + 1))
    first_goal = next((r['step'] for r in trace if r['goal']), None)
    windows = [hold_window(trace[i - WINDOW_STEPS:i]) for i in range(WINDOW_STEPS, len(trace) + 1)]
    first = next((i for i, w in enumerate(windows) if w['passed']), None)
    first_end = first + WINDOW_STEPS if first is not None else None
    later_failures = [] if first is None else [i + WINDOW_STEPS for i in range(first + 1, len(windows)) if not windows[i]['passed']]
    recontacts = [] if first_end is None else [r['step'] for r in trace[first_end:] if r['gripper_bowl_contact']]
    return {'benchmark_goal_within_280': first_goal is not None and first_goal <= 280, 'first_goal_step': first_goal, 'goal_at_end': trace[-1]['goal'], 'goal_steps_total': sum((r['goal'] for r in trace)), 'first_complete_40step_window_end': first_end, 'last_40_steps': windows[-1], 'later_failed_window_end_steps': later_failures, 'recontact_steps_after_first_complete_window': recontacts, 'complete_finish_automatic': first is not None and (not later_failures), 'automatic_label_note': 'First full 40-step finish window must pass; every subsequent rolling 40-step window through step440 must also pass. Video review remains required.'}
