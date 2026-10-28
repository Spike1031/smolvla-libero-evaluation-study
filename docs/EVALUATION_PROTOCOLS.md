# Evaluation protocols and source verification

## Early benchmark protocol

Task: `libero_spatial`, task 0, black bowl between plate and ramekin to plate. The upstream task goal is `On akita_black_bowl_1 plate_1`. This is not an explicit grasp, gripper-release, withdrawal or stability certificate. The current upstream predicate implementation checks relative object geometry and contact; its exact historical installed implementation has not yet been hash-pinned here. Do not present current upstream numerical tolerances as a verified historical fact.

The LeRobot v0.6.1 environment and evaluation sources were retrieved and their bytes match both historical SHA-256 values in the committed compatibility record; see `tables/source_protocol_audit.json`. In this verified source:

- The spatial suite's default episode cap is 280 policy actions.
- `is_success = self._env.check_success()`; termination includes `done or is_success`.
- Episode success aggregates the encountered per-step success values using `any` before termination.
- Video frames are cut at the done index and encoded using `render_fps`, which is 80 in the environment metadata.

Thus approximately 80 saved frames play in one second; 280 frames play in approximately 3.5 seconds. Exact duration requires inspecting that specific video. Playback duration is not wall-clock inference duration or a calibrated simulation-time measurement. Training update 150 is a checkpoint-selection/update count, not the 280-action rollout cap.

The original, round-1 and round-2 result fields support historical benchmark-goal attainment. They cannot establish what would happen after their early termination. First-round runner `experiments/08_finetuned_eval_v1.py` is extracted verbatim from notebook 04 and matches the historical runner hash. It delegates episode evaluation to the verified LeRobot evaluator. Other historical runners remain external until individually extracted and checked; common use of a protocol is not proof every wrapper was identical.

## Extended finishing protocol

Notebook-15 packaged traces contain 440 policy actions per episode. Three separate quantities must be reported:

1. `benchmark_goal_within_280`: the goal was true at least once in the first 280 actions.
2. `goal_at_end`: the goal is true at action 440.
3. `complete_finish_automatic`: a full 40-step finishing window passes, and every subsequent rolling 40-step window through action 440 also passes.

A passing window requires goal true at every step, no gripper-bowl contact, recorded whole-gripper clearance lower bound of at least 0.05 m, end-effector and bowl position diameters at most 0.002 m, and orientation ranges at most 2 degrees. These are project-defined automatic checks. They do not independently establish perfect posture, prove the clearance estimator correct, or replace video review. Forty control steps must not be described as a fixed number of video-playback seconds without a separately checked time base.

A goal may turn true, false and true again in this protocol because execution continues. That is different from an early-terminated episode. Apparent placement in a video is also different from the geometric predicate. Compare model, state, condition, attempt and protocol before calling labels inconsistent.

## Limitations

No early full-finish success rate can be recovered from omitted future frames. Existing complete-finish zeros apply only to the evaluated later traces and automatic definition. Validation on demonstration observations is not equivalent to closed-loop rollout performance. The dataset prefix, scripted teacher and autonomous candidate must retain separate identities.
