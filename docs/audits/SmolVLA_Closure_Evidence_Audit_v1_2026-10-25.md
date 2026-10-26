# SmolVLA Project Closure: Evidence Inventory and Audit — Version 1

**Audit date:** 25 October 2026  
**Status:** First closure audit completed; final project report and repository curation remain pending.  
**Language:** Final report, README, captions and explanatory material will be in English. Historical source records retain their original language.

## 1. Scope and decision

The project is being closed as a bounded, single-task empirical study of action execution steps, central visual occlusion, and limited-demonstration adaptation. No new training or simulation was run for this audit. The current fine-tuning branch remains stopped. Multi-task evaluation, brightness perturbations, reinforcement learning and physical-robot deployment are outside the closure scope and must not be presented as completed work.

This document records what was checked against current Google Drive files. It is an evidence inventory, not the final research report and not an assertion that every historical notebook has been reviewed.

## 2. Audit coverage and checks

Four experiment series were checked at the individual-result level: 40 execution-step rollouts, 40 original clean/occluded rollouts, and 40 rollouts for each of the first two fine-tuning rounds. All **160 committed `result.json` records** were read.

For each series, the audit checked a unique record for every state-condition combination, states 000–019, completed status, Boolean goal outcome, seed `42 + state_id`, plan-hash consistency, control steps in 1–280, and finite nonnegative elapsed time. The plan hash was independently recomputed using the protocol's sorted-key JSON serialization. Success counts and cumulative elapsed time were recomputed from the individual records and matched every saved summary (time tolerance: 0.000001 seconds).

The four compatibility records contain the same initial-state-array hash and the same evaluation-library and environment-source hashes. Key recorded versions agree: Python 3.12.3, LeRobot 0.6.1, PyTorch 2.11.0+cu128, hf-libero 0.1.4, robosuite 1.4.0, NumPy 2.2.6 and Transformers 5.5.4; GPU: Tesla T4. Runner hashes differ across series and must be retained separately. These are checks of recorded metadata; this audit did not recreate the historical runtime or re-download model weights to verify their bytes.

The audit also read the plans, compatibility records and summaries for the later 30-rollout finishing evaluation and the four-case evaluation of model 19. Their per-step traces and videos were **not independently re-audited in this closure pass**.

## 3. RQ1: execution steps and runtime

| Setting | Rollouts | Goal successes | Goal rate | Control actions, total | Evaluation time, total |
|---|---:|---:|---:|---:|---:|
| `n_action_steps=1` | 20 | 16 | 80% | 2,367 | 3,385.618221 s |
| `n_action_steps=5` | 20 | 15 | 75% | 2,565 | 1,959.911733 s |

The k=5 series used about **42.11% less recorded cumulative evaluation time**, with one fewer goal-success episode. Both settings succeeded in 14 states; k=1 alone succeeded in 000 and 005; k=5 alone succeeded in 007; both failed in 002, 013 and 018.

This supports a descriptive runtime–outcome trade-off on these 20 paired states. It does not establish superiority, equivalent success rates, or generalization to other tasks. There was one rollout per state and setting. Recorded evaluation time includes more than model inference, and episode lengths differ. It is not a pure inference-speed measurement. The individual result records do not directly report the number of newly generated action chunks; a queue-based estimate must not be labelled a measured model-call count.

RQ1's previously uncertain evidence location is now resolved. No repeat rollout is required merely to recover its success and elapsed-time results.

## 4. RQ2: original policy under central occlusion

| Visual condition | Goal successes | Goal rate | Evaluation time, total |
|---|---:|---:|---:|
| Clean | 16/20 | 80% | 3,287.614794 s |
| Occluded | 12/20 | 60% | 4,275.505978 s |

Twelve states succeeded in both conditions. States **001, 011, 012 and 019** changed from clean success to occluded failure. No state changed in the opposite direction. States **002, 007, 013 and 018** failed in both conditions.

The observed change is a 20-percentage-point reduction in goal rate under this specific input perturbation. The mask covers rows 96–159 and columns 96–159 of the 256×256 main-camera image, filled with 0.5 at each policy observation; the wrist camera is unchanged. It is image masking, not a physical obstacle. This result does not establish general visual robustness across tasks, mask locations or lighting conditions.

The separately executed clean series in RQ1 and RQ2 must retain separate identities even though their goal totals and total control actions agree. Their recorded elapsed times differ; do not merge or count them as new independent states.

## 5. RQ3: first two fine-tuning rounds

| Policy | Clean goal successes | Occluded goal successes | Total across conditions |
|---|---:|---:|---:|
| Original visual baseline | 16/20 | 12/20 | 28/40 |
| Round 1, selected update 150 | 13/20 | 10/20 | 23/40 |
| Round 2, fixed update 150 | 14/20 | 14/20 | 28/40 |

These are 20 paired initial states under two conditions, not 40 independent initial states. Fine-tuned policy exports are identified by hashes in their evaluation plans. The plans identify round 1 as validation-selected and round 2 as a fixed-update lower-learning-rate candidate. Training-run provenance and checkpoint-selection calculations still need their own closure audit.

| Comparison against original | Condition | Failure → success | Success → failure |
|---|---|---|---|
| Round 1 | Clean | None | 001, 008, 016 |
| Round 1 | Occluded | 011 | 005, 008, 016 |
| Round 2 | Clean | None | 005, 014 |
| Round 2 | Occluded | 012, 019 | None |

Round 1 has one repaired state-condition case and six regressions. Round 2 has two repairs and two regressions: its unchanged aggregate total conceals a redistribution of outcomes. The evidence does not support an overall improvement in goal success from either candidate. It also does not show that no individual case improved.

## 6. Success definitions and later finishing work

The historical success fields above represent the benchmark-goal outcome under the early evaluation protocol. They do not certify genuine release, safe withdrawal, absence of re-contact, or a stable final two seconds. Report them as benchmark-goal success, separately from complete finishing.

The current 30-rollout finishing summary records goal attainment within 280 steps in 10/10 original, 10/10 original-only control and 9/10 finish-augmented episodes; all three groups have 0/10 automatic complete finishes. These are selected development cases rather than a representative independent success-rate estimate. The original `reserved_project_test` label must be explained historically: states 020–022 subsequently became known development cases.

The current four-case model-19 summary records 4 completed episodes, 1,760 policy actions, zero benchmark goals within 280 steps, zero goals at the end and zero automatic complete finishes. Its paired original comparison and detailed traces still need to be linked into the final closure evidence set.

**Metadata discrepancy:** Both later summaries still say `manual_review: pending`. The user has subsequently supplied manual video observations in the project conversation. Do not interpret the stale field as proof that no review occurred. Preserve the automatic files and add a separate, source-labelled manual-observation table during curation. The early notebook's coarse failure descriptions likewise need the user's later corrections carried forward.

Scripted finishing demonstrations from notebooks 11–12 can support demonstration construction and action feasibility. They cannot be reported as autonomous fine-tuned-policy success.

## 7. Closure status and remaining work

| Work item | Status after this audit | Remaining action |
|---|---|---|
| RQ1 goal, time and paired outcomes | Checked against all 40 committed results | Curate execution code; explain unmeasured model-call counts |
| RQ2 goal and paired outcomes | Checked against all 40 committed results | Consolidate corrected manual failure annotations and selected videos |
| RQ3 first two rounds: closed-loop goal outcomes | Checked against all 80 committed results | Audit training protocols, splits, checkpoint selection and export provenance |
| Later finishing evaluations | Plans and summaries checked in this pass | Link traces, paired original results and manual observations; resolve stale review labels |
| Offline fitting and action diagnostics | Outside this pass | Select only evidence required to explain the negative result; link source reports |
| Code and environment delivery | Source notebook and compatibility records located | Extract minimal English-documented entry points; map historical versions |
| Final research report and README | Pending | Draft from verified evidence, with limitations and reproduction instructions |

The next step is provenance and annotation consolidation using existing files. This is not a request to restart training or simulation. A missing historical metric can be explicitly reported as unavailable; it does not automatically justify a new run.

## 8. Retention and curation policy

The public/final package should contain the report, README, necessary evaluation and training entry points, pinned configurations, the results supporting every reported comparison, an analysis/recalculation entry point, and an indexed selection of success, failure and regression videos. Failed candidates are retained when they support a reported finding.

Keep raw committed result records and essential provenance as an evidence archive, even when they are not all displayed in the report. Duplicate notebooks, installation output and abandoned troubleshooting cells need not appear in the final package. Historical development material can be kept in a separate backup until the final package has been checked. No cloud file was deleted, moved or replaced during this audit.

Large model weights, duplicate videos and intermediate checkpoints should be assessed individually after the evidence map is complete. Do not delete them simply because a candidate failed, and do not publish private files or access tokens with the repository.

## 9. Source map

Paths in the episode ledgers below are relative to these source folders. Each entry points to a committed result, not an uncommitted or abandoned attempt.

- **RQ1 execution-step comparison:** [rq1](https://drive.google.com/drive/folders/10YwHpbdwutP2I3aHZAl3DXJqqGAy5j7U). Read `plan.json`, `summary.json`, `compatibility.json`, and all 40 committed episode results. Also read `requirements.txt`.
- **Original visual-occlusion baseline:** [occlusion_center_task0_states0_19_v1](https://drive.google.com/drive/folders/11e9F94C_iVVdI3G7z6wsnV4SZ-a0yZ6e). Read `plan.json`, `summary.json`, `compatibility.json`, and all 40 committed episode results.
- **Fine-tuning round 1:** [finetuned_task0_clean_occluded_v1](https://drive.google.com/drive/folders/160RYzyn0ZU9N89MUgxFe6f-gICqNHRIJ). Read `plan.json`, `summary.json`, `compatibility.json`, and all 40 committed episode results.
- **Fine-tuning round 2:** [low_lr_fixed150_task0_clean_occluded_v1](https://drive.google.com/drive/folders/19K8ceWyGiqZg9Dd2J4fR5IDXDbVyy_L2). Read `plan.json`, `summary.json`, `compatibility.json`, and all 40 committed episode results.

- **Later finishing evaluation:** [finish_policy_eval_v1](https://drive.google.com/drive/folders/15Np85Y_vr5rcNwhWwZq-JaxJj_FFOSzi), plan/compatibility/summary only in this pass.
- **Later four-case evaluation:** [19_four_case_eval_v1](https://drive.google.com/drive/folders/1wshDXa6jOnzo5T0bepAeIfO0iVqmHgEC), plan/compatibility/summary only in this pass.
- **Planning reference:** `SmolVLA_Original_Research_Plan_and_Outlook_2026-10-25.md`, previously read for the agreed scope.
- **Historical code/log source:** uploaded `01_smolvla_finetune_data_ready (1)(1).ipynb`, inspected before this audit. The current closure pass checked the source results rather than treating notebook prose as the final authority.

Initial-state array SHA-256, recorded identically in all four compatibility records:

`f061b67788d541001f3d6872be9a3a3b0b7de3490dac4683e181f9b0703f8e20`

## Appendix: independently recomputed episode ledgers

Success means the stored historical benchmark-goal flag. Elapsed times are rounded here for readability; recomputation used full JSON precision. SHA-256 values below identify the exact downloaded source bytes of each complete 40-record series using a deterministic ledger of sorted relative paths and per-file hashes.

### RQ1 execution-step comparison

Plan hash: `a37bef13a335943e129e299a944c0e1d65c870aee07e4d0a36a88d7117a6c535`  
Source-ledger SHA-256: `7992bd16a5d28bf751c7790b4d7563daa480a0b61f4c0acf8b6f8f110e535c7a`

| Result path | Seed | Goal | Actions | Elapsed seconds |
|---|---:|---|---:|---:|
| `state_000/k1/result.json` | 42 | Success | 74 | 140.396862 |
| `state_000/k5/result.json` | 42 | Failure | 280 | 202.204041 |
| `state_001/k1/result.json` | 43 | Success | 77 | 114.585845 |
| `state_001/k5/result.json` | 43 | Success | 72 | 70.153377 |
| `state_002/k1/result.json` | 44 | Failure | 280 | 384.532173 |
| `state_002/k5/result.json` | 44 | Failure | 280 | 198.123044 |
| `state_003/k1/result.json` | 45 | Success | 71 | 105.961558 |
| `state_003/k5/result.json` | 45 | Success | 69 | 58.384279 |
| `state_004/k1/result.json` | 46 | Success | 75 | 120.125644 |
| `state_004/k5/result.json` | 46 | Success | 74 | 62.561378 |
| `state_005/k1/result.json` | 47 | Success | 73 | 107.963245 |
| `state_005/k5/result.json` | 47 | Failure | 280 | 197.994213 |
| `state_006/k1/result.json` | 48 | Success | 68 | 101.266506 |
| `state_006/k5/result.json` | 48 | Success | 78 | 64.507633 |
| `state_007/k1/result.json` | 49 | Failure | 280 | 375.944626 |
| `state_007/k5/result.json` | 49 | Success | 71 | 59.356733 |
| `state_008/k1/result.json` | 50 | Success | 96 | 144.095025 |
| `state_008/k5/result.json` | 50 | Success | 80 | 65.515794 |
| `state_009/k1/result.json` | 51 | Success | 72 | 104.538610 |
| `state_009/k5/result.json` | 51 | Success | 72 | 59.705414 |
| `state_010/k1/result.json` | 52 | Success | 69 | 101.981100 |
| `state_010/k5/result.json` | 52 | Success | 76 | 62.551969 |
| `state_011/k1/result.json` | 53 | Success | 71 | 104.895306 |
| `state_011/k5/result.json` | 53 | Success | 76 | 63.186339 |
| `state_012/k1/result.json` | 54 | Success | 74 | 108.119599 |
| `state_012/k5/result.json` | 54 | Success | 132 | 101.494775 |
| `state_013/k1/result.json` | 55 | Failure | 280 | 374.251682 |
| `state_013/k5/result.json` | 55 | Failure | 280 | 193.444403 |
| `state_014/k1/result.json` | 56 | Success | 72 | 106.391305 |
| `state_014/k5/result.json` | 56 | Success | 77 | 62.592583 |
| `state_015/k1/result.json` | 57 | Success | 71 | 103.527639 |
| `state_015/k5/result.json` | 57 | Success | 79 | 65.815288 |
| `state_016/k1/result.json` | 58 | Success | 71 | 106.592193 |
| `state_016/k5/result.json` | 58 | Success | 76 | 63.308784 |
| `state_017/k1/result.json` | 59 | Success | 71 | 105.693101 |
| `state_017/k5/result.json` | 59 | Success | 64 | 55.329301 |
| `state_018/k1/result.json` | 60 | Failure | 280 | 374.723847 |
| `state_018/k5/result.json` | 60 | Failure | 280 | 196.059781 |
| `state_019/k1/result.json` | 61 | Success | 142 | 200.032354 |
| `state_019/k5/result.json` | 61 | Success | 69 | 57.622604 |

### Original visual-occlusion baseline

Plan hash: `bfefa5a481ed80d290b34d4f5402b3358395455b9240627469a04230ad62ae7d`  
Source-ledger SHA-256: `f077a427c2364a5990fbc0eeb386ac14eece50883ab6c14e746d6e9d2b55878d`

| Result path | Seed | Goal | Actions | Elapsed seconds |
|---|---:|---|---:|---:|
| `state_000/clean/result.json` | 42 | Success | 74 | 117.063043 |
| `state_000/occluded/result.json` | 42 | Success | 79 | 115.002247 |
| `state_001/clean/result.json` | 43 | Success | 77 | 112.160225 |
| `state_001/occluded/result.json` | 43 | Failure | 280 | 378.248070 |
| `state_002/clean/result.json` | 44 | Failure | 280 | 372.880582 |
| `state_002/occluded/result.json` | 44 | Failure | 280 | 374.535058 |
| `state_003/clean/result.json` | 45 | Success | 71 | 102.897465 |
| `state_003/occluded/result.json` | 45 | Success | 69 | 99.260592 |
| `state_004/clean/result.json` | 46 | Success | 75 | 106.238169 |
| `state_004/occluded/result.json` | 46 | Success | 73 | 105.516135 |
| `state_005/clean/result.json` | 47 | Success | 73 | 104.699089 |
| `state_005/occluded/result.json` | 47 | Success | 75 | 106.871215 |
| `state_006/clean/result.json` | 48 | Success | 68 | 96.795762 |
| `state_006/occluded/result.json` | 48 | Success | 68 | 97.782056 |
| `state_007/clean/result.json` | 49 | Failure | 280 | 372.048298 |
| `state_007/occluded/result.json` | 49 | Failure | 280 | 372.470075 |
| `state_008/clean/result.json` | 50 | Success | 96 | 136.607375 |
| `state_008/occluded/result.json` | 50 | Success | 96 | 135.643786 |
| `state_009/clean/result.json` | 51 | Success | 72 | 105.521328 |
| `state_009/occluded/result.json` | 51 | Success | 67 | 96.868666 |
| `state_010/clean/result.json` | 52 | Success | 69 | 101.304187 |
| `state_010/occluded/result.json` | 52 | Success | 69 | 100.743350 |
| `state_011/clean/result.json` | 53 | Success | 71 | 103.595301 |
| `state_011/occluded/result.json` | 53 | Failure | 280 | 371.881240 |
| `state_012/clean/result.json` | 54 | Success | 74 | 105.171938 |
| `state_012/occluded/result.json` | 54 | Failure | 280 | 371.372650 |
| `state_013/clean/result.json` | 55 | Failure | 280 | 368.889734 |
| `state_013/occluded/result.json` | 55 | Failure | 280 | 370.241482 |
| `state_014/clean/result.json` | 56 | Success | 72 | 102.962084 |
| `state_014/occluded/result.json` | 56 | Success | 77 | 109.315167 |
| `state_015/clean/result.json` | 57 | Success | 71 | 101.563188 |
| `state_015/occluded/result.json` | 57 | Success | 73 | 105.106547 |
| `state_016/clean/result.json` | 58 | Success | 71 | 102.590886 |
| `state_016/occluded/result.json` | 58 | Success | 74 | 106.143980 |
| `state_017/clean/result.json` | 59 | Success | 71 | 103.293949 |
| `state_017/occluded/result.json` | 59 | Success | 75 | 109.257051 |
| `state_018/clean/result.json` | 60 | Failure | 280 | 371.251334 |
| `state_018/occluded/result.json` | 60 | Failure | 280 | 371.494528 |
| `state_019/clean/result.json` | 61 | Success | 142 | 200.080858 |
| `state_019/occluded/result.json` | 61 | Failure | 280 | 377.752085 |

### Fine-tuning round 1

Plan hash: `3f3217a243cd8e62f6516f96d1eb1b1765b33c95fa6f44363b6bf1a373465028`  
Source-ledger SHA-256: `c2d29d2608d0d639ddca609c53c178daa49b5897f393ec85c6b38324430f8591`

| Result path | Seed | Goal | Actions | Elapsed seconds |
|---|---:|---|---:|---:|
| `state_000/clean/result.json` | 42 | Success | 76 | 134.785605 |
| `state_000/occluded/result.json` | 42 | Success | 91 | 125.699579 |
| `state_001/clean/result.json` | 43 | Failure | 280 | 366.153886 |
| `state_001/occluded/result.json` | 43 | Failure | 280 | 372.147918 |
| `state_002/clean/result.json` | 44 | Failure | 280 | 367.093874 |
| `state_002/occluded/result.json` | 44 | Failure | 280 | 364.562155 |
| `state_003/clean/result.json` | 45 | Success | 66 | 93.395345 |
| `state_003/occluded/result.json` | 45 | Success | 67 | 93.612322 |
| `state_004/clean/result.json` | 46 | Success | 74 | 102.835647 |
| `state_004/occluded/result.json` | 46 | Success | 71 | 99.383799 |
| `state_005/clean/result.json` | 47 | Success | 74 | 104.640182 |
| `state_005/occluded/result.json` | 47 | Failure | 280 | 364.269149 |
| `state_006/clean/result.json` | 48 | Success | 73 | 104.107020 |
| `state_006/occluded/result.json` | 48 | Success | 68 | 97.397372 |
| `state_007/clean/result.json` | 49 | Failure | 280 | 378.109454 |
| `state_007/occluded/result.json` | 49 | Failure | 280 | 387.934083 |
| `state_008/clean/result.json` | 50 | Failure | 280 | 369.205845 |
| `state_008/occluded/result.json` | 50 | Failure | 280 | 380.132643 |
| `state_009/clean/result.json` | 51 | Success | 67 | 101.530358 |
| `state_009/occluded/result.json` | 51 | Success | 67 | 100.183042 |
| `state_010/clean/result.json` | 52 | Success | 71 | 107.686824 |
| `state_010/occluded/result.json` | 52 | Success | 65 | 99.928595 |
| `state_011/clean/result.json` | 53 | Success | 70 | 105.334058 |
| `state_011/occluded/result.json` | 53 | Success | 72 | 109.074240 |
| `state_012/clean/result.json` | 54 | Success | 72 | 108.135998 |
| `state_012/occluded/result.json` | 54 | Failure | 280 | 379.314584 |
| `state_013/clean/result.json` | 55 | Failure | 280 | 371.139943 |
| `state_013/occluded/result.json` | 55 | Failure | 280 | 372.908600 |
| `state_014/clean/result.json` | 56 | Success | 75 | 105.762436 |
| `state_014/occluded/result.json` | 56 | Success | 76 | 110.460777 |
| `state_015/clean/result.json` | 57 | Success | 72 | 104.323445 |
| `state_015/occluded/result.json` | 57 | Success | 68 | 100.212549 |
| `state_016/clean/result.json` | 58 | Failure | 280 | 368.840563 |
| `state_016/occluded/result.json` | 58 | Failure | 280 | 383.030641 |
| `state_017/clean/result.json` | 59 | Success | 73 | 108.858610 |
| `state_017/occluded/result.json` | 59 | Success | 68 | 102.218330 |
| `state_018/clean/result.json` | 60 | Failure | 280 | 373.126243 |
| `state_018/occluded/result.json` | 60 | Failure | 280 | 372.313729 |
| `state_019/clean/result.json` | 61 | Success | 186 | 255.022371 |
| `state_019/occluded/result.json` | 61 | Failure | 280 | 378.548942 |

### Fine-tuning round 2

Plan hash: `681d7a73b7c12e96af8d42f4070abc8a75c1d8670080194d088a96e91b2da83d`  
Source-ledger SHA-256: `52f51033a108b7b5ff03c8c05156d468663ee14a1b8bf20526636b1e3c99101b`

| Result path | Seed | Goal | Actions | Elapsed seconds |
|---|---:|---|---:|---:|
| `state_000/clean/result.json` | 42 | Success | 79 | 130.037356 |
| `state_000/occluded/result.json` | 42 | Success | 77 | 103.374453 |
| `state_001/clean/result.json` | 43 | Success | 75 | 101.658835 |
| `state_001/occluded/result.json` | 43 | Failure | 280 | 347.533582 |
| `state_002/clean/result.json` | 44 | Failure | 280 | 349.559537 |
| `state_002/occluded/result.json` | 44 | Failure | 280 | 350.739266 |
| `state_003/clean/result.json` | 45 | Success | 71 | 94.972206 |
| `state_003/occluded/result.json` | 45 | Success | 70 | 94.267184 |
| `state_004/clean/result.json` | 46 | Success | 75 | 99.623530 |
| `state_004/occluded/result.json` | 46 | Success | 73 | 96.415670 |
| `state_005/clean/result.json` | 47 | Failure | 280 | 341.410419 |
| `state_005/occluded/result.json` | 47 | Success | 74 | 98.005664 |
| `state_006/clean/result.json` | 48 | Success | 72 | 95.635652 |
| `state_006/occluded/result.json` | 48 | Success | 66 | 88.821584 |
| `state_007/clean/result.json` | 49 | Failure | 280 | 344.986770 |
| `state_007/occluded/result.json` | 49 | Failure | 280 | 345.526430 |
| `state_008/clean/result.json` | 50 | Success | 105 | 135.432549 |
| `state_008/occluded/result.json` | 50 | Success | 209 | 262.366995 |
| `state_009/clean/result.json` | 51 | Success | 72 | 97.794207 |
| `state_009/occluded/result.json` | 51 | Success | 71 | 96.361840 |
| `state_010/clean/result.json` | 52 | Success | 68 | 92.549021 |
| `state_010/occluded/result.json` | 52 | Success | 69 | 93.172882 |
| `state_011/clean/result.json` | 53 | Success | 67 | 92.017386 |
| `state_011/occluded/result.json` | 53 | Failure | 280 | 350.320414 |
| `state_012/clean/result.json` | 54 | Success | 131 | 169.869615 |
| `state_012/occluded/result.json` | 54 | Success | 89 | 118.392444 |
| `state_013/clean/result.json` | 55 | Failure | 280 | 348.899235 |
| `state_013/occluded/result.json` | 55 | Failure | 280 | 352.316723 |
| `state_014/clean/result.json` | 56 | Failure | 280 | 344.267021 |
| `state_014/occluded/result.json` | 56 | Success | 78 | 105.728856 |
| `state_015/clean/result.json` | 57 | Success | 71 | 96.618536 |
| `state_015/occluded/result.json` | 57 | Success | 73 | 99.340587 |
| `state_016/clean/result.json` | 58 | Success | 70 | 94.854952 |
| `state_016/occluded/result.json` | 58 | Success | 75 | 100.449249 |
| `state_017/clean/result.json` | 59 | Success | 71 | 95.729193 |
| `state_017/occluded/result.json` | 59 | Success | 75 | 101.680176 |
| `state_018/clean/result.json` | 60 | Failure | 280 | 339.594498 |
| `state_018/occluded/result.json` | 60 | Failure | 280 | 339.264164 |
| `state_019/clean/result.json` | 61 | Success | 130 | 165.398955 |
| `state_019/occluded/result.json` | 61 | Success | 69 | 91.833082 |

