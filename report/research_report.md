# Execution Steps, Visual Occlusion and Limited-Demonstration Adaptation of SmolVLA

> Release note, 28 October 2026: this is the dated 26 October report, matching the archived PDF. Subsequent source/hash inspection and video linkage are documented in [the source audit](../docs/TRAINING_AND_SOURCE_AUDIT.md), [protocol definitions](../docs/EVALUATION_PROTOCOLS.md) and [the unified 11-video catalogue](../videos/README.md). Statements below about pending delivery describe the report date, not current release readiness. No experimental scores have changed.

## A single-task empirical study in LIBERO

**Research report, version 1 — 26 October 2026**  
**Scope:** completed experiments and their bounded conclusions; no claim of a new state-of-the-art method.  
**Project status:** the current adaptation branch is closed. Evidence curation and publication of a complete reproduction repository remain separate delivery tasks.

## Abstract

This project evaluated a pretrained SmolVLA policy on one LIBERO spatial manipulation task under a constrained Colab T4 workflow. Three questions were examined: the behavior–runtime trade-off from executing one versus five predicted actions before replanning; sensitivity to a fixed central mask in the main camera; and whether limited-demonstration supervised adaptation improved closed-loop behavior. On 20 paired initial states, executing five actions produced 15 goal-success episodes versus 16 for one action, while reducing recorded cumulative evaluation time by 42.11%. Central occlusion reduced original-policy goal success from 16/20 to 12/20. Two early fine-tuning candidates achieved 23/40 and 28/40 condition-rollout successes versus the original 28/40; the unchanged second-round total concealed two repairs and two regressions. Subsequent inspection identified a gap between triggering the benchmark goal and completing release, withdrawal and stabilization. In a later ten-case development evaluation per policy, none of the original, original-data control or finish-augmented policies satisfied the automatic complete-finish rule. A hold-focused candidate substantially reduced fixed-observation action error but failed to reach the goal in all four targeted closed-loop cases. The tested interventions did not yield a verified better autonomous policy. The principal outcomes are controlled comparisons, explicit success definitions, failure evidence and a calculation-reproduction entry point.

## 1. Research questions and scope

The project used SmolVLA, LeRobot and LIBERO to study a complete observation-to-action workflow rather than treating one successful video as sufficient evidence. Its questions were:

1. **RQ1 — execution steps:** How does the number of actions executed before using a fresh observation affect task outcome and recorded evaluation cost?
2. **RQ2 — visual occlusion:** How does one fixed main-camera mask affect paired outcomes and observed failures?
3. **RQ3 — limited-demonstration adaptation:** Can supervised adaptation improve this task without losing existing capability?

The final experimental scope is one task, two execution-step settings, clean/occluded inputs, and bounded adaptation experiments. Earlier multi-task and brightness-perturbation proposals were not implemented as part of this core study. Reinforcement learning, agent systems and physical-robot deployment were outside scope.

Release and stabilization became an additional question after video inspection exposed behavior not captured by early goal termination. This later question is reported separately; it does not replace RQ1–RQ3 or retroactively change their results.

## 2. Task, policy and experimental controls

The task instruction was: pick up the black bowl between the plate and the ramekin and place it on the plate. Evaluation used `libero_spatial`, task 0. The corresponding demonstration-data task index was 34; dataset and simulator task numbering are different.

The pretrained policy was `HuggingFaceVLA/smolvla_libero`, revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Its VLM was `HuggingFaceTB/SmolVLM2-500M-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The recorded environment included Python 3.12.3, LeRobot 0.6.1, PyTorch 2.11.0+cu128, hf-libero 0.1.4, robosuite 1.4.0, NumPy 2.2.6 and a Tesla T4. Full dependency and compatibility records remain with the source experiments.

Inputs included two RGB cameras, an eight-dimensional robot state and the language instruction; actions had seven dimensions. The bottom-level simulation control rate was 20 Hz. `n_action_steps` specifies how many predicted actions are executed before replanning, not the physical control frequency.

The early studies paired states 000–019 with seed `42 + state_id`. RQ1 compared execution steps 1 and 5. Visual-baseline and early adapted-policy studies used execution steps 1 and two visual conditions. Each state-condition-setting was run once. The early horizon was 280 control actions, with termination when the environment reported success. Committed failures were retained rather than rerun until successful.

All four early series have matching recorded initial-state-array, evaluation-library and environment-source hashes; runner hashes remain distinct. This supports comparability of recorded configurations, not an assertion that every external dependency was freshly reproduced during closure.

## 3. Visual perturbation and outcome definitions

The mask replaced rows 96–159 and columns 96–159 in the 256×256 main-camera input with normalized grey 0.5 at every policy observation. Its 64×64 area is 6.25% of that image. The wrist camera remained unmasked. This is input corruption, not a physical obstacle or a claim about loss of 6.25% of all available visual information.

Three outcomes must be distinguished:

| Outcome | Meaning |
|---|---|
| Early benchmark-goal success | The environment goal was triggered within 280 actions under the historical termination protocol |
| Final goal state | The goal predicate was true at the end of a later fixed-duration rollout |
| Automatic complete finish | A later temporal, contact, clearance and stability rule passed; manual assessment remained complementary |

Early benchmark-goal success does not certify genuine release, safe retreat or sustained stability. Short, early-terminated videos cannot establish behavior after termination. Historical labels are preserved even where manual descriptions raise questions about their physical interpretation.

## 4. RQ1: execution steps and recorded evaluation cost

| Execution steps | Goal successes | Goal rate | Total control actions | Cumulative evaluation time |
|---|---:|---:|---:|---:|
| 1 | 16/20 | 80% | 2,367 | 3,385.618221 s |
| 5 | 15/20 | 75% | 2,565 | 1,959.911733 s |

Both settings succeeded in 14 states. Only execution steps 1 succeeded in 000 and 005; only execution steps 5 succeeded in 007; both failed in 002, 013 and 018. Thus more frequent replanning was not uniformly better on every state.

Executing five actions reduced recorded total evaluation time by 42.11%, with one fewer success. The elapsed time includes inference, simulation and other evaluation overhead, while episode lengths differ. It must not be presented as a pure inference-speed gain. Direct action-chunk-generation call counts were not stored in these result records. The evidence supports a descriptive trade-off in this sample, not a statistically established superiority or equivalence claim.

## 5. RQ2: original policy under central occlusion

| Input | Goal successes | Goal rate |
|---|---:|---:|
| Clean | 16/20 | 80% |
| Occluded | 12/20 | 60% |

Twelve states succeeded under both inputs. Clean successes 001, 011, 012 and 019 became occluded failures. No state changed in the opposite direction. States 002, 007, 013 and 018 failed under both inputs.

The 20-percentage-point reduction demonstrates sensitivity to this particular mask in this task. It does not establish general robustness across mask positions, illumination or other tasks. The separate RQ1 and RQ2 clean runs retain separate identities despite matching goal totals; their elapsed times differ and they do not constitute independent sets of initial states.

The user's baseline video observations distinguish empty grasps from contact without lift and post-lift drops. For example, 002 clean/occluded showed empty grasps; 012 occluded was lifted and then dropped onto the table. These observations improve failure-stage description but do not establish a causal explanation such as a specific visual-localization error.

## 6. RQ3: original-demonstration adaptation

### 6.1 Data and optimization

The task subset contained 45 original demonstrations and 4,487 frames. Whole-episode splitting produced 36 training episodes / 3,516 frames and 9 validation episodes / 971 frames. Episode sets are disjoint within this project; overlap with original pretraining is unknown. Validation used three anchors per held-out episode and fixed noise seeds.

Training froze the VLM and updated the action expert and related state projection. Microbatch size was 1 with four-step gradient accumulation and action chunk length 50. AdamW used betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10 and gradient clipping at 10. Later reports recorded 97,451,872 trainable parameters. Limited computational resources do not make this a tiny parameter update; no LoRA or reinforcement learning was used.

Round 1 used learning rate 1e-5 for 300 updates. Update 150 had the lowest validation loss among trained candidates checked every 50 updates. Round 2 restarted from the locked original policy, used learning rate 1e-6 and predetermined update 150, retaining the sampling/noise scheme and original normalization.

### 6.2 Offline and closed-loop results

| Policy | Fixed-anchor validation loss | Clean goals | Occluded goals | Total |
|---|---:|---:|---:|---:|
| Original | 0.705650717 | 16/20 | 12/20 | 28/40 |
| Round 1, update 150 | 0.709265547 | 13/20 | 10/20 | 23/40 |
| Round 2, update 150 | 0.706086817 | 14/20 | 14/20 | 28/40 |

Round 1's best trained candidate still had higher validation loss than the original; “best trained candidate” must not be shortened to “better model.” Round 2's lower update-50 loss was not used to change its fixed-update selection retrospectively. Neither selected candidate exceeded the original aggregate goal rate.

| Candidate vs original | Repairs | Regressions |
|---|---|---|
| Round 1 | 011 occluded | 001 clean; 005 occluded; 008 clean/occluded; 016 clean/occluded |
| Round 2 | 012 occluded; 019 occluded | 005 clean; 014 clean |

The second-round unchanged total masks an occluded improvement and a clean regression. Notebook 08 explicitly assigns the rightward tilt without lift in 005 clean and rightward translation without lift in 014 clean to round 2; the original and round 1 succeeded in both. The previously disputed attribution was a closure-consolidation error, not an inconsistency in the user's notebook logs.

## 7. Finishing demonstrations and autonomous evaluation

### 7.1 Demonstrations are an intervention, not a policy score

Video inspection revealed delayed separation, residual holding, jitter, re-contact and regrasping after apparent placement. Original demonstration episode 1500 ended while still holding the bowl; episode 1679 released without safe withdrawal. The former was validation-only in this project. These examples indicate incomplete finishing coverage; they do not prove that conflicting training labels caused every failure.

Scripted release, upward withdrawal and hold sequences were recorded after policy-generated prefixes. Three new episodes—001 clean, 005 clean and 016 occluded—were used for training; 008 clean was held out in full. The combined dataset contained 49 episodes. User review found the three batch demonstrations to release, withdraw and remain stable without delayed separation. This establishes useful teacher trajectories, not autonomous learned performance.

Two candidates started from the original policy and trained for 50 updates at 1e-6: an original-data-only control, and a finish-augmented arm. The latter used 100 original-data samples plus 43 placement/release, 26 lift and 31 hold samples. Both retained original processors and normalization.

### 7.2 Extended policy-only evaluation

Notebook 15 evaluated the original and both candidates on ten selected state-condition cases each: 001 clean, 005 clean, 016 occluded, 008 clean, and states 020–022 under both visual conditions. Each rollout ran 440 policy actions at 20 Hz without stopping at the first goal. There was no scripted finishing controller in this evaluation.

The automatic finishing rule required a 40-step window with the goal true throughout, no gripper–bowl contact, at least 5 cm conservative whole-gripper clearance, and stable end-effector/bowl positions and orientations. Position diameter was at most 2 mm and orientation range at most 2 degrees. Every later rolling window through action 440 also had to pass. Clearance used enclosing geometric bounds rather than exact surface distance. Finger-joint motion was diagnostic and required complementary manual inspection.

| Policy | Goal within 280 | Goal at action 440 | Automatic complete finish |
|---|---:|---:|---:|
| Original | 10/10 | 8/10 | 0/10 |
| Original-only control | 10/10 | 7/10 | 0/10 |
| Finish-augmented | 9/10 | 8/10 | 0/10 |

No policy obtained a first passing 40-step finishing window. These selected development cases are not representative independent task-rate estimates. States 020–022 were initially reserved within the project but became known development cases after inspection.

User observations illustrate the difference between aggregate goals and safe completion. In 008 clean, the original/control showed delayed separation and continuing jitter, while the finish-augmented model failed to lift. In 020 clean/occluded, the finish-augmented model avoided later bowl contact but still jittered. In 001 clean it regrasped after placement and dropped the bowl partly onto the table. A local improvement did not establish a reliable complete policy.

The original 020 clean was described as fully released and entirely on the plate, though slightly tilted, while the automatic final-goal label was false. Both records are preserved; appearance alone does not resolve the simulator predicate. Historical `manual_review: pending` fields are stale relative to later user feedback and are not evidence that no viewing occurred.

## 8. Offline action diagnostics and the final closed-loop gate

Later diagnostics compared first actions at fixed demonstrated states for release, lift and hold, including two fixed noise seeds. Such tests ask whether correct observed states produce appropriate action commands. They do not test recovery under the states reached during a rollout.

Model 19 used 50 updates at 1e-5 with 100 original-data samples and 100 hold samples. Model 20 used the same learning rate/budget and restored the original mixed-stage schedule. Both started from the original policy, not from earlier adapted weights.

| Hold translation-command L2 error | Original initialization | Model 19 | Model 20 |
|---|---:|---:|---:|
| Training-case anchors, six evaluations | 0.879252 | 0.283471 | 0.383318 |
| Held-out 008 anchors, two evaluations | 0.969696 | 0.329691 | 0.618788 |

These means were recalculated from the stored per-anchor diagnostic entries. Units are action-command coordinates, not physical displacement or velocity. Lower hold error coexisted with higher original-retention validation loss: 0.763575 for model 19 and 0.755922 for model 20, versus the original 0.705651. No autonomous improvement is claimed for model 20.

Notebook 22 subjected existing model 19 to a four-case gate: 001 clean, 005 clean, 016 occluded and 008 clean. Four matching original-policy cases were reused from notebook 15, with identical source traces and geometry verified during closure.

| Policy | Goal within 280 | Goal at end | Automatic complete finish |
|---|---:|---:|---:|
| Reused original baseline | 4/4 | 4/4 | 0/4 |
| Model 19 | 0/4 | 0/4 | 0/4 |

User review described contact without lift followed by misses in 001, repeated misses in 005, small bowl translation without lift in 016, and an initial miss with uncertain later grasp attempts in 008. The candidate did not pass the capability-retention gate. Lower fixed-state hold error was therefore insufficient grounds to continue this branch or expand its evaluation.

## 9. What the evidence establishes

RQ1 demonstrates a sample-specific runtime–outcome trade-off. RQ2 demonstrates vulnerability to one fixed central mask. RQ3 rejects the proposed aggregate benefit of the tested early adaptation candidates, while preserving individual repairs. The finishing work demonstrates a substantial gap between benchmark-goal attainment and stable autonomous completion. Its offline diagnostics show that improvement at selected demonstrated states need not transfer to closed-loop behavior.

The evidence does not establish a unique failure mechanism. Distribution shift during execution, incomplete demonstration coverage and objective/intervention design remain hypotheses. Neither catastrophic forgetting, a proven training-label conflict nor architectural inability has been isolated. In particular, inspecting validation-only trajectories cannot by itself demonstrate an optimization-data conflict.

The current branch is stopped because the tested candidates failed the relevant behavioral gate, not because additional hyperparameter sweeps were proven impossible. Reopening would require a materially different, evidence-based hypothesis, an explicit capability-retention gate and a bounded compute budget.

## 10. Limitations and reproducibility status

This is a single-task development study with one rollout per state-setting in the main comparisons. Repeated use and inspection of states limits generalization claims. Demonstration pretraining overlap is unknown. Historical physical replay alignment was not fully established. Video annotations come from the user and are not an independent blinded rating; some extended clips still lack exact path linkage. The automatic finishing rule is a defined conservative proxy, not a universally validated robotics standard.

The closure audit checked 160 committed early records and training/selection provenance. This report's later pass recalculated discrete scores for 38 trace records: 30 notebook-15 rollouts, four new notebook-22 rollouts and four reused originals. The reused four do not add independent experiments. Large policy weights were not freshly downloaded and rehashed. No model inference or simulation was run to prepare this report.

An English README, standard-library `reproduce_results.py`, and a CPU Colab notebook are available for recorded-result calculations. The new calculation entry was tested on 120 original/round-1/round-2 records, including rejection of a corrupted seed. The packaged calculation entry has now been run successfully on all 160 included committed records, including RQ1; results agree with the saved summaries. A full clean-runtime training/simulation reproduction package, a public repository and exhaustive video linkage remain delivery work. An eight-clip representative video index is included in this delivery; this report does not imply their completion.

## 11. Conclusion

The project produced reproducible recorded comparisons and a bounded negative adaptation result. Executing five actions reduced measured evaluation time with a small observed goal-count difference; central occlusion reduced goal success; early adaptation either regressed or redistributed capability without an aggregate gain. Added finishing demonstrations and improved offline hold commands did not produce a verified better autonomous policy. Future investigation should distinguish learning at demonstrated states from execution under reached states, and measure complete finishing alongside preservation of grasping and placement. These are conditional research directions, not new experiments initiated at closure.

## Evidence references

1. **Early outcome audit:** `SmolVLA_Closure_Evidence_Audit_v1_2026-10-25.md`. Forty committed results per early series, source links and episode ledgers.
2. **Training provenance and manual inventory, corrected version 1:** `SmolVLA_Training_Provenance_and_Manual_Cases_2026-10-25.md`. Whole-episode splits, checkpoint selection, export identifiers and user-observation boundaries.
3. **Round-1 source log:** [Notebook 04](https://colab.research.google.com/drive/1vakHYZiaUQ4iO_Cx_tL_v9jTCbxYlDDo#scrollTo=87vALz4clWy7).
4. **Round-2 source log and corrected 005/014 attribution:** [Notebook 08](https://colab.research.google.com/drive/1y9bsPROi0UyLWLWpPHIqtEmLH8IMFCFE#scrollTo=1H6BO1veGbyi), final log sections 6–7.
5. **Extended finishing evaluation:** [source folder](https://drive.google.com/drive/folders/15Np85Y_vr5rcNwhWwZq-JaxJj_FFOSzi). Archived `smolvla_finish_policy_eval_v1.json` SHA-256: `4bb29c96155c5becfdef2bb4924fcd1fd832c9f9a75704141066eae85c53314a`; its embedded plan agrees with the current source plan.
6. **Model-19 closed-loop gate:** [source folder](https://drive.google.com/drive/folders/1wshDXa6jOnzo5T0bepAeIfO0iVqmHgEC). `smolvla_19_four_case_eval_v1.json`; its baseline identifier matches reference 5. Reused traces were checked for exact equality.
7. **Model-19 offline diagnostics:** [training report](https://drive.google.com/file/d/1XcHdWFuKTAr_ocES2yTy_dGy24RtAJY4/view).
8. **Model-20 offline diagnostics:** [training report](https://drive.google.com/file/d/1eNz35WJfhzE0jlh_V7GlmV8Afk_GvyrH/view).

The references identify project evidence, not a literature review. No novelty, universal superiority or publication acceptance is asserted.
