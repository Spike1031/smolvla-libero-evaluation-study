# SmolVLA Closure: Training Provenance, Manual Observations and Retention Plan

**Date:** 25 October 2026  
**Status:** Closure audit supplement; no new training, inference or simulation.  
**Correction:** 25 October 2026 — cases 005/014 clean assigned to round 2 using the current notebook logs; the earlier baseline attribution was an analyst error.  
**Companion:** `SmolVLA_Closure_Evidence_Audit_v1_2026-10-25.md`.

## 1. Findings and audit boundary

The first two fine-tuning rounds have internally consistent data splits, checkpoint-selection records, training-report hashes and evaluation export identifiers. The later original-only and finish-augmented candidates match the model identifiers used in notebook 15, and the hold-focused model 19 matches notebook 22's evaluation plan. No model-identity mismatch was found in the records checked here.

This establishes recorded provenance, not a new execution test. Small exported JSON configurations for rounds 1–2 were downloaded and their hashes recomputed. Multi-gigabyte policy weights and optimizer checkpoints were not downloaded or rehashed. Later frozen-weight and reload checks below are historical checks recorded in the source reports, not checks rerun during this audit. The raw demonstration arrays were not re-decoded here.

The manual case inventory is a translation and consolidation of the user's observations in the project conversation. It is not a fresh independent video review. Automatic goal outcomes and manual physical observations remain separate. The current Drive copies of notebooks 04 and 08 were subsequently read to verify the round attribution of cases 005/014 clean. Other exact-video linkage limitations remain explicit.

## 2. Original demonstration split and training method

The 45 original task demonstrations contain 4,487 frames: 36 training episodes / 3,516 frames and 9 validation episodes / 971 frames. The data-plan hash was recomputed and agrees with the split. Both original fine-tuning protocols embed exactly that split. The two ID sets are disjoint.

Validation episodes: **1327, 1330, 1344, 1352, 1450, 1500, 1506, 1532, 1672**. Validation uses three anchors per episode (start, middle and end), giving 27 fixed observations, with fixed noise seeds. These are within-project holdouts; overlap with the policy's original pretraining remains unknown.

The inspected training scripts load the locked original SmolVLA checkpoint, enable action-expert/state-projection training, and assert that the VLM parameters are frozen. They use AdamW, microbatch 1, gradient accumulation 4, chunk length 50, betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10, and gradient clipping at 10. This is supervised adaptation, not LoRA or reinforcement learning. Later training reports record **97,451,872 trainable parameters**; describing the work as low-resource must not imply a tiny parameter update.

The first two scripts implement the same episode/frame sampling and per-slot seed scheme. Round 2 starts from the locked original checkpoint rather than continuing round 1. Recorded split consistency does not itself prove that temporal alignment or all demonstration targets are correct. The older protocol's unresolved replay-alignment note must not be silently removed from the historical record.

## 3. First two rounds: checkpoint selection and losses

| Item | Round 1 | Round 2 |
|---|---|---|
| Learning rate | 1e-5 | 1e-6 |
| Completed optimizer updates | 300 | 150 |
| Selection rule | Minimum recorded validation loss at 50-step intervals, among trained candidates | Predetermined update 150 |
| Exported update | 150 | 150 |
| Original validation loss | 0.7056507170200348 | 0.7056507170200348 |
| Exported candidate loss | 0.7092655472181462 | 0.7060868166111134 |
| Historical benchmark-goal total | 23/40 | 28/40 |

All recorded validation means were independently recomputed from the 27 per-anchor losses. Round 1's update 150 is the minimum of its six trained candidates, but it does **not** improve on the original model's validation loss. Its selection rule excluded the untrained baseline as a competing checkpoint; the final report must say “best trained candidate,” not “better than the original.”

Round 2's update-50 loss is 0.7031329097571196, lower than its exported update-150 loss. Exporting 150 follows the fixed-update protocol and is not a selection mistake. The lower update-50 loss is not evidence of better autonomous performance, and does not justify starting another evaluation during closure.

| Update | Round 1 validation loss | Round 2 validation loss |
|---:|---:|---:|
| 50 | 0.722136739228 | 0.703132909757 |
| 100 | 0.722373327723 | 0.705432435981 |
| 150 | 0.709265547218 | 0.706086816611 |
| 200 | 0.710157369022 | Not run |
| 250 | 0.724152851436 | Not run |
| 300 | 0.727456539317 | Not run |

For each round, the SHA-256 of the downloaded training report matches the value committed in its evaluation plan. The complete six-entry export-hash mapping also matches. The downloaded `config.json`, `policy_preprocessor.json` and `policy_postprocessor.json` match their respective recorded hashes.

## 4. Finishing data and later candidate provenance

The finishing dataset adds four recorded episodes to the 45 original demonstrations, for **49 episodes**. Training adds `finish_001_clean` (203 frames), `finish_005_clean` (194) and `finish_016_occluded` (188). `finish_008_clean` (194 frames) is wholly excluded from optimization and used for development validation. The manifest's training and validation key sets are disjoint.

The resulting split contains 39 training episodes / 4,101 frames and 10 validation episodes / 1,165 frames. These include policy-generated prefixes followed by scripted finishing actions; they are not autonomous-policy success demonstrations throughout.

For the original-only control, 50 updates × four microbatches yield 200 original-data samples. For the finish-augmented arm, 100 original samples are mixed with 100 new samples: 43 placement/release, 26 lift and 31 hold. These counts were recomputed from the schedule. Model 19 uses 100 original and 100 new hold samples. Model 20 restores the exact original finish-augmented schedule; that schedule equality was checked, as was membership of the model-19/20 schedule keys in the training set.

| Candidate | Notebook identity | LR / updates | Role and provenance |
|---|---|---|---|
| Original-only control | Trained in 14; evaluated in 15 | 1e-6 / 50 | Export mapping and training/reload report hashes match the notebook-15 evaluation plan |
| Finish-augmented mixture | Trained in 14; evaluated in 15 | 1e-6 / 50 | Same provenance checks pass; three new training episodes, one held-out finish episode |
| Hold-focused model 19 | Trained in 19; evaluated in 22 | 1e-5 / 50 | Export mapping matches notebook-22's model entry |
| Phase-mixture model 20 | Trained in 20; offline diagnosis | 1e-5 / 50 | Fixed update and reload/export records checked; no closed-loop performance claimed here |

All four later reports record the same original initialization hashes, 50 completed updates, selection of fixed update 50, changed trainable weights, unchanged frozen weights, strict weight round-trip success, and processors identical to the original. Their reload reports agree with the training reports' export mappings.

Historic protocol fields such as `fixed_before_training_not_executed` describe the plan's preparation stage. Completion is established by later training reports; do not treat that stale plan label as proof training did not occur. The original package report similarly remains a pre-training preparation record.

File-prefix numbers on Python scripts do not necessarily equal notebook numbers. For example, `17_finish_learning_pilot_v1.py` implements the notebook-14 training experiment, while `22_hold_lr1e5_fixed50_v1.py` implements notebook 19. The curated README must map experiment names to entry points explicitly.

## 5. Manual case inventory: annotation rules

- `Clean` and `occluded` identify visual input conditions, not separate independent initial states.
- `A` means an empty grasp attempt; `B` means contact with the bowl without successful lift. These describe observed attempts, not the cause of failure.
- “Apparent adhesion” describes delayed or one-sided attachment visible in the user's account. No physical adhesion mechanism is established.
- “First-attempt grasp and placement” does not certify actual release, withdrawal or a stable final two seconds.
- A short clip ending at placement leaves subsequent behavior unknown. Later, more detailed observations supersede earlier coarse descriptions for the same identified video.
- Original baseline, round-1, round-2, extended post-goal checks, scripted demonstrations, notebook-15 evaluations and notebook-22 evaluations are separate series. Matching state numbers do not make them the same rollout.
- The user-facing three-digit case labels are retained. Directions are the user's image-relative descriptions, not calibrated world-axis measurements.
- The following is a selected evidence inventory, not exhaustive manual labeling of every rollout. Blank or unavailable detail must not be converted to success.

### 5.1 Original clean/occluded baseline

| Case | User-reported observation | Interpretation / boundary |
|---|---|---|
| 002 clean and occluded | Empty grasp | A; no successful lift |
| 007 clean and occluded | Contact causes slight wobble; bowl stays on table | B |
| 013 clean and occluded | Contact tilts bowl to the right; bowl stays on table | B; this is the later baseline correction |
| 018 clean and occluded | Contact moves bowl backwards; bowl stays on table | B |
| 001 occluded | Contact causes slight wobble; no lift | B |
| 011 occluded | Contact moves bowl slightly left; no lift | B |
| 012 occluded | Bowl is lifted, then dropped onto the table rather than the plate | Post-lift drop; not merely a failed initial grasp |
| 005 clean | Original model succeeds, as explicitly recorded in notebook 08, log section 6 | The rightward-tilt failure belongs to round 2, not the original baseline |
| 014 clean | Original model succeeds, as explicitly recorded in notebook 08, log section 6 | The rightward-translation failure belongs to round 2, not the original baseline |

**Attribution corrected:** Notebook 08, final log section 6, explicitly labels both observations as second-round failures with original-model success. Section 7 explicitly records first-round success for both. There is no conflict in those notebook logs. The earlier closure supplement incorrectly placed the round-2 failure descriptions under the original baseline; responsibility lies with that consolidation, not with the user's logs. All audited automatic outcomes remain unchanged.

### 5.2 First fine-tuning round

| Case | User-reported observation | Interpretation / boundary |
|---|---|---|
| 001 clean | Right bowl rim is contacted; bowl tilts but remains on table | B; regression from original goal success |
| 005 occluded | Similar rim contact and tilt without lift | B; regression |
| 008 clean and occluded | Similar contact/tilt without lift | B; two regressions |
| 016 clean and occluded | Bowl is pushed to the right without tilting or lifting | B; two regressions |
| 002 clean | Empty grasp before and after fine-tuning | Persistent A |
| 007 clean | Slight leftward tilt after contact, no lift | Persistent B |
| 013 clean | Suspected empty grasp; slight leftward tilt/translation may come from the arm or outer contour of a finger | Uncertain contact source; do not force A/B classification |
| 018 clean | Empty grasp; original clip had contact without lift | Goal failure persists; observed attempt subtype changes |
| 011 occluded | First grasp attempt and placement reported successful | Local repair; complete finishing not established by this alone |
| 005 and 014 clean | First grasp attempt and placement reported successful | Preserve as round-1 observations; post-placement behavior not established |
| 012 occluded | Lift, drop onto table, then two further contact attempts move bowl without lifting | Post-lift drop followed by failed retries |
| 019 occluded | Grasp/place/release, then persistent jitter and repeated contact; bowl remains on plate | Manual physical placement differs from automatic goal failure; preserve both |

The remaining five occluded persistent-failure cases do not have detailed round-1 observations supplied here. Do not copy baseline or round-2 descriptions into those missing entries.

### 5.3 Second fine-tuning round and subsequent extended observations

| Case / series | User-reported observation | Interpretation / boundary |
|---|---|---|
| Round 2: 005 clean | Gripper slightly tilts the bowl right; contact without a secure grasp; bowl remains on table | Notebook 08 log section 6: original success, round-2 failure; section 7: first-round success |
| Round 2: 014 clean | Gripper moves the bowl slightly right; contact without a secure grasp; bowl remains on table | Notebook 08 log section 6: original success, round-2 failure; section 7: first-round success |
| Round 2: 001 clean, 016 clean, 005 occluded, 016 occluded | First grasp attempt and placement reported successful | Repairs relative to round 1; not a full-finish certificate |
| Round 2: 012 and 019 occluded | First grasp attempt and placement reported successful | New automatic-goal repairs relative to original |
| Round 2: 011 occluded | Contact causes slight tilt, no lift | B; loses round-1 repair |
| Original extended 008 clean and occluded, 7 s | Grasp/move above plate; opening followed by about 1 s apparent attachment to left finger, then bowl drops onto plate; subsequent jitter without bowl contact | Final bowl on plate, but delayed release and jitter |
| Original extended 019 occluded, 14 s | Grasp/place/release/withdrawal; jitter; no re-contact before about 7 s, repeated contact from then to end; bowl remains on plate and visibly moves | Placement retained; finishing unsafe/unstable by user criterion |
| Round-2 extended 008 clean, 7 s | Bowl appears attached only to left finger without right-finger contact; release delayed until about second 6; then video ends | Bowl eventually on plate; later jitter/re-contact unknown |
| Round-2 extended 008 occluded, 13 s | Grasp/place/release/withdrawal, persistent jitter, contacts at approximately 10 and 12 s; bowl remains on plate | Post-release re-contact |
| Round-2 extended 019 occluded, 6 s | Grasp/place/release/withdrawal, persistent jitter, contact at approximately 5 s; bowl remains on plate | Post-release re-contact |

Extended clips above require linkage to their particular historical evaluation series; they are not assumed to be the short early-termination videos. The 005/014 clean observations are now explicitly assigned to round 2 based on notebook 08's log, rather than inferred from an ambiguous conversation reply.

### 5.4 Demonstration and dataset inspection

| Material | User-reported observation | Role |
|---|---|---|
| Original demonstration episode 1500, validation | Video ends while bowl is still held; no genuine release | Demonstration endpoint limitation; not an optimization sample in this project |
| Original demonstration episode 1679, training | Release without safe withdrawal or upward clearance | Training-demo finishing coverage concern; no proof of sole failure cause |
| Scripted pilot 008 clean | Bowl truly separates and stays on plate, arm lifts clear without re-contact, last ~2 s stable; about 1 s delayed separation during placement | Scripted teacher demonstration; not autonomous policy improvement |
| Three scripted batch demonstrations: 001 clean, 005 clean, 016 occluded | Genuine release, bowl remains on plate, upward withdrawal, no further contact, final ~2 s stable, no delayed separation | User-approved demonstration quality; scripted finishing remains explicit |

### 5.5 Notebook-15 autonomous finishing evaluation

| Policy / case | User-reported observation | Final state / boundary |
|---|---|---|
| Original / 008 clean | Apparent attachment at placement, then release; continued arm/gripper jitter and repeated bowl contact | Bowl remains on plate at end; full finishing not achieved |
| Original-only control / 008 clean | Delayed separation, release, jitter, contacts near 9 and 12 s, later upward motion with continued jitter | Bowl remains on plate |
| Finish-augmented / 008 clean | Failed grasp; only rightward bowl tilt | No successful lift |
| Original / 020 clean | First grasp/place followed by jitter and contact nearly displacing bowl; second grasp returns it to plate center; later jitter without further contact | Bowl fully released, slightly tilted, entirely on plate; preserve automatic final-goal disagreement |
| Original / 020 occluded | Similar second grasp, but second placement at lower right causes tipping | Bowl partly on table and partly on plate |
| Finish-augmented / 020 clean and occluded | One grasp/place, release, later jitter without further bowl contact | Local improvement in re-contact behavior; residual jitter |
| Original / 001 clean | One grasp/place, followed by repeated bowl contact | Bowl upright and entirely on plate at end |
| Finish-augmented / 001 clean | Initial grasp/place, renewed contact, then fingers spread inside bowl and lift it; bowl drops | Bowl tilted, partly on table and partly on plate |

### 5.6 Notebook-22 evaluation of model 19

| Case | User-reported observation | Boundary |
|---|---|---|
| 001 clean | First attempt only slightly shakes bowl; next two attempts miss | B → A → A |
| 005 clean | First attempt misses, subsequent attempt also misses | A → A |
| 016 occluded | First attempt moves bowl slightly right; subsequent attempt fails | B, then failure subtype unspecified |
| 008 clean | First attempt misses; later gripper wavers near plate before rising | Subsequent grasp attempts uncertain; do not invent their number |

All four cases fail automatically in the previously audited summary. The manual uncertainty does not change the decision to stop this candidate; additional viewing is not required just to justify that stop.

## 6. What to retain in the final package

| Material | Decision | Reason |
|---|---|---|
| Audited original, RQ1 and visual-occlusion result records | Retain as evidence | Recompute main comparisons |
| Round-1/2 protocols, splits, validation arrays, reports and export identifiers | Retain | Explain model selection, negative outcomes and provenance |
| Finish dataset manifest, actual recorded-action preparation code and sampling schedules | Retain | Explain the intervention and held-out episode |
| Original-only / finish-augmented / model-19 reports, matching evaluation plans and essential traces | Retain | Support finishing and capability-retention conclusions |
| Model-20 report | Retain as supporting offline negative evidence if discussed | No claim of closed-loop improvement |
| Scripted demonstration generation | Retain separately, clearly labelled | Teaching data and feasibility, not policy success |
| Selected success, failure, re-contact and regression videos | Retain with exact series and path index | Avoid selected-success-only presentation |
| Every notebook as a top-level reproduction entry | Do not include all | Extract a minimal, documented workflow and retain history separately |
| Installation output, duplicate notebooks, old patch backups, cache folders and empty lock files | Exclude from public package | Do not support reported scientific conclusions |
| Large policy weights and recovery checkpoints | Retain pending final evidence mapping; decide individually later | Failed candidates may still be needed for reproducing reported findings |

No file has been deleted, moved, or overwritten in Google Drive. This is a retention plan, not completed repository cleanup. The selected training scripts have been inspected, but clean reproduction entry points and dependency links still need assembly.

## 7. Remaining closure tasks

1. Link remaining manual observations to exact historical video paths. The round attribution of 005/014 clean has been resolved using notebook 08; no additional review is needed for that attribution. Attach separate annotations to stale `manual_review: pending` summaries without rewriting raw automatic outcomes.
2. Assemble the minimal reproduction entry points and result-recalculation script with English documentation. Trace dependencies before extracting code; do not merely rename all old notebooks.
3. Consolidate the later trace evidence and the offline diagnostics that will actually be discussed in the final report. Notebook 17/18 details are not newly audited here.
4. Draft the final English report and README, supported by the two audit documents and source files. State the bounded negative finding: the tested fine-tuning interventions did not yield a verified better autonomous policy. Do not infer catastrophic forgetting, a proven label-conflict mechanism, or architectural impossibility from these observations alone.

No additional training or long simulation is authorized by this checklist. Missing historical observations can remain explicitly unknown if they do not affect the conclusion.

## Appendix A. Model export identifiers

These hashes are recorded identifiers cross-checked across training and evaluation records; they are not new hashes of downloaded full policy weights.

| Candidate | Recorded model.safetensors SHA-256 |
|---|---|
| round1 | `df1f27f3e23cf131c6eb1b215d25ea16d9f8c0bfd7d42d60ac71b36aa5d6f406` |
| round2 | `0c4e5d8416674322ce6791d0fa82c45867079f16fafbae32f3f099f21aad4970` |
| original_only | `e6095e61f040da08a1d3c76fcd40d21b017983a2f92249798df4e60adf96d91f` |
| with_finish | `f4dd0d94d34839079846134f7887960d6fa7d99d1f623cb60329e2b749b3b6fe` |
| model19 | `ab5daaf1068f7ea7b7edb5f814ab92169b4b509d4bdc256db921ac3c833807c4` |
| model20 | `2337fd377f58ca020b5be665fd293e6edbaa0ebfc08878745bbc1788d352420a` |

## Appendix B. Read sources and byte hashes

The following source files were read from the project Drive during this closure pass. Hashes identify downloaded source bytes; cloud files were not modified. The automatic v1 outcome ledger remains the companion source for historical goal flags. Manual observations above come from the user's project messages, not these training files.

| Source path relative to this audit | Drive source | SHA-256 |
|---|---|---|
| `eval19/plan.json` | [source](https://drive.google.com/file/d/1prmU-dfqE4KmcDNMn1iGywh__6JIiQYh/view?usp=drivesdk) | `8808d2e9916dd1b84f7e450f21232d210729c443a1a08e46dadd1b11456e4317` |
| `finish_eval/plan.json` | [source](https://drive.google.com/file/d/1SO90tCtFt5MsUgNf052me2EmbKeGo7QA/view?usp=drivesdk) | `d6141734e50e80be14a03b118eac84851f024186cee151d8bba4dfd4ef3faa3a` |
| `round1/baseline_validation.json` | [source](https://drive.google.com/file/d/1fWS6Uhh_r7tt8PUuuy3byhHlUsPS1AGN/view?usp=drivesdk) | `a15ef2a0521ca1dc6989292888a528de954d0799bb2d1a8aeb5a6d574d6ea1e9` |
| `round1/checkpoint_000250.json` | [source](https://drive.google.com/file/d/13HcVdiLYhnlejBi9tTP8uoAfC8QFvO7P/view?usp=drivesdk) | `55837dc7a92c09c990d34e88f20005ad162d20d94b743fd2134829280349ad23` |
| `round1/checkpoint_000300.json` | [source](https://drive.google.com/file/d/19mpJ9F9si9lkie9V_GdVdoc5oxrN0uWl/view?usp=drivesdk) | `e471cb0bb70b0796cf07766c2c13f14b84d6f78b0ebc61a9ce8a4cc62a51c840` |
| `round1/evaluation_plan.json` | [source](https://drive.google.com/file/d/1f_BQO1NuENHk0fSgmvG5Bw7aI-uqhG6P/view?usp=drivesdk) | `f22da099245ab11e07abf457e4e3fabc34ab70e8e277265c95eef80410de6ea1` |
| `round1/progress.json` | [source](https://drive.google.com/file/d/17TmuO2NtKrwKyoOiv3snSyQdMGLSbNWj/view?usp=drivesdk) | `e5ff1c2f17af9a807ac6e30c75d6af72766394944b84f95175fe31583ab3e748` |
| `round1/protocol.json` | [source](https://drive.google.com/file/d/1PGJAu6iA3Xm5oSA0IfEjjaWdTk-WppKI/view?usp=drivesdk) | `3e72108088d09603466b1d6b84a1c2ae24a6026fda8eb892478021d0c2c3acf7` |
| `round1/selected_model/config.json` | [source](https://drive.google.com/file/d/11VNODZB0BDeLGf9uegCaYJhgu_6Evz7g/view?usp=drivesdk) | `e50464d33e3a8a0247991ee6ef272606b69b1fc61bad04b937072427ce904ae8` |
| `round1/selected_model/policy_postprocessor.json` | [source](https://drive.google.com/file/d/1XN6OPmuq7IouFdUBy1kSexMlPlvpL9Pi/view?usp=drivesdk) | `ad2d22405cb63be4932c4a4d9a69bc9953df02a01235c5f420b679dbe6bfe7c5` |
| `round1/selected_model/policy_preprocessor.json` | [source](https://drive.google.com/file/d/1qX19zkyCt50TOdSb6AZ_cA81ZiSR6U2s/view?usp=drivesdk) | `29267ff4423fcce93bbd76ad0f386905f9de5d267ca13379a5795a514a81b7a9` |
| `round1/training_report.json` | [source](https://drive.google.com/file/d/1cqJSBnncjx-GOJ2PIg2ip9kICSMIfXnj/view?usp=drivesdk) | `3edd4b40a304c3fc8641c7b5ffa3d216eec229f0df61336e82e8216b2c3e4cfc` |
| `round2/baseline_validation.json` | [source](https://drive.google.com/file/d/1vnt52cVcJDoK-9L128d9ciVpCb3z55OQ/view?usp=drivesdk) | `a15ef2a0521ca1dc6989292888a528de954d0799bb2d1a8aeb5a6d574d6ea1e9` |
| `round2/checkpoint_000100.json` | [source](https://drive.google.com/file/d/1JYqShmgi9EhQTQBNAq1JfmkSV-B-dI1h/view?usp=drivesdk) | `3861989ec41fd64b3abb058a52a76d5c832d3ff8f54eaa2af4d93d1a5d7356ae` |
| `round2/checkpoint_000150.json` | [source](https://drive.google.com/file/d/1i9CvXMbkorLdc5wrO0s0-xvaF6yl0i5S/view?usp=drivesdk) | `9c37ec053fb98c2a5117e7802b1d6c8e13f5d78476cb17370b75da4961907215` |
| `round2/evaluation_plan.json` | [source](https://drive.google.com/file/d/1zGwQou-bavxst_ozV3DkGyrkhD17-xna/view?usp=drivesdk) | `1518e742a9a8d5fa3c896e4a7d41d19083d59267250eec7bb83a50a761053a04` |
| `round2/progress.json` | [source](https://drive.google.com/file/d/1X_Xf81K1YdBo919chd9jRG6DTY2qqwqD/view?usp=drivesdk) | `c8c4f64a00c9ac34c178ecafd118ffbb4462778dde1283fbc6374da2bf1ecb7e` |
| `round2/protocol.json` | [source](https://drive.google.com/file/d/1huojP5PHhySaF8ZRM-bsyreaPD0C2fzw/view?usp=drivesdk) | `d81e7301ea80266cb5c4c410b0f45682cf21cc7cc3dff596397ef8730755bbc9` |
| `round2/selected_model/config.json` | [source](https://drive.google.com/file/d/1ny6PK5ECjwEYnJXexYSEzRKyHM0cVT67/view?usp=drivesdk) | `9aec46548bb90d7564b9690cf42f235c79b8cf63e8f4cb4b29104212f1ff21ca` |
| `round2/selected_model/policy_postprocessor.json` | [source](https://drive.google.com/file/d/1tVSSWLXiq2eTqv-ANdr14d1fvvSqxKYm/view?usp=drivesdk) | `ad2d22405cb63be4932c4a4d9a69bc9953df02a01235c5f420b679dbe6bfe7c5` |
| `round2/selected_model/policy_preprocessor.json` | [source](https://drive.google.com/file/d/1NVzwtLQUkt-BLU0WKuZkPcleuX1ezSiZ/view?usp=drivesdk) | `29267ff4423fcce93bbd76ad0f386905f9de5d267ca13379a5795a514a81b7a9` |
| `round2/training_report.json` | [source](https://drive.google.com/file/d/1iaOs8a4RY4FXhia8HDa7gI6avdjqRous/view?usp=drivesdk) | `5e6dd10d51e44915eecadfa5c23b1d70bb7dd39a2607958fd7940670805e8efd` |
| `scripts/07_task0_finetune_v1.py` | [source](https://drive.google.com/file/d/1MWaS4T-YwAK38rF_ZKjUM_2Q7N2-LAlk/view?usp=drivesdk) | `56dd11fac0639f52741cc1516ebcaf39266f094737b6e1f993b9195ec51bedb6` |
| `scripts/10_low_lr_fixed150_v1.py` | [source](https://drive.google.com/file/d/1pIFuto6M0ywduzqWaAbvucBYaIHh2pAW/view?usp=drivesdk) | `b28a5a6298681d46ae6ad8f8921cf60321fc6ad2e15193a3cfb0600125b2534c` |
| `scripts/17_finish_learning_pilot_v1.py` | [source](https://drive.google.com/file/d/1Z0gbiOxmtefhoZyV3t3NdoTNgcoYOJlG/view?usp=drivesdk) | `8115d8a740a1861d3e830c737878afeb17cea84a7c01bc5feefe84774a12a634` |
| `scripts/22_hold_lr1e5_fixed50_v1.py` | [source](https://drive.google.com/file/d/1cyJxo3yS4I8rY4XoUV41hCtcrdiw4_Ub/view?usp=drivesdk) | `113470f7ec915b9dda28cd086cce71a01834c3d7f06045a835177f8c3a8576f1` |
| `scripts/23_phase_mix_lr1e5_fixed50_v1.py` | [source](https://drive.google.com/file/d/1H-Ndm6c6MnFgwjsLcl0t_9SMJ2JtyVLl/view?usp=drivesdk) | `dfd19819fbd861cf110cf836a741f2362be58dbb9cbd60a636de757fec0146d9` |
| `task0_demonstrations_v1/plan.json` | [source](https://drive.google.com/file/d/1DLKg5uSTAXX3qXPR_T0u6SILqcGv12XB/view?usp=drivesdk) | `8f1c8f96c59c8926712040dd56e04ad3f47e2ddbd0c4dbd39b707ff44608fa3c` |
| `task0_demonstrations_v1/preparation_report.json` | [source](https://drive.google.com/file/d/1vXGDBhQCKALKj2M86kuQ9TSyzVrwX5vK/view?usp=drivesdk) | `5cc1d3349e9dfcdc7eb4d1cbfc11c3901c4a44d54a130c16553adb396b588a17` |
| `task0_demonstrations_v1/split.json` | [source](https://drive.google.com/file/d/1sfjKe3P24wp-3g7p7Z66NVSGsyRlMgP1/view?usp=drivesdk) | `20b4954173b4fc25ead6552052483ef34586f40c3ebe1ba5edc161638d5b889e` |
| `task0_finish_dataset_v1/manifest.json` | [source](https://drive.google.com/file/d/1Q9TIBKT2UyA6NkkJMD_3N2aZR_uQlz26/view?usp=drivesdk) | `4099edf622ff18b11350c44005b22ae685c2b0c9ee37f12c87a4979b6fcc2dbb` |
| `task0_finish_dataset_v1/package_report.json` | [source](https://drive.google.com/file/d/1xlC0QLCdl2BS4y916T9yjP1gH23Krug2/view?usp=drivesdk) | `5df072ceaa0a1875b8c0080510a76809ddd66f6714e59bd1aecac644740d88a0` |
| `task0_finish_dataset_v1/protocol.json` | [source](https://drive.google.com/file/d/104HvgS5r8TmUaWY4Yz3EqFPvft3H3mHp/view?usp=drivesdk) | `e4c3cb047520a71acaa7d7822ab28e42dfe46c5b24b6625ea77048cdea58daa8` |
| `task0_finish_dataset_v1/sampling_schedule.json` | [source](https://drive.google.com/file/d/1ZoxK6BwpPhsPaRNMct4vV6FROnhMZXXd/view?usp=drivesdk) | `7032b0c4a98272461f7e91a4b1e0f92fe12cb70d470e79ecb98cd73ec4976229` |
| `task0_finish_dataset_v1/validation_anchors.json` | [source](https://drive.google.com/file/d/1Mz0J8x5fs0DQ9V3fqeSMxlbBx2y7296S/view?usp=drivesdk) | `bab927fa9a2ed90b1e132137a65255c916782be194029d307e9729fb37bc5d30` |
| `task0_finish_learning_pilot_v1/original_only/reload_check.json` | [source](https://drive.google.com/file/d/17JyeOrKHqzmcqCHrKZKX0pewCX4IHnrL/view?usp=drivesdk) | `a7796ed371e5e240ff7f08ca0011bf09948ea6179768f8ca27ba3636216e46eb` |
| `task0_finish_learning_pilot_v1/original_only/training_report.json` | [source](https://drive.google.com/file/d/11rpEv0FQ4YV7m3VJb3e65-hR1lCj9b-Z/view?usp=drivesdk) | `7a1a158360a70c5031cb5413e66cd8e4fbde9c790257aadfefec202a407d5103` |
| `task0_finish_learning_pilot_v1/run_protocol.json` | [source](https://drive.google.com/file/d/1MGivLrP0QdDywaK838GazCFurDSI-Xkl/view?usp=drivesdk) | `7bb4d7705553843c135b93c7a4a441f7f48b49830cd6aad165d31423e8ac8f06` |
| `task0_finish_learning_pilot_v1/with_finish/reload_check.json` | [source](https://drive.google.com/file/d/11ci2HmoGQNDdkcyIV1cAunb8xfgbcVyS/view?usp=drivesdk) | `871d11ab23839222734520e43abdd65d3489ecb9661be29e47559065fc9df672` |
| `task0_finish_learning_pilot_v1/with_finish/training_report.json` | [source](https://drive.google.com/file/d/14uV8d87d6GB4LA0kfbb1J_jErKbeVMcX/view?usp=drivesdk) | `657287c2128e32a19ee0547be6a146488eac38e66b9218aba8eb390f8a1cc7ff` |
| `task0_hold_lr1e5_fixed50_v1/hold_lr1e5_fixed50/reload_check.json` | [source](https://drive.google.com/file/d/1TBOpMtG725wghiIhoXYOJHzTQ0ga07eh/view?usp=drivesdk) | `34a25d16262fa245f9b84fc5cc346ee4e932814b15c7e511448867aee1cab83b` |
| `task0_hold_lr1e5_fixed50_v1/hold_lr1e5_fixed50/training_report.json` | [source](https://drive.google.com/file/d/1XcHdWFuKTAr_ocES2yTy_dGy24RtAJY4/view?usp=drivesdk) | `c9a02cbd967e59ef3ba504bd6fcdb169cb07f628266acd4363dbea9fad09c0c4` |
| `task0_hold_lr1e5_fixed50_v1/run_protocol.json` | [source](https://drive.google.com/file/d/1hai2AhfWD61MvDnEj75p6KhU5dv6G908/view?usp=drivesdk) | `ce00ebe5835d9ba6871bbef9aa3123156bf7109751b4154bf1e31956706273e3` |
| `task0_phase_mix_lr1e5_fixed50_v1/phase_mix_lr1e5_fixed50/reload_check.json` | [source](https://drive.google.com/file/d/1HHxoHxDk7LPkK93F2SyiYH7gOOJhNG-0/view?usp=drivesdk) | `478bac62d425c8881bf73bcfccf7072fa1789e0e5335a0595701912ced147434` |
| `task0_phase_mix_lr1e5_fixed50_v1/phase_mix_lr1e5_fixed50/training_report.json` | [source](https://drive.google.com/file/d/1eNz35WJfhzE0jlh_V7GlmV8Afk_GvyrH/view?usp=drivesdk) | `08cbb36b2de6ae63f42b139fe9826f77683c5432769150808d3b8a147f85683f` |
| `task0_phase_mix_lr1e5_fixed50_v1/run_protocol.json` | [source](https://drive.google.com/file/d/1mPSrShSM-Tfh3Q_s1xUqh9sToZX14NDa/view?usp=drivesdk) | `e7675131dcf5725124d0a14314df88ed061a9b0d8e700a702c14c75f2c15113a` |


## Appendix C. Notebook-log correction evidence

The current Google Drive copies were read on 25 October 2026. Neither source notebook was modified.

- [Notebook 08 — final experiment log](https://colab.research.google.com/drive/1y9bsPROi0UyLWLWpPHIqtEmLH8IMFCFE#scrollTo=1H6BO1veGbyi), markdown cell 11 (zero-based), titled “SmolVLA second fine-tuning round — low-learning-rate comparison and simulation evaluation.” Section 6 explicitly states original success and second-round failure for 005 clean and 014 clean, followed by the rightward-tilt and rightward-translation observations respectively. Section 7 explicitly states that both succeeded on the first grasp attempt in round 1 and failed in round 2.
- [Notebook 04 — final experiment log](https://colab.research.google.com/drive/1vakHYZiaUQ4iO_Cx_tL_v9jTCbxYlDDo#scrollTo=87vALz4clWy7), markdown cell 9 (zero-based), titled “SmolVLA first fine-tuning round — paired simulation evaluation.” Section 5 lists the six regressions as 001 clean, 005 occluded, 008 clean/occluded and 016 clean/occluded; it does not identify 005 clean or 014 clean as regressions. Notebook 08 supplies the explicit first-round success statements.

Verified attribution: **005 clean and 014 clean: original success; round 1 success; round 2 failure.** This correction retracts the earlier claim of a conflict in the user's observations. It changes manual attribution only, not the automatic results.

| Read source | SHA-256 of downloaded notebook bytes |
|---|---|
| `04_smolvla_finetuned_evaluation.ipynb` | `2da864313d463dc1905c398bd03894b9a34b70d9eefafdd07f1a4fe6dd775ff3` |
| `08_smolvla_low_lr_evaluation.ipynb` | `1be15452d9ca45f98a5517848f6724b23c40034fe27d4e94df6ee4e7aeec4dff` |
