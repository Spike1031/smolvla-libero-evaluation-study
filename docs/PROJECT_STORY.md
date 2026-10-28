# Project progression and conclusions

## Starting questions

This single-task study evaluated a pretrained SmolVLA policy: action execution frequency, central image masking, and adaptation using task demonstrations. Reliable post-placement finishing became a separate requirement after visual review exposed limitations in early termination. It was not a preregistered starting objective.

| Stage | Question and action | Recorded result | Reason for next step |
|---|---|---|---|
| Execution setting | Compare 1 versus 5 executed actions before replanning, on 20 paired states | Goal 16/20 versus 15/20; total recorded evaluation time 42.1% lower for 5 | Establish a working baseline; test visual sensitivity |
| Occlusion | Mask central 6.25% of main-camera pixels; leave wrist camera unchanged | Clean 16/20; masked 12/20 | Test whether task-demonstration adaptation helps |
| First adaptation | Train on task demonstrations; select a trained checkpoint using validation loss | 23/40 versus original 28/40 | Local repair occurred alongside more regressions; try a more conservative protocol |
| Second adaptation | Lower learning rate; fixed update-150 export | 28/40, with two repairs and two regressions versus original | Aggregate recovery was not overall improvement; inspect behavior |
| Endpoint review | Inspect policy videos and selected original demonstrations | Release, withdrawal and stability were not established by early short clips; selected demonstration endings were also incomplete | Define additional finishing checks and construct finishing demonstrations |
| Finishing intervention | Build scripted finishing demonstrations and train policies; compare on extended rollouts | In notebook 15, each of three groups had 0/10 automatic complete finishes | Diagnose failed adaptation without treating scripted demonstrations as autonomous successes |
| Final small gate | Evaluate model 19 on four known development cases | Candidate goal 0/4, matched original 4/4; no automatic complete finishes | Stop this candidate line; preserve negative results and evaluation limitations |

## What was achieved and what was not

The study produced paired benchmark-goal comparisons, selected failure analyses and reproducible calculations. It did not produce a verified better autonomous policy, a robust finishing solution, a generalization claim across tasks, or a fully identified cause of training failure. Early short rollouts cannot be retroactively scored for unseen post-goal behavior. Their complete-finish rate is unknown, not zero and not equal to their benchmark-goal rate.

The present stop is a decision to stop an unsuccessful adaptation line. Evidence interpretation and release preparation are separate from proving a model improvement. No new GPU experiment is required merely to explain these records.

## Reading order

Read `EVALUATION_PROTOCOLS.md`, then `CASE_RECONCILIATION.md`, then the numerical tables. The existing report remains a dated synthesis; these audit notes provide the more explicit protocol explanation added on 28 October 2026. Unresolved cases are not silently relabeled.
