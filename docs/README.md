> Historical document dated 25 October 2026. For the current source inventory and protocol explanation, read [experiment sources](../experiments/README.md) and [training/source audit](TRAINING_AND_SOURCE_AUDIT.md). Historical completion-status statements below are retained as dated provenance.

# SmolVLA: Single-Task Evaluation and Adaptation Study

This project studies action execution steps, central visual occlusion, and limited-demonstration adaptation of a pretrained SmolVLA policy in LeRobot/LIBERO. The task is to pick up the black bowl between the plate and ramekin and place it on the plate (`libero_spatial`, task 0).

**Closure status, 25 October 2026:** the recorded-result recalculation entry point is ready. The final research report and a fully assembled training/simulation reproduction environment are not yet complete. No verified better autonomous fine-tuned policy was obtained. The present adaptation branch is stopped.

## What this entry point does

`reproduce_results.py` reads the original committed JSON records, checks coverage, plans, seeds and saved summaries, and generates English CSV tables and a JSON audit. It uses only the Python standard library. It does not import a model, install GPU dependencies, run simulation, change model weights, alter source results or judge videos.

`00_smolvla_reproduce_recorded_results_cpu.ipynb` embeds the same code and is the standalone Colab entry point. You do not need to run both the notebook and the script.

This is **reproduction of reported calculations from recorded evidence**, not a new independent reproduction of policy behavior. New rollouts and retraining are separate, resource-intensive operations and are not required for closure.

## Quick start: Colab CPU

1. Open `00_smolvla_reproduce_recorded_results_cpu.ipynb` in Colab with a CPU runtime.
2. Run A to mount the Google Drive account holding `embodied-smolvla`.
3. Run B to load the embedded analysis code.
4. Run C to audit the four historical series and create tables under a new `/content/smolvla_closure_tables_*` directory.

The notebook does not automatically launch training or simulation. Results are written to temporary Colab storage, outside the source Drive project. Copy them elsewhere if you want to retain them after the runtime ends. Missing or inconsistent inputs stop the audit; nothing is silently regenerated.

## Quick start: local Python

Use Python 3.10 or later. No additional packages are needed:

```bash
python reproduce_results.py --project /path/to/embodied-smolvla --output /path/to/new-closure-tables
```

The output directory must be empty or absent and outside the source project. For a deliberately partial audit, use `--series original round1 round2`; partial coverage is explicitly recorded in `audit.json`.

## Required evidence layout

All paths are beneath `embodied-smolvla/results/`:

| Series | Directory | Per-state conditions |
|---|---|---|
| RQ1 | `paired_task0_states0_19_v1` | `k1`, `k5` |
| Original visual baseline | `occlusion_center_task0_states0_19_v1` | `clean`, `occluded` |
| Fine-tuning round 1 | `finetuned_task0_clean_occluded_v1` | `clean`, `occluded` |
| Fine-tuning round 2 | `low_lr_fixed150_task0_clean_occluded_v1` | `clean`, `occluded` |

Each directory needs `plan.json`, `summary.json`, and 40 committed files named `state_000/<condition>/result.json` through `state_019/<condition>/result.json`. Attempts, videos and model weights are unnecessary for this calculation entry point; retain them separately for provenance and physical-behavior analysis.

## Reported results and interpretation

| Comparison | Historical benchmark-goal result |
|---|---|
| RQ1, execution steps 1 vs 5 | 16/20 vs 15/20; cumulative evaluation time 3385.618221 vs 1959.911733 seconds |
| Original, clean vs occluded | 16/20 vs 12/20 |
| Round 1, clean / occluded | 13/20 / 10/20; total 23/40 |
| Round 2, clean / occluded | 14/20 / 14/20; total 28/40 |

These are paired development states. Forty condition-rollouts are not forty independent initial states. The observed 42.11% lower RQ1 evaluation time is not a pure inference-speed measurement; episode lengths differ. Direct chunk-generation call counts were not stored in those result records.

Historical success means triggering the benchmark goal under the early stopping protocol. It does not establish genuine release, safe withdrawal, no re-contact or two seconds of stability. Scripted finishing demonstrations are not autonomous policy success.

**Corrected case attribution:** 005 clean and 014 clean succeed in the original and first-round records, and fail in round 2. Rightward tilt (005) and rightward translation (014) without lift belong to round 2. Notebook 08 explicitly establishes this; the user's notebook logs did not contain the conflict previously alleged during consolidation.

## Output files

| Output | Contents |
|---|---|
| `summary.csv` | Per-series goal counts, rates, action totals and elapsed time |
| `episode_ledger.csv` | Each committed record, relative source path and source-byte SHA-256 |
| `transitions.csv` | Original-to-candidate paired repairs and regressions |
| `audit.json` | Explicit pass status, scope and selected series |

No output is a new success label or video annotation. Exact input rows support the recomputed aggregates. Source plan hashes use the historical sorted-key JSON serialization.

## Minimal historical workflow map

Historical scripts are preserved in the Drive project. This is a dependency map, not an instruction to run all notebooks or proof that a clean runtime has been retested.

| Purpose | Source entry point | Required evidence/dependencies |
|---|---|---|
| RQ1 original evaluation | project `03_paired_eval_v1.py` | Locked original model, LIBERO assets, environment and plan |
| Visual-occlusion baseline | project `04_occlusion_center_v1.py` | RQ1 environment/compatibility records and mask protocol |
| Round-1 training | `finetune/07_task0_finetune_v1.py` | Prepared original demonstrations, split, original processors and training adapter |
| Round-1 evaluation | [Notebook 04](https://colab.research.google.com/drive/1vakHYZiaUQ4iO_Cx_tL_v9jTCbxYlDDo), project `08_finetuned_eval_v1.py` | Selected update-150 export and original visual baseline |
| Round-2 training | `finetune/10_low_lr_fixed150_v1.py` | Same demonstration split and locked original start; fixed update 150 |
| Round-2 evaluation | [Notebook 08](https://colab.research.google.com/drive/1y9bsPROi0UyLWLWpPHIqtEmLH8IMFCFE), project `12_low_lr_fixed150_eval_v1.py` | Round-2 export; reused original/round-1 comparisons |
| Later finishing work | notebooks 11–22, selected sources only | Separate scripted-data, training, diagnostic and policy-evaluation stages |

Script filename prefixes and notebook numbers differ. Minimal end-to-end extraction must resolve the training adapter, model snapshots, assets and initialization before any rerun. It is still pending; this package does not pretend those dependencies are bundled.

Key historical environment: Python 3.12.3, LeRobot 0.6.1, PyTorch 2.11.0+cu128, hf-libero 0.1.4, robosuite 1.4.0, NumPy 2.2.6, Tesla T4. These are recorded simulation/training versions; the CPU calculation script does not require that stack.

## Final delivery and remaining work

Retain necessary source/configuration files, committed result evidence, training provenance, and indexed success/failure/regression videos. Keep duplicates, installation output and exploratory notebooks in a historical backup rather than exposing all 22 as reproduction steps. No Drive source files have been deleted or reorganized.

The next deliverable is the English research report. Its later finishing section still needs the selected trace/diagnostic evidence linked; this CPU entry point covers the early 160-rollout main comparisons only. A public repository, a clean GPU rerun and a publication-quality scientific claim are not implied by this package.

## Validation of this calculation entry point

Locally tested on the existing 120-record original/round-1/round-2 evidence archive: all saved counts, rates and elapsed-time sums matched; corrected cases 005/014 passed. A deliberately changed seed was rejected. The notebook JSON parsed and every code cell compiled successfully; full nbformat schema validation was unavailable in this environment. Subsequent assembled-package validation executed all four branches against all 160 included source records and matched the saved summaries. The Colab mount flow has not been run here.
