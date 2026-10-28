# SmolVLA: Execution, Occlusion and Adaptation

A single-task empirical study using SmolVLA, LeRobot and LIBERO. The tested fine-tuning interventions did not produce a verified better autonomous policy. The report explains the controlled comparisons, negative results and limits.

## Start here

1. Read the [project progression](docs/PROJECT_STORY.md): why each experiment was run, what failed and why this adaptation line stopped.
2. Read the [success and termination protocols](docs/EVALUATION_PROTOCOLS.md). Early goal attainment is not complete finishing.
3. Inspect [case reconciliation](docs/CASE_RECONCILIATION.md), [main tables](tables/summary.csv) and [paired repairs/regressions](tables/transitions.csv).
4. Browse the [historical experiment sources](experiments/README.md) and [unified video index](videos/README.md). The single catalogue contains 11 unique videos, with six recommended clips and five supplementary clips. All have anyone-with-link reader permissions; the owner confirms anonymous playback for the six recommended clips.
5. Read the [dated report](report/research_report.pdf), supplemented by the protocol notes above, and reproduce the recorded calculations below.

## Reproduce the tables from included evidence

From this folder, with Python 3.10 or later:

```bash
python analysis/reproduce_results.py --project evidence --output ../smolvla_recalculated
```

Choose a new or empty output directory outside `evidence`. No third-party Python packages are required for the main table calculation. All 160 committed result JSON files, plans and summaries are included. Missing or inconsistent records cause an error; the script never launches a model to fill a gap. It records source-byte hashes and generates a ledger, aggregate table and transition table.

The packaged script was executed successfully on all 160 records. The earlier 120-record check also confirmed rejection of a deliberately corrupted seed. This is reproduction of calculations from historical evidence, not a new independent execution of the policies.

## Recalculate the later trace checks

This optional check needs NumPy (`pip install -r analysis/requirements-finishing.txt`):

```bash
python analysis/verify_later_traces.py
```

It checks 38 trace records, including four reused baselines, against stored discrete scores. It does not rerun physical simulation or claim fresh visual video review.

## Main findings

| Comparison | Historical benchmark-goal result |
|---|---|
| Execute 1 vs 5 actions before replanning | 16/20 vs 15/20; 42.11% lower recorded total evaluation time for 5 |
| Original clean vs central occlusion | 16/20 vs 12/20 |
| First fine-tuning round | 23/40, compared with original 28/40 |
| Second fine-tuning round | 28/40, with two repairs and two regressions |
| Later complete finishing | 0/10 for each of original, original-only and finish-augmented policies |
| Final four-case gate | Model 19: 0/4 goal attainment; matched original: 4/4 |

Cases are paired development initial states, not independent condition-rollouts. Recorded evaluation time is not pure inference time. Early goal attainment does not certify release, withdrawal or stability. Scripted teacher demonstrations are not autonomous policy results.

## Folder map

| Path | Purpose |
|---|---|
| `report/` | English PDF report and editable Markdown |
| `analysis/` | Standard-library main result calculator; historical NumPy finishing-score functions |
| `evidence/results/` | Four series of committed raw results and protocols |
| `evidence/later/` | Later policy-only trace archives for finishing and model-19 gate |
| `tables/` | Recomputed main tables, episode ledger and later audit summary |
| `videos/` | Verified links and indexed user observations; binaries remain in Drive |
| `docs/audits/` | Outcome audit and corrected training/manual provenance supplement |
| `docs/README.md` | Detailed historical entry-point map and computational context |
| `notebooks/` | Optional Colab CPU calculator for the existing Drive project |
| `MANIFEST.sha256` | Hash inventory of package files |

## Source and training inspection

The [source map](experiments/README.md) includes all four early evaluation runners, both initial training scripts, the training smoke test and both later evaluation runners. Six evaluation runners match their historical source hashes. Included training reports and protocols agree on the data split and run settings; historical training source byte identity is not established. Read the [audit and remaining dependencies](docs/TRAINING_AND_SOURCE_AUDIT.md) before attempting a GPU run.

## Scope and access

The package supports CPU recalculation of the principal comparisons from included records. It does not bundle a clean training/simulation environment, demonstration images, model checkpoints or video binaries. The detailed historical map identifies external experiment entry points and data dependencies. The optional Colab notebook mounts the existing Drive project rather than uploading this package automatically.

Full training/simulation re-execution has not been independently tested in a fresh runtime. Policy weights and dependencies retain their upstream terms. No new license is asserted for third-party assets. Source/video links may require access to the original Drive; they are not a public anonymous-access guarantee. Historical paths identify experiment artifacts; they are not portable installation paths. See [release review](docs/PUBLIC_RELEASE_REVIEW.md) for the inspected scope and remaining access limitations.

## Reproducibility and release scope

This repository presents a completed single-task evaluation study, including negative fine-tuning results. The recorded-result calculations are reproducible on CPU. Model training, simulation rollouts and video playback are separate from those calculations; no clean end-to-end rerun is claimed.

The repository was assembled on 26 October 2026 and prepared for external review on 27 October 2026. Visibility remains private until the owner elects to publish. Historical provenance documents are retained as dated records, not current access guarantees. See [third-party notices](THIRD_PARTY_NOTICES.md) and [release review](docs/PUBLIC_RELEASE_REVIEW.md).

## Trace storage

The two later JSON archives are losslessly compressed and split into small binary parts under `evidence/later/*.json.gz.parts/`. Each index records part, compressed-stream and original-byte SHA-256 checksums. `analysis/verify_later_traces.py` verifies and decompresses the parts automatically in memory. No manual reconstruction is needed.
