# SmolVLA: Execution, Occlusion and Adaptation

A single-task empirical study using SmolVLA, LeRobot and LIBERO. The tested fine-tuning interventions did not produce a verified better autonomous policy. The report explains the controlled comparisons, negative results and limits.

## Start here

1. Read [the research report](report/research_report.pdf).
2. Inspect [the main results](tables/summary.csv) and [paired repairs/regressions](tables/transitions.csv).
3. Watch [eight representative videos](videos/README.md).
4. Reproduce the main tables with the command below. No GPU is needed.

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

## Scope and access

The package supports CPU recalculation of the principal comparisons from included records. It does not bundle a clean training/simulation environment, demonstration images, model checkpoints or video binaries. The detailed historical map identifies external experiment entry points and data dependencies. The optional Colab notebook mounts the existing Drive project rather than uploading this package automatically.

Full training/simulation re-execution has not been independently tested in a fresh runtime. Policy weights and dependencies retain their upstream terms. No new license is asserted for third-party assets. Source/video links may require access to the original Drive; they are not a public anonymous-access guarantee. The local evidence includes project paths and should be reviewed before public release.

The 005/014 clean failure observations belong to round 2; both cases succeeded in the original and round 1. This attribution is verified in notebook 08, and corrected in the provenance supplement.

## Delivery status

This is an assembled, unpublished review package dated 26 October 2026. All main table calculations pass. Eight selected video identities were checked against committed paths. The PDF was rendered for layout inspection. No training or simulation was run during assembly; no cloud source file was changed or deleted. GitHub publication and a complete end-to-end rerun are separate steps.

## Trace storage

The two later JSON trace archives are stored as lossless `.json.gz` files. The verification script decompresses them in memory; the original decompressed bytes and reported scores are unchanged.
