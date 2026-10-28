# Historical experiment source map

These sources are preserved for inspection, not launched by the CPU reproduction commands. No training or simulation was run during curation. Script numbers differ from notebook numbers.

| Source | Role | Verification |
|---|---|---|
| `03_paired_eval_v1.py` | Original 1-versus-5 action execution comparison | Matches historical runner hash |
| `04_occlusion_center_v1.py` | Original clean/masked paired evaluation | Matches historical runner hash |
| `08_finetuned_eval_v1.py` | First adapted policy evaluation | Extracted from notebook 04; matches historical runner hash |
| `12_low_lr_fixed150_eval_v1.py` | Second adapted policy evaluation | Matches historical runner hash |
| `06_training_smoke_v1.py` | Training interface/resume smoke test | Current Drive source; no historical runner hash claim |
| `07_task0_finetune_v1.py` | First training round | Current Drive source; protocol/report consistency checked |
| `10_low_lr_fixed150_v1.py` | Second training round | Current Drive source; protocol/report consistency checked |
| `18_finish_policy_eval_v1.py` | Notebook-15 fixed-440-action evaluation and scoring | Matches historical runner hash |
| `25_eval19_four_cases_v1.py` | Notebook-22 model-19 four-case gate | Matches historical runner hash; imports source 18 through the original project path |

Exact hashes and Drive source URLs are in `tables/source_protocol_audit.json`. Training protocols, split metadata, preparation report and training reports are in `evidence/training/`. Their inclusion permits inspection without downloading demonstration images or policy weights.

## Execution dependencies

The original code expects the Colab project root `/content/drive/MyDrive/embodied-smolvla`. Keep the source bytes unchanged when checking historical hashes. Evaluation runners use the original project result layout, selected model exports, pinned policy/tokenizer/assets revisions and the recorded GPU environment. Training runners additionally require prepared demonstration Parquet files, the original audit evaluation plan and a passed smoke-test report. These artifacts are not all bundled here.

Do not copy all sources into one folder and assume their original directory lookups will resolve: the original evaluation scripts live in the project root, while training and later evaluation sources live under its `finetune/` folder. Source 25 dynamically loads source 18 from there and checks its hash. Source 06 is a separate smoke prerequisite for 07/10, not a replacement for dataset preparation.

## Read before running

The scripts create attempts, checkpoints and result files, acquire locks and download model assets. They have not been validated as a clean fresh-runtime end-to-end workflow. Use `analysis/reproduce_results.py` for CPU main-table reproduction and `analysis/verify_later_traces.py` for trace scoring. Do not run historical scripts against an immutable evidence archive merely to inspect it.

Later finishing-training variants and demonstration-generation scripts are still external; the packaged later evaluator and traces do not imply the entire later training pipeline is reproduced. Model checkpoints and source demonstration images are intentionally absent.
