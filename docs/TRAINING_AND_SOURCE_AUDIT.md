# Training and source audit

Audit date: 28 October 2026. This is a static source and recorded-metadata audit, not a new GPU run.

## First two training rounds

Both start from the locked original checkpoint; round 2 does not continue round 1. Both use the same disjoint 36-training/9-validation episode split, 45 episodes and 4,487 frames in total. Split and preparation metadata are included under `evidence/training/demonstrations/`. A split within the project does not demonstrate independence from original pretraining.

| Setting | Round 1 | Round 2 |
|---|---|---|
| Learning rate | 1e-5 | 1e-6 |
| Updates | 300 | 150 |
| Selection | Lowest validation loss among trained candidates at 50-step intervals | Fixed update 150 |
| Exported update | 150 | 150 |
| Original validation loss | 0.7056507170200348 | 0.7056507170200348 |
| Exported candidate validation loss | 0.7092655472181462 | 0.7060868166111134 |

The exported round-1 candidate is best among trained candidates, not better than the untrained original on validation loss. Round 2's fixed selection does not mean update 150 has the lowest validation loss. Neither fact establishes autonomous improvement.

Both scripts configure frozen VLM parameters and trainable action-expert/state-projection parameters, AdamW, microbatch 1, accumulation 4 and chunk size 50. No LoRA or reinforcement-learning update is used here. They retain original checkpoint preprocessing rather than replacing normalization with subset statistics.

The packaged protocol JSON equals the protocol embedded in each training report, and both splits equal the packaged split. The current training source files match the documented configuration on inspection; historical training reports do not record a runner source hash, so byte identity to the originally executed training source remains unverified. AST parsing is not an execution test.

## Evaluation sources

All four early evaluation runners now have matching historical runner hashes. The two packaged later evaluation runners also match their historical hashes. In particular, source 18 explicitly continues the same episode after goal attainment until 440 policy actions, sets video fps to 20, and expects 441 saved frames including the final observation. Source 25 checks and imports this core. This differs from the early 80-fps goal-terminated clips. Video duration must always be tied to the specific protocol and actual binary; no new binary validation is claimed.

## Remaining boundaries

- Physical observation/action replay alignment remains unresolved in the historical round-2 protocol. Metadata consistency does not resolve it.
- Original dataset preparation and later finishing-training source chains are not yet complete as a portable execution bundle.
- Weights, original demonstration images, a fully tested environment setup and fresh independent rollouts are not included.
- Earlier ambiguous visual/automatic labels remain listed in `CASE_RECONCILIATION.md`; source recovery does not create missing early per-step measurements.
