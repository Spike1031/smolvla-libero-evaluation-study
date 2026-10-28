# Historical experiment entry points

`08_finetuned_eval_v1.py` is extracted verbatim from code cell 2 of notebook 04: https://colab.research.google.com/drive/1vakHYZiaUQ4iO_Cx_tL_v9jTCbxYlDDo . Its SHA-256 matches the first-round compatibility record; see `tables/source_protocol_audit.json`.

This script is preserved for code inspection. It expects the original Drive project, model exports, original results, GPU dependencies and assets; it is not a standalone fresh-runtime launcher. It creates new attempt outputs and must not be run against an archive merely to read it. Do not change its historical source while claiming byte identity.

The portable CPU entry points remain `analysis/reproduce_results.py` and `analysis/verify_later_traces.py`. Additional original evaluation and training runners/configurations have not yet been individually extracted and packaged. Their external entry-point map remains in `docs/README.md`. A full experiment rerun package is not complete.
