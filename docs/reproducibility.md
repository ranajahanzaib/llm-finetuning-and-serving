# Reproducibility

Dataset preparation accepts a seed and validation fraction. It records the pinned Hugging Face dataset commit, dataset fingerprint, label mapping, split counts, per-file SHA-256, and a warning that official test data is evaluation-only.

Each training run records the base model and snapshot, adapter path, arguments, run ID, training log, duration, adapter size, Git commit, Python/platform, and installed optional dependency versions. Peak memory and parsed validation-loss curves are not recorded by the current implementation.

Evaluation reports include model and dataset revisions, model variant, adapter ID, evaluation-file hash, per-example expected and predicted labels/raw output, per-intent metrics, confusion counts, and latency. Use validation for prompt/configuration decisions and preserve the official test split for the final report. Few-shot uses four stable training examples and is reported separately.
