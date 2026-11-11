# Limitations and remaining work

- No model download, real MLX fine-tune, or real evaluation has been executed in this checkout. No adapter or performance results are claimed.
- Structured ticket labels (priority, entities, advisory action, review) need a separately authored and reviewed corpus. The current implementation does not invent one or report those metrics.
- Zero-shot, four-example few-shot, and adapted intent evaluations are implemented. Few-shot is not a substitute for a comprehensive comparison study.
- Training records dependency versions, Git commit, base model snapshot, data hashes, configuration, duration, adapter size, and a post-training inference check. Peak memory and parsed training/validation-loss curves are not recorded or shown.
- AfterTune execution requires an appropriate held-out suite and config matched to the base checkpoint and adapter. Its product-extraction data is not a banking benchmark.
- `datasets/unsupported_inputs.jsonl` provides 18 synthetic unsupported-input and safety smoke cases, with review/abstention metrics available through custom-suite evaluation. It has not had independent annotation review, is too small for performance claims, and does not validate calibrated abstention or provide a robust OOD detector.
- The development and standard training presets are starting points. Validate CLI compatibility, memory, and convergence on the target Mac.
- The API tests, frontend test, and production build pass from the locked dependencies. The production chart chunk is 524 kB before gzip (157 kB gzipped), which Vite reports as above its default 500 kB chunk warning threshold.
- Native MLX training and real-model evaluation remain unverified because this environment does not provide an available Apple Metal device.
