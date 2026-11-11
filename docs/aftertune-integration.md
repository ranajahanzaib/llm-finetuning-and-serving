# AfterTune integration

The local AfterTune implementation exposes the `aftertune` CLI. LLM Fine-tuning and Serving detects that executable, records its version, invokes it with argument arrays (no shell), captures output, applies timeouts, validates comparison schema version `1`, and stores native capability metrics and paired case evidence without changing their meaning.

For an LLM Fine-tuning and Serving adapter, evaluate an existing compatible AfterTune suite using the same base-model config for both the baseline and adapter. The CLI command writes baseline JSONL, adapted JSONL, `comparison.json`, reports, and an LLM Fine-tuning and Serving wrapper containing run ID, source paths, command, version, stdout, and stderr. The original comparison is embedded unchanged. It is reported as unavailable or failed when appropriate; no demo fixture is treated as a real regression run.

AfterTune's synthetic product-extraction task is distinct from banking support. A compatible capability-retention suite and matching base-model configuration must be supplied for an evaluation. Product-extraction scores describe that task and are not banking-support results. When no compatible suite is available, LLM Fine-tuning and Serving reports that no regression evaluation was run.
