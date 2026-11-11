# Dataset policy

BANKING77 is downloaded from `PolyAI/banking77` at the pinned source commit in `llm-finetuning-and-serving.labels`. Its official train and test partitions are kept separate. LLM Fine-tuning and Serving makes a deterministic, stratified validation split from official train data and stores hashes and provenance in a local manifest. Data and model weights are not checked into Git.

BANKING77 contains intent labels only. It is not ground truth for priority, entity extraction, suggested actions, or human review. Any curated ticket corpus must identify examples as synthetic or public, include annotation source/rationale/review status, use scenario-family splits to limit leakage, and be reviewed before use as a benchmark. Keep customer data out of this repository.

The BANKING77 dataset card identifies CC-BY-4.0 and cites Casanueva et al., “Efficient Intent Detection with Dual Sentence Encoders” (2020), [paper and citation](https://arxiv.org/abs/2003.04807). Preserve that attribution in publications and redistributions.

# Local evaluation fixtures

`unsupported_inputs.jsonl` is a small, hand-authored synthetic guardrail smoke suite. It contains no real customer data and records expected abstention/review outcomes. It is not independently reviewed, statistically representative, or suitable for performance claims. Run it with `llm-finetuning-and-serving eval --input datasets/unsupported_inputs.jsonl --model base`. Reports include the suite category and abstention precision/recall when review labels are present.
